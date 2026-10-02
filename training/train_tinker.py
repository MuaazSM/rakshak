"""LoRA SFT of the detector on Tinker with run hygiene (PRD FR-33, §9.1, §11.1–11.3).

    uv run --extra train python -m training.train_tinker --dry-run            # tokens + $ only
    uv run --extra train python -m training.train_tinker --limit 50 --epochs 1 --batch 8 --run-name smoke
    uv run --extra train python -m training.train_tinker --run-name full     # 3 epochs, full train
    uv run --extra train python -m training.train_tinker --spend 2026-10-03  # Tinker billing usage

APIs (verified against tinker 0.32.0 / tinker-cookbook 0.5.7 sources in .venv):
  tinker.ServiceClient(api_key).create_lora_training_client(base_model, rank, seed, user_metadata)
  renderers.get_renderer("qwen3_5_disable_thinking", tokenizer)   # matches HF enable_thinking=False
  supervised.data.conversation_to_datum(msgs, renderer, max_len, TrainOnWhat.LAST_ASSISTANT_MESSAGE)
      -> loss only on the assistant JSON + <|im_end|>; reduction "mean" (cookbook SFT default)
  TrainingClient.forward_backward(batch, "cross_entropy"); .optim_step(tinker.AdamParams(...))
  supervised.common.compute_mean_nll(logprobs, weights)                # train loss per step
  utils.lr_scheduling.compute_schedule_lr_multiplier("linear", step, total)   # cookbook default
  hyperparam_utils.get_lr(base)                                       # default LR (§11.1)
  TrainingClient.save_weights_for_sampler(name, ttl_seconds).result().path   # tinker://...
  TrainingClient.create_sampling_client(path) -> SamplingClient.sample_async (via eval.systems)
  RestClient.get_billing_usage(start, end)                            # actual spend (lags ≤ hours)

Each epoch: save a sampler checkpoint, evaluate dev greedily through the Tinker sampler with
eval/metrics, log; the best epoch is chosen by dev macro-F1 then scam recall. Writes
`training/runs/{run_id}/config.json` (§11.3) and `metrics.json`, plus per-epoch dev results
without message text. Only redacted/synthetic split files may be used (CLAUDE.md); the test
split is refused.
"""

import argparse
import asyncio
import json
import random
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from eval.metrics import selection_key
from eval.run_eval import build_result, git_sha, sha256_file, write_result
from eval.systems import TINKER_RENDERER, Item, item_from_row, run_system
from hub.settings import get_settings

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
RUNS = ROOT / "training" / "runs"

# PRD §11.2: Tinker training price for Qwen3.5-4B.
TRAIN_USD_PER_M = 0.737
# Sampling price is not in the SDK; the billing API reports the effective rate after the fact.
# Assumption (conservative): prefill and sampled tokens cost no more than training tokens.
SAMPLE_USD_PER_M = 0.737
DEV_OUTPUT_SLACK = 16  # tuned replies ≈ gold target length; slack for the eos/think tokens
CHECKPOINT_TTL_DAYS = 30
ADAM = {"beta1": 0.9, "beta2": 0.95, "eps": 1e-8}  # cookbook supervised defaults


class TrainError(Exception):
    pass


# --- data ---


def guard_split(path: Path, role: str) -> None:
    """Never train or select on the test set (CLAUDE.md hard rules, FR-32)."""
    p = path.resolve()
    if p.name == "test.jsonl" or p == (SPLITS / "test.jsonl").resolve():
        raise TrainError(f"--{role} must not be the test split")
    if not p.exists():
        raise TrainError(f"{path} does not exist")


def load_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]


def select_rows(rows: list[dict], limit: int | None, seed: int) -> list[dict]:
    """Seeded shuffle, then the first `limit` (so a smoke run sees a class mix)."""
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def epoch_batches(n: int, batch: int, epoch: int, seed: int) -> list[list[int]]:
    """Shuffled index batches for one epoch; the last partial batch is kept."""
    idx = list(range(n))
    random.Random(seed * 1000 + epoch).shuffle(idx)
    return [idx[i : i + batch] for i in range(0, n, batch)]


# --- cost ---


def estimate_cost(
    train_tokens_per_epoch: int,
    epochs: int,
    dev_prompt_tokens: int,
    dev_sample_tokens: int,
    dev_evals: int,
) -> dict:
    train_tokens = train_tokens_per_epoch * epochs
    sample_tokens = (dev_prompt_tokens + dev_sample_tokens) * dev_evals
    train_usd = train_tokens * TRAIN_USD_PER_M / 1e6
    sample_usd = sample_tokens * SAMPLE_USD_PER_M / 1e6
    return {
        "train_tokens": train_tokens,
        "dev_eval_prompt_tokens": dev_prompt_tokens * dev_evals,
        "dev_eval_sample_tokens": dev_sample_tokens * dev_evals,
        "train_usd": round(train_usd, 4),
        "dev_eval_usd": round(sample_usd, 4),
        "total_usd": round(train_usd + sample_usd, 4),
        "assumptions": (
            f"training ${TRAIN_USD_PER_M}/M tokens on full sequence length (PRD §11.2); "
            f"sampling assumed ≤ ${SAMPLE_USD_PER_M}/M for prefill and sampled tokens "
            "(not published in the SDK); checkpoint storage not included"
        ),
    }


def count_tokens(train_rows: list[dict], dev_items: list[Item], renderer, max_len: int) -> dict:
    """Training tokens per epoch (truncated sequence length) and dev-eval token estimates."""
    from tinker_cookbook.renderers import TrainOnWhat

    train = 0
    trained = 0
    truncated = 0
    for r in train_rows:
        mi, w = renderer.build_supervised_example(
            r["messages"], train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )
        train += min(mi.length, max_len)
        trained += int((w > 0).sum())
        truncated += mi.length > max_len
    prompt = sample = 0
    for it in dev_items:
        prompt += renderer.build_generation_prompt(it.prompt_messages).length
        target = next(m["content"] for m in it.messages if m["role"] == "assistant")
        sample += len(renderer.tokenizer.encode(target, add_special_tokens=False))
        sample += DEV_OUTPUT_SLACK
    return {
        "train_tokens_per_epoch": train,
        "assistant_tokens_per_epoch": trained,
        "truncated_examples": truncated,
        "dev_prompt_tokens": prompt,
        "dev_sample_tokens": sample,
    }


def billing_usage(since: datetime, until: datetime | None = None) -> dict:
    """Actual Tinker usage from the billing API (estimated gross USD; lags up to a few hours)."""
    import tinker

    s = get_settings()
    key = s.tinker_api_key.get_secret_value() if s.tinker_api_key else None
    rest = tinker.ServiceClient(api_key=key).create_rest_client()
    start = since.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    end = (until or datetime.now(UTC)).astimezone(UTC).replace(
        minute=0, second=0, microsecond=0
    ) + timedelta(hours=1)
    by_type: dict[str, dict] = {}
    total = 0.0
    unpriced = 0
    cursor = start
    while cursor < end:  # the API allows at most 14 days per call
        stop = min(end, cursor + timedelta(days=14))
        resp = rest.get_billing_usage(
            cursor.isoformat().replace("+00:00", "Z"), stop.isoformat().replace("+00:00", "Z")
        ).result()
        for e in resp.data:
            info = e.event_info
            row = by_type.setdefault(info.type, {"tokens": 0, "usd": 0.0, "rates": set()})
            row["tokens"] += getattr(info, "token_count", 0) or 0
            if e.estimated_cost_usd is None:
                unpriced += 1
            else:
                row["usd"] += e.estimated_cost_usd
                total += e.estimated_cost_usd
            if e.effective_rate_usd_per_million_tokens is not None:
                row["rates"].add(e.effective_rate_usd_per_million_tokens)
        cursor = stop
    for row in by_type.values():
        row["rates"] = sorted(row["rates"])
        row["usd"] = round(row["usd"], 4)
    return {
        "since": start.isoformat(),
        "until": end.isoformat(),
        "total_usd": round(total, 4),
        "unpriced_events": unpriced,
        "by_type": by_type,
    }


# --- training ---


async def eval_dev(sampling_client, dev_items: list[Item]):
    from eval.systems import TunedTinker

    system = TunedTinker(sampling_client=sampling_client)
    preds = await run_system(system, dev_items)
    return preds, system.prompt_tokens, system.sampled_tokens


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", "utf-8")


def train(args: argparse.Namespace) -> int:
    guard_split(args.train, "train")
    guard_split(args.dev, "dev")
    if args.train.resolve() == args.dev.resolve():
        raise TrainError("--train and --dev are the same file")

    import tinker
    from tinker_cookbook import hyperparam_utils, renderers
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised.common import compute_mean_nll
    from tinker_cookbook.supervised.data import conversation_to_datum
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier

    s = get_settings()
    base = args.base or s.detector_base
    lr = args.lr or hyperparam_utils.get_lr(base)

    train_rows = select_rows(load_rows(args.train), args.limit, args.seed)
    dev_items = [item_from_row(r) for r in load_rows(args.dev)]
    if args.dev_limit:
        dev_items = dev_items[: args.dev_limit]
    renderer = renderers.get_renderer(TINKER_RENDERER, get_tokenizer(base))
    counts = count_tokens(train_rows, dev_items, renderer, args.max_len)
    est = estimate_cost(
        counts["train_tokens_per_epoch"],
        args.epochs,
        counts["dev_prompt_tokens"],
        counts["dev_sample_tokens"],
        args.epochs,
    )
    n_steps = len(epoch_batches(len(train_rows), args.batch, 0, args.seed)) * args.epochs
    print(
        f"base {base}  renderer {TINKER_RENDERER}  rank {args.rank}  lr {lr:.3g}  "
        f"batch {args.batch}  epochs {args.epochs}  steps {n_steps}"
    )
    print(
        f"train {len(train_rows)} ex, {counts['train_tokens_per_epoch']:,} tok/epoch "
        f"({counts['assistant_tokens_per_epoch']:,} trained, {counts['truncated_examples']} "
        f"truncated at {args.max_len}); dev {len(dev_items)} ex"
    )
    print(
        f"ESTIMATED COST ${est['total_usd']:.3f} = training ${est['train_usd']:.3f} "
        f"({est['train_tokens']:,} tok × ${TRAIN_USD_PER_M}/M) + dev eval ${est['dev_eval_usd']:.3f} "
        f"({args.epochs} × {counts['dev_prompt_tokens'] + counts['dev_sample_tokens']:,} tok, "
        f"assumed ≤ ${SAMPLE_USD_PER_M}/M)"
    )
    if args.dry_run:
        return 0
    if est["total_usd"] > args.max_cost:
        raise TrainError(f"estimate ${est['total_usd']:.2f} exceeds --max-cost ${args.max_cost}")

    started = datetime.now(UTC)
    run_id = f"{started:%Y%m%d-%H%M%S}-{args.run_name}"
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    lock = args.train.parent / "splits.lock.json"
    config = {
        "run_id": run_id,
        "status": "running",
        "started": started.isoformat(timespec="seconds"),
        "base_model": base,
        "renderer": TINKER_RENDERER,
        "method": "LoRA SFT, loss on the final assistant message only (TrainOnWhat.LAST_ASSISTANT_MESSAGE, mean reduction)",
        "hyperparams": {
            "rank": args.rank,
            "lr": lr,
            "lr_default_get_lr": args.lr is None,
            "lr_schedule": "linear",
            "batch": args.batch,
            "epochs": args.epochs,
            "steps": n_steps,
            "max_len": args.max_len,
            "seed": args.seed,
            "limit": args.limit,
            "dev_limit": args.dev_limit,
            "adam": ADAM,
            "eval_decoding": "greedy, temperature 0, max_tokens 512",
        },
        "data": {
            "train": {
                "path": str(args.train),
                "sha256": sha256_file(args.train),
                "n_used": len(train_rows),
            },
            "dev": {"path": str(args.dev), "sha256": sha256_file(args.dev), "n": len(dev_items)},
            "splits_lock_sha256": sha256_file(lock) if lock.exists() else None,
        },
        "git_sha": git_sha(),
        "sdk": _versions(),
        "token_counts": counts,
        "cost_estimate": est,
        "checkpoints": [],
        "best": None,
        "actual_cost": None,
    }
    write_json(run_dir / "config.json", config)

    service = tinker.ServiceClient(
        api_key=s.tinker_api_key.get_secret_value() if s.tinker_api_key else None
    )
    tc = service.create_lora_training_client(
        base_model=base, rank=args.rank, seed=args.seed, user_metadata={"run_id": run_id}
    )
    datums = [
        conversation_to_datum(
            r["messages"], renderer, args.max_len, TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )
        for r in train_rows
    ]
    log = (run_dir / "log.jsonl").open("a", encoding="utf-8")
    epochs_out: list[dict] = []
    best: dict | None = None
    step = 0
    sample_tokens = {"prompt": 0, "sampled": 0}
    try:
        for epoch in range(args.epochs):
            losses = []
            for batch_idx in epoch_batches(len(datums), args.batch, epoch, args.seed):
                batch = [datums[i] for i in batch_idx]
                step_lr = lr * compute_schedule_lr_multiplier("linear", step, n_steps)
                t0 = time.perf_counter()
                fb = tc.forward_backward(batch, "cross_entropy")
                opt = tc.optim_step(tinker.AdamParams(learning_rate=step_lr, **ADAM))
                res = fb.result()
                opt.result()
                loss = compute_mean_nll(
                    [o["logprobs"] for o in res.loss_fn_outputs],
                    [d.loss_fn_inputs["weights"] for d in batch],
                )
                losses.append(loss)
                rec = {
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "loss": round(loss, 5),
                    "lr": step_lr,
                    "n": len(batch),
                    "sec": round(time.perf_counter() - t0, 2),
                }
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(
                    f"epoch {epoch + 1} step {step + 1}/{n_steps} loss {loss:.4f} lr {step_lr:.2e}"
                )
                step += 1

            name = f"{run_id}-e{epoch + 1}"
            path = (
                tc.save_weights_for_sampler(name=name, ttl_seconds=CHECKPOINT_TTL_DAYS * 86400)
                .result()
                .path
            )
            sampler = tc.create_sampling_client(path)
            t0 = time.perf_counter()
            preds, ptok, stok = asyncio.run(eval_dev(sampler, dev_items))
            sample_tokens["prompt"] += ptok
            sample_tokens["sampled"] += stok
            result = build_result(
                f"tuned_tinker_e{epoch + 1}",
                "dev",
                args.dev,
                dev_items,
                preds,
                {"checkpoint": path, "run_id": run_id},
            )
            write_result(result, dev_items, preds, run_dir)
            m = result["metrics"]
            ep = {
                "epoch": epoch + 1,
                "checkpoint": path,
                "train_loss_first": losses[0],
                "train_loss_last": losses[-1],
                "train_loss_mean": sum(losses) / len(losses),
                "dev": {
                    k: m[k]
                    for k in (
                        "macro_f1",
                        "scam_recall",
                        "scam_recall_strict",
                        "fpr_genuine",
                        "fpr_hard_negative",
                        "category_accuracy",
                        "span_f1",
                        "grounding_rate",
                        "json_validity",
                        "json_parse_rate",
                        "unknown_rate",
                        "errors",
                    )
                },
                "dev_ci95": result["ci95"],
                "dev_eval_sec": round(time.perf_counter() - t0, 1),
            }
            epochs_out.append(ep)
            config["checkpoints"].append({"epoch": epoch + 1, "sampler_path": path})
            if best is None or selection_key(m) > selection_key(best["dev"]):
                best = {"epoch": epoch + 1, "checkpoint": path, "dev": ep["dev"]}
            config["best"] = best
            write_json(run_dir / "config.json", config)
            write_json(
                run_dir / "metrics.json",
                {
                    "provisional": result["provisional"],
                    "split": result["split"],
                    "epochs": epochs_out,
                    "best": best,
                },
            )
            d = ep["dev"]
            print(
                f"epoch {epoch + 1}: loss {ep['train_loss_first']:.4f} → {ep['train_loss_last']:.4f}; "
                f"dev macro-F1 {_f(d['macro_f1'])} recall {_f(d['scam_recall'])} "
                f"FPR {_f(d['fpr_genuine'])} JSON {_f(d['json_validity'])} → {path}"
                + ("  [PROVISIONAL synthetic dev]" if result["provisional"] else "")
            )
        config["status"] = "done"
    except BaseException:
        config["status"] = "failed"
        raise
    finally:
        log.close()
        config["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
        config["sampling_tokens_actual"] = sample_tokens
        config["cost_estimate_actual_tokens"] = estimate_cost(
            counts["train_tokens_per_epoch"],
            len(epochs_out) or 1,
            sample_tokens["prompt"],
            sample_tokens["sampled"],
            1,
        )
        write_json(run_dir / "config.json", config)
    print(f"best epoch {best['epoch']}: {best['checkpoint']}")
    print(f"run dir {run_dir.relative_to(ROOT)}")
    return 0


def _f(v: float | None) -> str:
    return "—" if v is None else f"{v:.3f}"


def _versions() -> dict:
    from importlib.metadata import version

    return {p: version(p) for p in ("tinker", "tinker-cookbook", "transformers")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", type=Path, default=SPLITS / "train.jsonl")
    ap.add_argument("--dev", type=Path, default=SPLITS / "dev.jsonl")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="default: hyperparam_utils.get_lr")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None, help="train examples (smoke runs)")
    ap.add_argument("--dev-limit", type=int, default=None)
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--base", default=None, help="default: DETECTOR_BASE")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-cost", type=float, default=3.0, help="abort if estimate exceeds ($)")
    ap.add_argument("--dry-run", action="store_true", help="count tokens, print estimate, exit")
    ap.add_argument("--spend", metavar="SINCE", help="print Tinker billing usage since a date")
    args = ap.parse_args(argv)
    try:
        if args.spend:
            since = datetime.fromisoformat(args.spend)
            since = since if since.tzinfo else since.replace(tzinfo=UTC)
            print(json.dumps(billing_usage(since), indent=2))
            return 0
        return train(args)
    except TrainError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
