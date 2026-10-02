"""Export parity: served GGUF vs Tinker sampler on dev (PRD §11.4, FR-34).

    uv run --extra train python -m eval.parity --run-id 20261003-0612-full
    uv run --extra train python -m eval.parity --checkpoint tinker://... --url http://127.0.0.1:8081/v1

The GGUF side runs `eval.systems.TunedDetector` (hub.detector.detect against llama-server); the
Tinker side runs `eval.systems.TunedTinker` on the same checkpoint (greedy). With --run-id and
no --checkpoint, the Tinker predictions are reused from the run dir's per-epoch dev items
(`tuned_tinker_e{N}_dev.items.jsonl` of the best epoch) when they cover every dev item, so no
Tinker sampling is paid for twice; --resample forces a fresh Tinker pass.

Reports verdict agreement (target ≥ 0.95), category agreement, JSON validity on both sides and
the verdict confusion (Tinker × GGUF). Writes eval/results/parity_dev.json; disagreeing items
are listed by id only. Never prints message text. Refuses any split but dev.
"""

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from eval.run_eval import git_sha, is_provisional, sha256_file
from eval.systems import Item, load_items, run_system

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "data" / "splits" / "dev.jsonl"
RUNS = ROOT / "training" / "runs"
RESULTS = ROOT / "eval" / "results"
TARGET = 0.95
VERDICTS = ("SCAM", "SUSPICIOUS", "SAFE", "UNKNOWN")


class ParityError(Exception):
    pass


# --- pure ---


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def parity(ids: list[str], ref: list[dict], served: list[dict]) -> dict:
    """Agreement between two prediction lists of dicts with pred_verdict / pred_category /
    json_valid / error (the run_eval items.jsonl shape). `ref` = Tinker, `served` = GGUF."""
    if not (len(ids) == len(ref) == len(served)):
        raise ParityError("prediction lists differ in length")
    n = len(ids)
    same = [r["pred_verdict"] == s["pred_verdict"] for r, s in zip(ref, served, strict=True)]
    both_ok = [
        i
        for i, (r, s) in enumerate(zip(ref, served, strict=True))
        if not r.get("error") and not s.get("error")
    ]
    cat_pairs = [
        (r, s)
        for r, s in zip(ref, served, strict=True)
        if r["pred_verdict"] == s["pred_verdict"] and r["pred_verdict"] in ("SCAM", "SUSPICIOUS")
    ]
    confusion = {a: dict.fromkeys(VERDICTS, 0) for a in VERDICTS}
    for r, s in zip(ref, served, strict=True):
        confusion[_v(r)][_v(s)] += 1

    def validity(rows: list[dict]) -> float | None:
        vals = [r["json_valid"] for r in rows if r.get("json_valid") is not None]
        return _rate(sum(vals), len(vals))

    agreement = _rate(sum(same), n)
    return {
        "n": n,
        "verdict_agreement": agreement,
        "verdict_agreement_no_errors": _rate(sum(same[i] for i in both_ok), len(both_ok)),
        "category_agreement_on_agreed_scams": _rate(
            sum(r.get("pred_category") == s.get("pred_category") for r, s in cat_pairs),
            len(cat_pairs),
        ),
        "json_validity_tinker": validity(ref),
        "json_validity_gguf": validity(served),
        "errors_tinker": sum(bool(r.get("error")) for r in ref),
        "errors_gguf": sum(bool(s.get("error")) for s in served),
        "confusion_tinker_x_gguf": confusion,
        "disagreements": [i for i, ok in zip(ids, same, strict=True) if not ok],
        "target": TARGET,
        "pass": agreement is not None and agreement >= TARGET,
    }


def _v(row: dict) -> str:
    return row["pred_verdict"] if row["pred_verdict"] in VERDICTS else "UNKNOWN"


def pred_row(p) -> dict:
    return {
        "pred_verdict": p.verdict,
        "pred_category": p.category,
        "json_valid": p.json_valid,
        "error": p.error,
    }


def best_epoch_items(run_dir: Path) -> tuple[str, Path]:
    """(checkpoint, items path) of the best epoch recorded by train_tinker."""
    cfg = json.loads((run_dir / "config.json").read_text("utf-8"))
    best = cfg.get("best") or {}
    if not best.get("checkpoint"):
        raise ParityError(f"{run_dir}/config.json has no best checkpoint")
    return best["checkpoint"], run_dir / f"tuned_tinker_e{best['epoch']}_dev.items.jsonl"


def reuse_rows(path: Path, ids: list[str]) -> list[dict] | None:
    """Rows in dev order if the cached items file covers every id, else None."""
    if not path.exists():
        return None
    rows = {}
    for line in path.read_text("utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["id"]] = r
    if not all(i in rows for i in ids):
        return None
    return [rows[i] for i in ids]


def use_url(url: str | None) -> str:
    """Point hub.detector at --url (settings are cached; env wins over .env)."""
    from hub.settings import get_settings

    if url:
        os.environ["DETECTOR_URL"] = url
        get_settings.cache_clear()
    return get_settings().detector_url


# --- runs ---


async def _predict(system, items: list[Item]) -> list[dict]:
    try:
        return [pred_row(p) for p in await run_system(system, items)]
    finally:
        await system.aclose()


def run_gguf(items: list[Item]) -> list[dict]:
    from eval.systems import TunedDetector

    return asyncio.run(_predict(TunedDetector(), items))


def run_tinker(checkpoint: str, items: list[Item]) -> list[dict]:
    from eval.systems import TunedTinker

    return asyncio.run(_predict(TunedTinker(checkpoint), items))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--run-id", help="training/runs/{run_id} (best epoch checkpoint + cached preds)"
    )
    ap.add_argument("--checkpoint", help="tinker://... (forces fresh Tinker sampling)")
    ap.add_argument("--url", help="llama-server base URL (default settings.detector_url)")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--dev", type=Path, default=DEV)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resample", action="store_true", help="ignore cached Tinker predictions")
    ap.add_argument("--out", type=Path, default=RESULTS / "parity_dev.json")
    args = ap.parse_args(argv)
    try:
        if args.split != "dev" or args.dev.name == "test.jsonl":
            raise ParityError("parity runs on dev only")
        if not args.run_id and not args.checkpoint:
            raise ParityError("give --run-id or --checkpoint")
        if not args.dev.exists():
            raise ParityError(f"{args.dev} does not exist")
        items = load_items(args.dev, args.limit)
        ids = [it.id for it in items]

        checkpoint, cached, source = args.checkpoint, None, "tinker_sampler"
        if args.run_id:
            ck, items_path = best_epoch_items(RUNS / args.run_id)
            checkpoint = checkpoint or ck
            if not args.checkpoint and not args.resample:
                cached = reuse_rows(items_path, ids)
                if cached is not None:
                    source = f"cached:{items_path.relative_to(ROOT)}"
        url = use_url(args.url)
        print(f"GGUF side: {url}  ({len(items)} dev items)", flush=True)
        served = run_gguf(items)
        print(f"Tinker side: {checkpoint} ({source})", flush=True)
        ref = cached if cached is not None else run_tinker(checkpoint, items)
        rep = parity(ids, ref, served)
    except ParityError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2

    from hub.settings import get_settings

    provisional = is_provisional(items)
    result = {
        "provisional": provisional,
        "split": "dev_synthetic" if provisional else "dev",
        "checkpoint": checkpoint,
        "tinker_source": source,
        "detector_url": url,
        "detector_version": get_settings().detector_version,
        "limit": args.limit,
        "data": {
            "path": str(args.dev.relative_to(ROOT))
            if args.dev.is_relative_to(ROOT)
            else str(args.dev),
            "sha256": sha256_file(args.dev),
        },
        "git_sha": git_sha(),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        **rep,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    tag = "  [PROVISIONAL synthetic dev]" if provisional else ""
    fmt = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
    print(
        f"{'PASS' if rep['pass'] else 'FAIL'}  verdict agreement {fmt(rep['verdict_agreement'])} "
        f"(target ≥ {TARGET}; {len(rep['disagreements'])} of {rep['n']} differ){tag}\n"
        f"      JSON validity tinker {fmt(rep['json_validity_tinker'])} / gguf "
        f"{fmt(rep['json_validity_gguf'])}; errors tinker {rep['errors_tinker']} / gguf "
        f"{rep['errors_gguf']}; category agreement {fmt(rep['category_agreement_on_agreed_scams'])}\n"
        f"      verdicts tinker×gguf {dict(Counter((_v(r), _v(s)) for r, s in zip(ref, served, strict=True) if _v(r) != _v(s)))}\n"
        f"      → {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}"
    )
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
