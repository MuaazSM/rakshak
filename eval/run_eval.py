"""Eval harness: every ablation system, metrics, results JSON (PRD FR-35, §12.1, §12.3, §12.4).

    uv run python -m eval.run_eval --systems rules_only,gemma_zeroshot,qwen_base_fewshot --split dev
    uv run --extra train python -m eval.run_eval --systems tuned_tinker --checkpoint tinker://...
    uv run python -m eval.run_eval --table            # print the table from existing results

Writes `eval/results/{system}_{split}.json` (metrics, 95% bootstrap CIs, §12.4 slices, latency,
provenance) and `eval/results/{system}_{split}.items.jsonl` (per-item labels, verdicts, p_scam;
no message text — calibrate.py reads p_scam from it). Deterministic: file order, greedy
decoding, fixed seeds. Never prints message text.

If every item in the split is synthetic the results are marked `"provisional": true` with
split name `{split}_synthetic`, and printed tables say PROVISIONAL.

`--split test` is refused unless `--i-am-the-final-eval` is given AND
`data/splits/test.jsonl` exists (CLAUDE.md: the test set is used once, at the very end).
"""

import argparse
import asyncio
import hashlib
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from eval.bootstrap import headline_cis
from eval.metrics import Gold, Pred, compute
from eval.systems import (
    SYSTEMS,
    Item,
    gold_of,
    load_items,
    run_system,
    served_detector_is_tuned,
    served_models,
    strip_sender,
)
from hub.settings import get_settings

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
RESULTS = ROOT / "eval" / "results"
SLICE_KEYS = ("language", "category", "obfuscated", "source", "source_phone")
SLICE_METRICS = (
    "n",
    "scam_recall",
    "scam_recall_strict",
    "fpr_genuine",
    "fpr_hard_negative",
    "macro_f1",
    "category_accuracy",
    "json_validity",
)


class EvalRefused(Exception):
    pass


# --- provenance (no git subprocess: read .git directly) ---


def git_sha(root: Path = ROOT) -> str | None:
    """HEAD commit read from .git files (no git command is run)."""
    git = root / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        loose = git / ref
        if loose.exists():
            return loose.read_text().strip()
        for line in (git / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except OSError:
        return None
    return None


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_path(split: str, final_eval: bool, splits_dir: Path = SPLITS) -> Path:
    """Resolve a split file, guarding the test set."""
    if split == "test":
        test = splits_dir / "test.jsonl"
        if not final_eval:
            raise EvalRefused("--split test needs --i-am-the-final-eval (test is used once)")
        if not test.exists():
            raise EvalRefused(f"{test} does not exist; there is no frozen test set")
        return test
    if split not in ("train", "dev"):
        raise EvalRefused(f"unknown split {split!r}")
    return splits_dir / f"{split}.jsonl"


def is_provisional(items: list[Item]) -> bool:
    return bool(items) and all(it.meta.get("source") == "synthetic" for it in items)


# --- results ---


def slice_metrics(items: list[Item], golds: list[Gold], preds: list[Pred]) -> dict:
    """§12.4: per language, category, obfuscated, source, source_phone."""
    out: dict = {}
    for key in SLICE_KEYS:
        groups: dict[str, list[int]] = defaultdict(list)
        for i, it in enumerate(items):
            value = it.gold.category if key == "category" else it.meta.get(key)
            if value is None:
                continue
            groups[str(value).lower()].append(i)
        if not groups:
            continue
        out[key] = {}
        for value in sorted(groups):
            idx = groups[value]
            m = compute([golds[i] for i in idx], [preds[i] for i in idx])
            out[key][value] = {k: m[k] for k in SLICE_METRICS}
    return out


def item_rows(items: list[Item], golds: list[Gold], preds: list[Pred]) -> list[dict]:
    """Per-item record without any message text or quotes."""
    return [
        {
            "id": it.id,
            "gold_verdict": g.verdict,
            "gold_category": g.category,
            "hard_negative": g.hard_negative,
            "pred_verdict": p.verdict,
            "pred_category": p.category,
            "p_scam": p.p_scam,
            "json_valid": p.json_valid,
            "n_flags": len(p.quotes),
            "n_grounded": sum(bool(q) and q in g.text for q in p.quotes),
            "latency_ms": round(p.latency_ms, 1) if p.latency_ms is not None else None,
            "error": p.error,
        }
        for it, g, p in zip(items, golds, preds, strict=True)
    ]


def _rel(path: Path) -> str:
    """Repo-relative path when possible (accepts relative or absolute input)."""
    p = Path(path).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def build_result(
    system: str,
    split: str,
    path: Path,
    items: list[Item],
    preds: list[Pred],
    extra: dict | None = None,
) -> dict:
    golds = [gold_of(it) for it in items]
    provisional = is_provisional(items)
    return {
        "system": system,
        "split": f"{split}_synthetic" if provisional else split,
        "provisional": provisional,
        "n": len(items),
        "metrics": compute(golds, preds),
        "ci95": headline_cis(golds, preds),
        "slices": slice_metrics(items, golds, preds),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "data": {"path": _rel(path), "sha256": sha256_file(path)},
        **(extra or {}),
    }


def write_result(
    result: dict, items: list[Item], preds: list[Pred], out_dir: Path, suffix: str = ""
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{result['system']}_{result['split'].removesuffix('_synthetic')}{suffix}"
    path = out_dir / f"{stem}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", "utf-8")
    golds = [gold_of(it) for it in items]
    rows = item_rows(items, golds, preds)
    (out_dir / f"{stem}.items.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8"
    )
    return path


# --- table ---


def _fmt(v: float | None, ci: list[float] | None = None) -> str:
    if v is None:
        return "—"
    s = f"{v:.3f}"
    return f"{s} [{ci[0]:.2f}–{ci[1]:.2f}]" if ci else s


def format_table(results: list[dict]) -> str:
    provisional = any(r.get("provisional") for r in results)
    head = "PROVISIONAL (synthetic dev) — " if provisional else ""
    lines = [
        f"{head}split={','.join(sorted({r['split'] for r in results}))}",
        "| System | n | Scam recall | Strict recall | FPR genuine | FPR hard-neg | Macro-F1 "
        "| Cat acc | Span F1 | Grounding | JSON valid | p50 / p95 ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        m, ci = r["metrics"], r.get("ci95", {})
        lat = m["latency"]
        p50 = f"{lat['p50_ms']:.0f}" if lat["p50_ms"] is not None else "—"
        p95 = f"{lat['p95_ms']:.0f}" if lat["p95_ms"] is not None else "—"
        lines.append(
            f"| {r['system']} | {r['n']} "
            f"| {_fmt(m['scam_recall'], ci.get('scam_recall'))} "
            f"| {_fmt(m['scam_recall_strict'])} "
            f"| {_fmt(m['fpr_genuine'], ci.get('fpr_genuine'))} "
            f"| {_fmt(m['fpr_hard_negative'], ci.get('fpr_hard_negative'))} "
            f"| {_fmt(m['macro_f1'], ci.get('macro_f1'))} "
            f"| {_fmt(m['category_accuracy'])} | {_fmt(m['span_f1'])} "
            f"| {_fmt(m['grounding_rate'])} | {_fmt(m['json_validity'])} | {p50} / {p95} |"
        )
    return "\n".join(lines)


# --- main ---


def make_system(name: str, args: argparse.Namespace):
    if name not in SYSTEMS:
        raise EvalRefused(f"unknown system {name!r}; choose from {sorted(SYSTEMS)}")
    if name == "tuned_tinker":
        if not args.checkpoint:
            raise EvalRefused("tuned_tinker needs --checkpoint tinker://...")
        return SYSTEMS[name](args.checkpoint)
    return SYSTEMS[name]()


DETECTOR_SYSTEMS = ("qwen_base_fewshot", "tuned_detector", "full_system")


def detector_provenance(name: str) -> dict:
    """Which detector a llama-server-backed system scored (PRD §11.3 hygiene)."""
    if name not in DETECTOR_SYSTEMS:
        return {}
    s = get_settings()
    served = served_models()
    out: dict = {"detector_url": s.detector_url, "served_models": served}
    if name != "qwen_base_fewshot":
        out["detector_version"] = s.detector_version
        out["detector_version_matches_served"] = any(s.detector_version in m for m in served)
    return out


async def evaluate_system(name: str, args: argparse.Namespace, path: Path, items: list[Item]):
    system = make_system(name, args)
    try:
        try:
            system.check_items(items)
        except ValueError as e:
            raise EvalRefused(str(e)) from e
        extra: dict = {"limit": args.limit, **detector_provenance(name)}
        preds = await run_system(system, items)
    finally:
        await system.aclose()
    if name == "tuned_tinker":
        extra["checkpoint"] = args.checkpoint
    if args.strip_sender:
        extra["variant"] = "nosender"
        extra["variant_note"] = "SENDER forced to unknown, RULE_SIGNALS recomputed with sender None"
    result = build_result(name, args.split, path, items, preds, extra)
    out = write_result(
        result, items, preds, args.out_dir, suffix="_nosender" if args.strip_sender else ""
    )
    return result, out


BASELINES = ("rules_only", "gemma_zeroshot", "qwen_base_fewshot")


def runnable_systems(checkpoint: str | None, detector_tuned: bool) -> list[str]:
    """`--all`: the baselines, tuned_tinker when a checkpoint is given, and tuned_detector +
    full_system only once llama-server serves a tuned rakshak-detector GGUF (otherwise they
    would silently measure the base model)."""
    names = list(BASELINES)
    if checkpoint:
        names.append("tuned_tinker")
    if detector_tuned:
        names += ["tuned_detector", "full_system"]
    return names


def load_existing(out_dir: Path, split: str) -> list[dict]:
    files = sorted(out_dir.glob(f"*_{split}.json"))
    return [json.loads(f.read_text("utf-8")) for f in files]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--systems", default="rules_only,gemma_zeroshot,qwen_base_fewshot")
    ap.add_argument(
        "--all", action="store_true", help="every runnable system (see runnable_systems)"
    )
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--checkpoint", default=None, help="Tinker sampler path for tuned_tinker")
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    ap.add_argument(
        "--strip-sender",
        action="store_true",
        help="evaluate every item with sender=None → {system}_{split}_nosender.json",
    )
    ap.add_argument("--table", action="store_true", help="only print the table from results")
    ap.add_argument("--i-am-the-final-eval", dest="final_eval", action="store_true")
    args = ap.parse_args(argv)

    if args.table:
        results = load_existing(args.out_dir, args.split)
        print(format_table(results) if results else "no results yet")
        return 0
    try:
        path = split_path(args.split, args.final_eval)
        if not path.exists():
            raise EvalRefused(f"{path} does not exist yet")
        items = load_items(path, args.limit)
        if args.strip_sender:
            items = [strip_sender(it) for it in items]
        if args.all:
            names = runnable_systems(args.checkpoint, served_detector_is_tuned())
            print(f"--all → {','.join(names)}", flush=True)
        else:
            names = [s.strip() for s in args.systems.split(",") if s.strip()]
        results = []
        for name in names:
            result, out = asyncio.run(evaluate_system(name, args, path, items))
            m = result["metrics"]
            print(
                f"{name}: n={m['n']} errors={m['errors']} unknown={m['unknown_rate']:.3f} "
                f"→ {out.relative_to(ROOT)}",
                flush=True,
            )
            results.append(result)
    except (EvalRefused, NotImplementedError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    print()
    print(format_table(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
