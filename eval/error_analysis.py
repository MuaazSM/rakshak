"""Run summary and error analysis inputs for notes/failures.md (PRD §12.5, PROMPTBOOK P4.2).

    uv run python -m eval.error_analysis [--runs training/runs] [--glob '*-full-*']

Reads each run's `config.json`, `metrics.json` and cached per-epoch Tinker dev predictions
(`tuned_tinker_e*_dev.items.jsonl`, ids and labels only) and writes
`eval/results/runs_summary_dev.json`: per run × epoch dev metrics, the selected epoch, and
per-seed-group verdict counts for gold SUSPICIOUS items (epoch-to-epoch instability).
Dev only; never reads the test split; prints no message text.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "training" / "runs"
DEV = ROOT / "data" / "splits" / "dev.jsonl"
OUT = ROOT / "eval" / "results" / "runs_summary_dev.json"
KEEP = (
    "macro_f1",
    "scam_recall",
    "scam_recall_strict",
    "fpr_genuine",
    "fpr_hard_negative",
    "category_accuracy",
    "span_f1",
    "grounding_rate",
    "json_validity",
    "unknown_rate",
)


def seed_groups(dev: Path) -> dict[str, str]:
    out = {}
    for line in dev.read_text("utf-8").splitlines():
        if line.strip():
            meta = json.loads(line).get("meta", {})
            out[str(meta.get("id"))] = str(meta.get("seed_group"))
    return out


def item_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]


def suspicious_by_group(rows: list[dict], groups: dict[str, str]) -> dict[str, dict[str, int]]:
    """gold-SUSPICIOUS seed group -> predicted verdict counts."""
    out: dict[str, Counter] = {}
    for r in rows:
        if r["gold_verdict"] == "SUSPICIOUS":
            out.setdefault(groups.get(r["id"], "?"), Counter())[r["pred_verdict"]] += 1
    return {g: dict(sorted(c.items())) for g, c in sorted(out.items())}


def summarize_run(run_dir: Path, groups: dict[str, str]) -> dict:
    cfg = json.loads((run_dir / "config.json").read_text("utf-8"))
    met = json.loads((run_dir / "metrics.json").read_text("utf-8"))
    epochs = []
    for ep in met["epochs"]:
        e = ep["epoch"]
        items = run_dir / f"tuned_tinker_e{e}_dev.items.jsonl"
        rows = item_rows(items) if items.exists() else []
        epochs.append(
            {
                "epoch": e,
                "train_loss_mean": ep.get("train_loss_mean"),
                "train_loss_last": ep.get("train_loss_last"),
                "checkpoint": ep.get("checkpoint"),
                **{k: ep["dev"].get(k) for k in KEEP},
                "suspicious_pred_by_seed_group": suspicious_by_group(rows, groups),
            }
        )
    return {
        "run_id": cfg["run_id"],
        "lr": cfg["hyperparams"]["lr"],
        "epochs_planned": cfg["hyperparams"]["epochs"],
        "status": cfg.get("status"),
        "dev_sha256": cfg["data"]["dev"]["sha256"],
        "train_sha256": cfg["data"]["train"]["sha256"],
        "provisional": met.get("provisional"),
        "split": met.get("split"),
        "best_epoch": (met.get("best") or {}).get("epoch"),
        "epochs": epochs,
    }


def build(runs_dir: Path, pattern: str, dev: Path) -> dict:
    groups = seed_groups(dev)
    runs = [
        summarize_run(d, groups)
        for d in sorted(runs_dir.glob(pattern))
        if d.is_dir() and (d / "metrics.json").exists() and "crashed" not in d.name
    ]
    return {
        "provisional": all(r["provisional"] for r in runs) if runs else True,
        "split": "dev_synthetic",
        "selection_rule": "dev macro-F1, then scam recall (PRD §11.1)",
        "runs": runs,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--glob", default="*-full-*")
    ap.add_argument("--dev", type=Path, default=DEV)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    if args.dev.name == "test.jsonl":
        print("refused: dev only", file=sys.stderr)
        return 2
    summary = build(args.runs, args.glob, args.dev)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
    print("PROVISIONAL (synthetic dev)" if summary["provisional"] else "dev")
    print("| run | epoch | loss | macro-F1 | recall | FPR gen | cat acc | span F1 | JSON |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in summary["runs"]:
        for e in r["epochs"]:
            star = " *" if e["epoch"] == r["best_epoch"] else ""
            print(
                f"| {r['run_id']} | {e['epoch']}{star} | {e['train_loss_mean']:.4f} "
                f"| {e['macro_f1']:.3f} | {e['scam_recall']:.3f} | {e['fpr_genuine']:.3f} "
                f"| {e['category_accuracy']:.3f} | {e['span_f1']:.3f} | {e['json_validity']:.3f} |"
            )
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
