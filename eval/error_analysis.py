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
import re
import sys
from collections import Counter
from pathlib import Path

from eval.metrics import selection_key
from eval.run_eval import sha256_file

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


_VARIANT = re.compile(r"-v\d+$")
_SAMPLE = re.compile(r"-(?:s)?\d+$")


def group_from_id(item_id: str) -> str:
    """Seed group from a synthetic id: `syn-{seed_group}-{s<n>|<n>}[-v<n>]`
    (e.g. syn-seed-043-s0-v1, syn-calls-c9-0, syn-links-genuine_otp-0-0), for runs whose dev
    file has since been rebuilt."""
    if not item_id.startswith("syn-"):
        return "?"
    rest = _VARIANT.sub("", item_id.removeprefix("syn-"))
    stripped = _SAMPLE.sub("", rest)
    return stripped if stripped != rest else "?"


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
            out.setdefault(groups.get(r["id"]) or group_from_id(r["id"]), Counter())[
                r["pred_verdict"]
            ] += 1
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
        "n_dev": cfg["data"]["dev"].get("n"),
        "train_sha256": cfg["data"]["train"]["sha256"],
        "provisional": met.get("provisional"),
        "split": met.get("split"),
        "best_epoch": (met.get("best") or {}).get("epoch"),
        "epochs": epochs,
    }


def _best(runs: list[dict]) -> dict | None:
    """Best (run, epoch) in one dev group by the §11.1 rule."""
    cands = [(r, e) for r in runs for e in r["epochs"]]
    if not cands:
        return None
    r, e = max(cands, key=lambda c: selection_key(c[1]))
    return {"run_id": r["run_id"], "epoch": e["epoch"], "checkpoint": e["checkpoint"]}


def build(runs_dir: Path, patterns: list[str], dev: Path) -> dict:
    """Runs grouped by the dev file they were selected on (SHA-256). Groups aren't comparable."""
    current = sha256_file(dev) if dev.exists() else None
    groups = seed_groups(dev) if dev.exists() else {}
    dirs = sorted({d for p in patterns for d in runs_dir.glob(p)})
    runs = [
        summarize_run(d, groups)
        for d in dirs
        if d.is_dir() and (d / "metrics.json").exists() and "crashed" not in d.name
    ]
    by_dev: dict[str, list[dict]] = {}
    for r in runs:
        by_dev.setdefault(r["dev_sha256"], []).append(r)
    dev_groups = [
        {
            "dev_sha256": sha,
            "is_current_dev": sha == current,
            "n_dev": rs[0]["n_dev"],
            "best": _best(rs),
            "runs": rs,
        }
        for sha, rs in by_dev.items()
    ]
    dev_groups.sort(key=lambda g: (g["is_current_dev"], g["runs"][0]["run_id"]))
    return {
        "provisional": all(r["provisional"] for r in runs) if runs else True,
        "split": "dev_synthetic",
        "selection_rule": "dev macro-F1, then scam recall (PRD §11.1)",
        "note": (
            "Runs are grouped by the dev file they were evaluated on. Different dev SHAs are "
            "different item sets (and v1 train overlaps the current dev's seed groups), so "
            "metrics are not comparable across groups. 'best' is per group."
        ),
        "dev_groups": dev_groups,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--glob", action="append", default=None, help="run dir glob (repeatable)")
    ap.add_argument("--dev", type=Path, default=DEV)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    if args.dev.name == "test.jsonl":
        print("refused: dev only", file=sys.stderr)
        return 2
    summary = build(args.runs, args.glob or ["*-full-*", "*-v2-*"], args.dev)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
    print("PROVISIONAL (synthetic dev)" if summary["provisional"] else "dev")
    print("| run | epoch | loss | macro-F1 | recall | FPR gen | cat acc | span F1 | JSON |")
    print("|---|---|---|---|---|---|---|---|---|")
    for g in summary["dev_groups"]:
        best = g["best"] or {}
        print(f"-- dev {g['dev_sha256'][:12]} (n={g['n_dev']}, current={g['is_current_dev']})")
        for r in g["runs"]:
            for e in r["epochs"]:
                star = (
                    " *"
                    if (r["run_id"], e["epoch"]) == (best.get("run_id"), best.get("epoch"))
                    else ""
                )
                print(
                    f"| {r['run_id']} | {e['epoch']}{star} | {e['train_loss_mean']:.4f} "
                    f"| {e['macro_f1']:.3f} | {e['scam_recall']:.3f} | {e['fpr_genuine']:.3f} "
                    f"| {e['category_accuracy']:.3f} | {e['span_f1']:.3f} | {e['json_validity']:.3f} |"
                )
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
