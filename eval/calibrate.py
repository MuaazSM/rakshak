"""Calibrate T_HIGH/T_LOW on dev and write config/thresholds.json (PRD FR-36, §7.4, FR-15).

    uv run python -m eval.calibrate --split dev                 # run llama-server on dev, then search
    uv run python -m eval.calibrate --split dev --from-cache    # reuse calibrate_dev.items.jsonl
    uv run python -m eval.calibrate --split dev --dry-run       # search + print, write no config

1. The tuned detector (llama-server, `eval.systems.TunedDetector` → hub.detector.detect) runs on
   every dev item; per item only {id, label, p_scam, status} is cached to
   eval/results/calibrate_dev.items.jsonl (status ok | invalid | unavailable).
2. Rules are recomputed per item exactly as the hub does (hub.rules.evaluate on the normalized
   text + sender), and every candidate threshold pair is scored with hub.fusion.fuse itself.
3. T_HIGH = lowest candidate T such that the share of gold SAFE ("genuine") items fused to SCAM
   is ≤ 3%. T_LOW = highest candidate T (with T_HIGH fixed) such that scam recall — gold SCAM
   items fused to SCAM or SUSPICIOUS, which includes the ≥ 2 soft rules path — is ≥ 0.98.
   Candidates are the observed p_scam values (thresholds compare with ≥). If T_LOW > T_HIGH the
   SUSPICIOUS band would be empty, so T_LOW is clamped to T_HIGH and the clamp is recorded.
   If a target is unreachable the closest threshold is used and `feasible: false` is recorded.
4. Writes config/thresholds.json (hub.fusion reads only T_HIGH / T_LOW; the rest is provenance)
   and eval/results/calibration_dev.json with dev metrics at the chosen thresholds.

Refuses any split but dev (thresholds come from dev only; CLAUDE.md). Never prints message text.
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from eval.metrics import Pred, compute
from eval.run_eval import git_sha, is_provisional, sha256_file
from eval.systems import Item, gold_of, load_items

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "data" / "splits" / "dev.jsonl"
RESULTS = ROOT / "eval" / "results"
THRESHOLDS = ROOT / "config" / "thresholds.json"
MAX_FPR_GENUINE = 0.03  # §7.4 T_HIGH
MIN_SCAM_RECALL = 0.98  # §7.4 T_LOW
REPORT_METRICS = (
    "n",
    "scam_recall",
    "scam_recall_strict",
    "fpr_genuine",
    "fpr_hard_negative",
    "macro_f1",
    "unknown_rate",
)


class CalibrationError(Exception):
    pass


@dataclass(frozen=True)
class Case:
    """One dev item reduced to what fusion needs (no text kept beyond grounding)."""

    id: str
    label: str  # gold verdict
    p_scam: float | None  # None: detector invalid twice / unavailable → rules-only fusion
    rules: object  # hub.rules.RuleSignals
    text: str = ""


# --- fusion with candidate thresholds ---


def _det(p: float | None):
    """Minimal stand-in for DetectorResult: fuse() reads p_scam, output.category, red_flags."""
    if p is None:
        return None
    return SimpleNamespace(p_scam=p, output=SimpleNamespace(category="other_scam"), red_flags=[])


def verdicts(cases: list[Case], t_high: float, t_low: float) -> list[str]:
    from hub.fusion import Thresholds, fuse

    th = Thresholds(t_high, t_low, True)
    return [fuse(c.rules, _det(c.p_scam), c.text, th).verdict for c in cases]


def fpr_scam_genuine(cases: list[Case], t_high: float) -> float | None:
    """Gold SAFE items fused to SCAM (t_low does not affect the SCAM decision)."""
    v = verdicts(cases, t_high, t_high)
    safe = [x for c, x in zip(cases, v, strict=True) if c.label == "SAFE"]
    return sum(x == "SCAM" for x in safe) / len(safe) if safe else None


def scam_recall(cases: list[Case], t_high: float, t_low: float) -> float | None:
    v = verdicts(cases, t_high, t_low)
    scams = [x for c, x in zip(cases, v, strict=True) if c.label == "SCAM"]
    return sum(x in ("SCAM", "SUSPICIOUS") for x in scams) / len(scams) if scams else None


def candidates(cases: list[Case]) -> list[float]:
    return sorted({c.p_scam for c in cases if c.p_scam is not None})


def search_t_high(cases: list[Case], max_fpr: float = MAX_FPR_GENUINE) -> dict:
    """Lowest observed p with SCAM-FPR on genuine ≤ max_fpr (FPR is non-increasing in T)."""
    cands = candidates(cases)
    if not any(c.label == "SAFE" for c in cases):
        return {"value": None, "feasible": False, "reason": "no gold SAFE items on dev"}
    if not cands:
        return {"value": None, "feasible": False, "reason": "no detector p_scam on dev"}
    for t in cands:
        fpr = fpr_scam_genuine(cases, t)
        if fpr is not None and fpr <= max_fpr:
            return {"value": t, "feasible": True, "fpr_genuine_scam": fpr}
    t = cands[-1]
    return {
        "value": t,
        "feasible": False,
        "fpr_genuine_scam": fpr_scam_genuine(cases, t),
        "reason": f"FPR > {max_fpr} even at the highest p_scam (hard rules or p=1.0 on genuine)",
    }


def search_t_low(cases: list[Case], t_high: float, min_recall: float = MIN_SCAM_RECALL) -> dict:
    """Highest observed p (T_HIGH fixed) with SCAM∪SUSPICIOUS recall ≥ min_recall; clamped to
    ≤ T_HIGH."""
    cands = candidates(cases)
    if not any(c.label == "SCAM" for c in cases):
        return {"value": None, "feasible": False, "reason": "no gold SCAM items on dev"}
    if not cands:
        return {"value": None, "feasible": False, "reason": "no detector p_scam on dev"}
    out: dict | None = None
    for t in reversed(cands):
        r = scam_recall(cases, t_high, t)
        if r is not None and r >= min_recall:
            out = {"value": t, "feasible": True, "scam_recall": r}
            break
    if out is None:
        t = cands[0]
        out = {
            "value": t,
            "feasible": False,
            "scam_recall": scam_recall(cases, t_high, t),
            "reason": f"recall < {min_recall} even at the lowest p_scam",
        }
    if out["value"] > t_high:
        out["clamped_from"] = out["value"]
        out["value"] = t_high
        out["note"] = "T_LOW > T_HIGH: clamped to T_HIGH (empty SUSPICIOUS band from p alone)"
    return out


def metrics_at(cases: list[Case], items: list[Item], t_high: float, t_low: float) -> dict:
    v = verdicts(cases, t_high, t_low)
    golds = [gold_of(it) for it in items]
    preds = [Pred(verdict=x, p_scam=c.p_scam) for c, x in zip(cases, v, strict=True)]
    m = compute(golds, preds)
    return {k: m[k] for k in REPORT_METRICS}


# --- data ---


def build_cases(items: list[Item], cached: dict[str, dict]) -> list[Case]:
    from hub.normalize import normalize
    from hub.rules import evaluate

    cases = []
    for it in items:
        row = cached[it.id]
        norm = normalize(it.text, it.sender)
        cases.append(
            Case(
                id=it.id,
                label=it.gold.verdict,
                p_scam=row["p_scam"] if row.get("status", "ok") == "ok" else None,
                rules=evaluate(norm),
                text=norm.text,
            )
        )
    return cases


async def _score(items: list[Item]) -> list[dict]:
    from eval.systems import TunedDetector, run_system

    system = TunedDetector()
    try:
        preds = await run_system(system, items)
    finally:
        await system.aclose()
    rows = []
    for it, p in zip(items, preds, strict=True):
        if p.error:
            status = "unavailable"
        elif p.json_parsed is False:
            status = "invalid"
        else:
            status = "ok"
        rows.append(
            {
                "id": it.id,
                "label": it.gold.verdict,
                "p_scam": p.p_scam if status == "ok" else None,
                "status": status,
            }
        )
    return rows


def read_cache(path: Path) -> dict[str, dict]:
    rows = [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]
    return {r["id"]: r for r in rows}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")


def _rel(p: Path) -> str:
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def _f(v: float | None) -> str:
    return "—" if v is None else f"{v:.3f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--dev", type=Path, default=DEV)
    ap.add_argument("--url", help="llama-server base URL (default settings.detector_url)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--from-cache", action="store_true", help="reuse the p_scam cache")
    ap.add_argument("--cache", type=Path, default=RESULTS / "calibrate_dev.items.jsonl")
    ap.add_argument("--out", type=Path, default=THRESHOLDS)
    ap.add_argument("--report", type=Path, default=RESULTS / "calibration_dev.json")
    ap.add_argument("--detector-version", default=None, help="default settings.detector_version")
    ap.add_argument("--max-fpr", type=float, default=MAX_FPR_GENUINE)
    ap.add_argument("--min-recall", type=float, default=MIN_SCAM_RECALL)
    ap.add_argument("--dry-run", action="store_true", help="don't write config/thresholds.json")
    args = ap.parse_args(argv)
    try:
        if args.split != "dev" or args.dev.name == "test.jsonl":
            raise CalibrationError("thresholds come from dev only; --split test is refused")
        if not args.dev.exists():
            raise CalibrationError(f"{args.dev} does not exist")
        from hub.settings import get_settings

        if args.url:
            os.environ["DETECTOR_URL"] = args.url
            get_settings.cache_clear()
        s = get_settings()
        version = args.detector_version or s.detector_version
        items = load_items(args.dev, args.limit)
        if args.from_cache:
            if not args.cache.exists():
                raise CalibrationError(f"{args.cache} not found; run without --from-cache")
            cached = read_cache(args.cache)
            if missing := [it.id for it in items if it.id not in cached]:
                raise CalibrationError(f"cache misses {len(missing)} dev items; re-run without it")
        else:
            print(f"scoring {len(items)} dev items on {s.detector_url} ({version})", flush=True)
            rows = asyncio.run(_score(items))
            write_jsonl(args.cache, rows)
            cached = {r["id"]: r for r in rows}
        statuses = [cached[it.id].get("status", "ok") for it in items]
        if statuses.count("unavailable") > len(items) // 2:
            raise CalibrationError("detector unavailable for most items; is llama-server up?")
        cases = build_cases(items, cached)
        hi = search_t_high(cases, args.max_fpr)
        if hi["value"] is None:
            raise CalibrationError(f"T_HIGH: {hi['reason']}")
        lo = search_t_low(cases, hi["value"], args.min_recall)
        if lo["value"] is None:
            raise CalibrationError(f"T_LOW: {lo['reason']}")
    except CalibrationError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2

    t_high, t_low = round(hi["value"], 6), round(lo["value"], 6)
    m = metrics_at(cases, items, t_high, t_low)
    provisional = is_provisional(items)
    split = "dev_synthetic" if provisional else "dev"
    now = datetime.now(UTC).isoformat(timespec="seconds")
    thresholds = {
        "T_HIGH": t_high,
        "T_LOW": t_low,
        "split": split,
        "provisional": provisional,
        "n_dev": len(items),
        "detector_version": version,
        "created_at": now,
    }
    report = {
        **thresholds,
        "targets": {"max_fpr_genuine_scam": args.max_fpr, "min_scam_recall": args.min_recall},
        "t_high_search": hi,
        "t_low_search": lo,
        "metrics_at_thresholds": m,
        "detector_status": {k: statuses.count(k) for k in ("ok", "invalid", "unavailable")},
        "limit": args.limit,
        "data": {"path": _rel(args.dev), "sha256": sha256_file(args.dev)},
        "cache": _rel(args.cache),
        "git_sha": git_sha(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    if not args.dry_run:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(thresholds, indent=2) + "\n", "utf-8")
    tag = "PROVISIONAL (synthetic dev) " if provisional else ""
    print(
        f"{tag}T_HIGH {t_high:.4f}{'' if hi['feasible'] else ' (target not met)'}  "
        f"T_LOW {t_low:.4f}{'' if lo['feasible'] else ' (target not met)'}"
        f"{' (clamped to T_HIGH)' if 'clamped_from' in lo else ''}  n_dev {len(items)}\n"
        f"full fusion on dev: recall {_f(m['scam_recall'])} (strict {_f(m['scam_recall_strict'])})  "
        f"FPR genuine {_f(m['fpr_genuine'])}  FPR hard-neg {_f(m['fpr_hard_negative'])}  "
        f"macro-F1 {_f(m['macro_f1'])}  unknown {_f(m['unknown_rate'])}\n"
        f"→ {_rel(args.report)}" + ("" if args.dry_run else f", {_rel(args.out)}")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
