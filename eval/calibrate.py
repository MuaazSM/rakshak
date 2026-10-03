"""Calibrate T_HIGH/T_LOW on dev and write config/thresholds.json (PRD FR-36, §7.4, FR-15).

    uv run python -m eval.calibrate --split dev                 # run llama-server on dev, then search
    uv run python -m eval.calibrate --split dev --from-cache    # reuse calibrate_dev.items.jsonl
    uv run python -m eval.calibrate --split dev --dry-run       # search + print, write no config

1. The tuned detector (llama-server, hub.detector.detect) runs on every dev item; per item only
   {id, label, p_scam, p_safe, status} is cached to eval/results/calibrate_dev.items.jsonl
   (status ok | invalid | unavailable).
2. Rules are recomputed per item exactly as the hub does (hub.rules.evaluate on the normalized
   text + sender), and every candidate threshold pair is scored with hub.fusion.fuse itself
   (§7.4 as amended 3 Oct: SCAM if p_scam ≥ T_HIGH; SUSPICIOUS if risk = 1 − p_safe ≥ T_LOW or
   ≥ 2 soft rules).
3. T_HIGH (on p_scam) = lowest threshold such that the share of gold SAFE ("genuine") items
   fused to SCAM is ≤ 3%. T_LOW (on risk, T_HIGH fixed) = highest threshold such that scam
   recall — gold SCAM items fused to SCAM or SUSPICIOUS, incl. the ≥ 2 soft rules path — is
   ≥ 0.98. Candidates are the observed scores plus 1.0 (thresholds compare with ≥, so every T
   in (c[j-1], c[j]] behaves like c[j]).
4. Midpoint rule (§7.4): the edge found in 3 is widened to the interval of thresholds that are
   equivalent on dev — still meeting the target and with the same value of what the edge
   optimizes (T_HIGH: number of gold SCAM fused to SCAM; T_LOW: number of gold SAFE fused to
   SCAM/SUSPICIOUS) — and the midpoint of that interval is used. On separable scores this puts
   T_HIGH halfway between the highest equivalent genuine score and the lowest scam score
   instead of on a genuine item's score. T_LOW is on risk, so it is not clamped to T_HIGH. If a
   target is unreachable the closest candidate is used and `feasible: false` is recorded.
5. Writes config/thresholds.json (hub.fusion reads only T_HIGH / T_LOW; the rest is provenance)
   and eval/results/calibration_dev.json with full-fusion dev metrics at the chosen thresholds.

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
    p_safe: float | None = None  # None: fusion's risk falls back to p_scam

    @property
    def risk(self) -> float | None:
        if self.p_scam is None:
            return None
        return self.p_scam if self.p_safe is None else 1.0 - self.p_safe


# --- fusion with candidate thresholds ---


def _det(c: Case):
    """Minimal stand-in for DetectorResult: fuse() reads p_scam, p_safe, output.category and
    red_flags."""
    if c.p_scam is None:
        return None
    return SimpleNamespace(
        p_scam=c.p_scam,
        p_safe=c.p_safe,
        output=SimpleNamespace(category="other_scam"),
        red_flags=[],
    )


def verdicts(cases: list[Case], t_high: float, t_low: float) -> list[str]:
    from hub.fusion import Thresholds, fuse

    th = Thresholds(t_high, t_low, True)
    return [fuse(c.rules, _det(c), c.text, th).verdict for c in cases]


def fpr_scam_genuine(cases: list[Case], t_high: float) -> float | None:
    """Gold SAFE items fused to SCAM (t_low does not affect the SCAM decision)."""
    v = verdicts(cases, t_high, t_high)
    safe = [x for c, x in zip(cases, v, strict=True) if c.label == "SAFE"]
    return sum(x == "SCAM" for x in safe) / len(safe) if safe else None


def scam_recall(cases: list[Case], t_high: float, t_low: float) -> float | None:
    v = verdicts(cases, t_high, t_low)
    scams = [x for c, x in zip(cases, v, strict=True) if c.label == "SCAM"]
    return sum(x in ("SCAM", "SUSPICIOUS") for x in scams) / len(scams) if scams else None


def candidates(values) -> list[float]:
    """Observed scores plus 1.0 (the region above every score)."""
    return sorted({v for v in values if v is not None} | {1.0})


def _count(cases: list[Case], v: list[str], label: str, caught: tuple[str, ...]) -> int:
    return sum(c.label == label and x in caught for c, x in zip(cases, v, strict=True))


def _widen(cands: list[float], j: int, same) -> dict:
    """Interval (c[lo-1], c[hi]] of thresholds equivalent to candidate j (contiguous run of
    candidates where `same(k)` holds) and its midpoint."""
    lo = j
    while lo > 0 and same(lo - 1):
        lo -= 1
    hi = j
    while hi + 1 < len(cands) and same(hi + 1):
        hi += 1
    left = cands[lo - 1] if lo > 0 else 0.0
    return {"interval": [left, cands[hi]], "value": (left + cands[hi]) / 2}


def search_t_high(cases: list[Case], max_fpr: float = MAX_FPR_GENUINE) -> dict:
    """Lowest p_scam threshold with SCAM-FPR on genuine ≤ max_fpr (FPR is non-increasing in T),
    then the midpoint of the equivalent interval (same gold SCAM → SCAM count, still ≤ max_fpr)."""
    n_safe = sum(c.label == "SAFE" for c in cases)
    if not n_safe:
        return {"value": None, "feasible": False, "reason": "no gold SAFE items on dev"}
    if all(c.p_scam is None for c in cases):
        return {"value": None, "feasible": False, "reason": "no detector p_scam on dev"}
    cands = candidates(c.p_scam for c in cases)
    sig = []
    for t in cands:
        v = verdicts(cases, t, t)  # t_low does not affect the SCAM decision
        fpr = _count(cases, v, "SAFE", ("SCAM",)) / n_safe
        sig.append((fpr <= max_fpr, _count(cases, v, "SCAM", ("SCAM",)), fpr))
    j = next((k for k, s in enumerate(sig) if s[0]), None)
    if j is None:
        return {
            "value": cands[-1],
            "feasible": False,
            "fpr_genuine_scam": sig[-1][2],
            "reason": f"FPR > {max_fpr} even above every p_scam (hard rules on genuine)",
        }
    w = _widen(cands, j, lambda k: sig[k][0] and sig[k][1] == sig[j][1])
    return {
        "value": w["value"],
        "feasible": True,
        "edge": cands[j],
        "interval": w["interval"],
        "fpr_genuine_scam_at_edge": sig[j][2],
    }


def search_t_low(cases: list[Case], t_high: float, min_recall: float = MIN_SCAM_RECALL) -> dict:
    """Highest risk threshold (T_HIGH fixed) with SCAM∪SUSPICIOUS recall ≥ min_recall, then the
    midpoint of the equivalent interval (same gold SAFE flagged count, still ≥ min_recall).
    Not clamped to T_HIGH: T_LOW is on risk = 1 − P(SAFE) ≥ p_scam, a different score, and the
    SCAM branch is checked first (PRD §7.4 as amended 3 Oct)."""
    n_scam = sum(c.label == "SCAM" for c in cases)
    if not n_scam:
        return {"value": None, "feasible": False, "reason": "no gold SCAM items on dev"}
    if all(c.p_scam is None for c in cases):
        return {"value": None, "feasible": False, "reason": "no detector p_scam on dev"}
    cands = candidates(c.risk for c in cases)
    caught = ("SCAM", "SUSPICIOUS")
    sig = []
    for t in cands:
        v = verdicts(cases, t_high, t)
        r = _count(cases, v, "SCAM", caught) / n_scam
        sig.append((r >= min_recall, _count(cases, v, "SAFE", caught), r))
    j = next((k for k in reversed(range(len(cands))) if sig[k][0]), None)
    if j is None:
        out = {
            "value": cands[0],
            "feasible": False,
            "scam_recall": sig[0][2],
            "reason": f"recall < {min_recall} even at the lowest risk",
        }
    else:
        w = _widen(cands, j, lambda k: sig[k][0] and sig[k][1] == sig[j][1])
        out = {
            "value": w["value"],
            "feasible": True,
            "edge": cands[j],
            "interval": w["interval"],
            "scam_recall_at_edge": sig[j][2],
        }
    return out


def metrics_at(cases: list[Case], items: list[Item], t_high: float, t_low: float) -> dict:
    v = verdicts(cases, t_high, t_low)
    golds = [gold_of(it) for it in items]
    preds = [Pred(verdict=x, p_scam=c.p_scam) for c, x in zip(cases, v, strict=True)]
    m = compute(golds, preds)
    out = {k: m[k] for k in REPORT_METRICS}
    out["per_class_f1"] = m["per_class_f1"]
    out["confusion_gold_x_pred"] = m["confusion"]
    sus = [x for c, x in zip(cases, v, strict=True) if c.label == "SUSPICIOUS"]
    out["suspicious_flagged_rate"] = (
        sum(x in ("SCAM", "SUSPICIOUS") for x in sus) / len(sus) if sus else None
    )
    out["suspicious_exact_rate"] = sum(x == "SUSPICIOUS" for x in sus) / len(sus) if sus else None
    return out


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
                p_safe=row.get("p_safe") if row.get("status", "ok") == "ok" else None,
            )
        )
    return cases


async def _score(items: list[Item], concurrency: int = 2) -> list[dict]:
    """hub.detector.detect per item (same call as the hub); keeps p_scam and p_safe only."""
    import httpx

    from hub.detector import DetectorUnavailable, detect

    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=180.0) as client:

        async def one(it: Item) -> dict:
            async with sem:
                try:
                    res = await detect(
                        it.channel, it.sender, it.rule_signals, it.text, client=client
                    )
                    status = "invalid" if res is None else "ok"
                except (DetectorUnavailable, httpx.HTTPError, TimeoutError):
                    res, status = None, "unavailable"
            return {
                "id": it.id,
                "label": it.gold.verdict,
                "p_scam": res.p_scam if res else None,
                "p_safe": res.p_safe if res else None,
                "status": status,
            }

        return list(await asyncio.gather(*(one(it) for it in items)))


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
        f"{tag}T_HIGH {t_high:.4f} (p_scam; equivalent interval {hi.get('interval')})"
        f"{'' if hi['feasible'] else ' (target not met)'}\n"
        f"T_LOW {t_low:.4f} (risk = 1 - p_safe; interval {lo.get('interval')})"
        f"{'' if lo['feasible'] else ' (target not met)'}"
        f"{' (clamped to T_HIGH)' if 'clamped_from' in lo else ''}  n_dev {len(items)}\n"
        f"full fusion on dev: recall {_f(m['scam_recall'])} (strict {_f(m['scam_recall_strict'])})  "
        f"FPR genuine {_f(m['fpr_genuine'])}  FPR hard-neg {_f(m['fpr_hard_negative'])}  "
        f"macro-F1 {_f(m['macro_f1'])}  unknown {_f(m['unknown_rate'])}\n"
        f"per-class F1 {', '.join(f'{k} {_f(v)}' for k, v in m['per_class_f1'].items())}  "
        f"gold SUSPICIOUS → flagged {_f(m['suspicious_flagged_rate'])}, "
        f"exact {_f(m['suspicious_exact_rate'])}\n"
        f"→ {_rel(args.report)}" + ("" if args.dry_run else f", {_rel(args.out)}")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
