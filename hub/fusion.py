"""Fusion of rules and detector into the final verdict (PRD FR-15, §7.4).

    hard rule              -> SCAM (rule red flags)
    p_scam >= T_HIGH       -> SCAM (model red flags)
    p_scam >= T_LOW or >= 2 soft rules -> SUSPICIOUS (model + rule red flags)
    otherwise              -> SAFE

Detector unavailable or invalid twice (rules-only): hard -> SCAM, >= 1 soft -> SUSPICIOUS,
else UNKNOWN. Never SAFE without the detector (PRD §13). Red flags shown to users are exact
substrings of the normalized text (PRD FR-14, NFR-5): rule flags whose evidence is not a
substring are skipped.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from hub.detector import DetectorResult
from hub.rules import RuleSignals
from hub.schemas import SafeCategory
from hub.settings import get_settings

DEFAULT_T_HIGH = 0.80
DEFAULT_T_LOW = 0.35
THRESHOLDS_FILE = Path(__file__).resolve().parent.parent / "config" / "thresholds.json"

# Rule-only SCAM -> category (hard signals only).
HARD_CATEGORY = {
    "apk_link": "malicious_apk",
    "upi_pin_to_receive": "upi_collect_refund",
    "lookalike_domain": "other_scam",
}


@dataclass(frozen=True)
class Thresholds:
    t_high: float
    t_low: float
    calibrated: bool


@dataclass(frozen=True)
class FusionResult:
    verdict: str  # SCAM | SUSPICIOUS | SAFE | UNKNOWN
    category: str | None
    red_flags: list[dict]
    p_scam: float | None


def _read_file(path: Path) -> tuple[float | None, float | None]:
    try:
        data = json.loads(path.read_text("utf-8"))
        hi, lo = data.get("T_HIGH"), data.get("T_LOW")
        return (None if hi is None else float(hi)), (None if lo is None else float(lo))
    except (OSError, ValueError, AttributeError):
        return None, None


def load_thresholds(path: Path | None = None) -> Thresholds:
    """config/thresholds.json (dev calibration), else settings T_HIGH/T_LOW, else defaults."""
    hi, lo = _read_file(path or THRESHOLDS_FILE)
    s = get_settings()
    hi = hi if hi is not None else s.t_high
    lo = lo if lo is not None else s.t_low
    calibrated = hi is not None and lo is not None
    return Thresholds(
        hi if hi is not None else DEFAULT_T_HIGH,
        lo if lo is not None else DEFAULT_T_LOW,
        calibrated,
    )


@lru_cache
def get_thresholds() -> Thresholds:
    return load_thresholds()


_SAFE_CATEGORIES = set(SafeCategory.__args__)


def _scam_category(det: DetectorResult, fallback: str) -> str:
    """Detector category when it is a scam category, else `fallback` (e.g. the detector said
    SAFE/genuine_otp but a rule escalated the verdict)."""
    c = det.output.category
    return fallback if c in _SAFE_CATEGORIES else c


def rule_red_flags(rules: RuleSignals, names: list[str], text: str) -> list[dict]:
    """One flag per signal: reason = signal name, quote = first evidence that is an exact
    substring of `text`. Signals without such evidence are skipped."""
    flags = []
    for name in names:
        quote = next((e for e in rules.evidence.get(name, []) if e and e in text), None)
        if quote:
            flags.append({"quote": quote, "reason": name, "source": "rule"})
    return flags


def fuse(
    rules: RuleSignals,
    det: DetectorResult | None,
    text: str,
    thresholds: Thresholds | None = None,
) -> FusionResult:
    th = thresholds or get_thresholds()
    p = det.p_scam if det else None
    if rules.hard:
        fallback = HARD_CATEGORY.get(rules.hard[0], "other_scam")
        category = _scam_category(det, fallback) if det else fallback
        return FusionResult("SCAM", category, rule_red_flags(rules, rules.hard, text), p)
    if det is None:  # rules-only mode
        if rules.soft:
            return FusionResult(
                "SUSPICIOUS", "other_scam", rule_red_flags(rules, rules.soft, text), None
            )
        return FusionResult("UNKNOWN", None, [], None)
    if det.p_scam >= th.t_high:
        return FusionResult(
            "SCAM", _scam_category(det, "other_scam"), list(det.red_flags), det.p_scam
        )
    if det.p_scam >= th.t_low or len(rules.soft) >= 2:
        flags = list(det.red_flags) + rule_red_flags(rules, rules.soft, text)
        return FusionResult("SUSPICIOUS", _scam_category(det, "other_scam"), flags, det.p_scam)
    c = det.output.category
    return FusionResult("SAFE", c if c in _SAFE_CATEGORIES else None, [], det.p_scam)
