"""Eval metrics as pure functions (PRD §12.1, NFR-3/4/5/6, §7.4 recall/FPR definitions).

Definitions (one place, used by run_eval, train_tinker and calibrate):
- **Scam recall** (headline, §7.4 "SCAM ∪ SUSPICIOUS"): among items whose gold verdict is SCAM,
  the share predicted SCAM *or* SUSPICIOUS — both put a warning in front of the parent.
  **Strict scam recall**: the share predicted SCAM.
- **FPR genuine**: among gold SAFE items, the share predicted SCAM or SUSPICIOUS.
  **FPR hard-neg**: the same over gold SAFE hard negatives (`meta.hard_negative`, or category
  `genuine_otp` / `transaction_alert`). Gold SUSPICIOUS items count in neither.
- `UNKNOWN` (detector unavailable / unparseable) is never "caught" and never a false positive;
  for macro-F1 it is wrong for every class. `macro_f1_unknown_as_safe` also scores it as SAFE.
- **Macro-F1**: mean per-class F1 over SCAM / SUSPICIOUS / SAFE, over classes that occur in
  gold or predictions (the sklearn convention).
- **Category accuracy**: among gold SCAM items, predicted category == gold category.
- **Span F1**: token overlap (multiset, lower-cased word tokens incl. Devanagari) between
  predicted and gold red-flag quotes, micro-averaged over gold items that have gold quotes.
- **Grounding rate**: share of predicted quotes (before the FR-14 filter) that are exact
  substrings of the normalized input text.
- **JSON validity**: share of model replies that parse as strict §8.2 JSON (`DetectorOutput`);
  `json_parse_rate` is the share that parse as any JSON object. Not applicable (None) to
  systems that don't emit JSON (rules only).
"""

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

LABELS = ("SCAM", "SUSPICIOUS", "SAFE")
CAUGHT = frozenset({"SCAM", "SUSPICIOUS"})
HARD_NEG_CATEGORIES = frozenset({"genuine_otp", "transaction_alert"})
_TOKEN = re.compile(r"[\wऀ-ॿ]+")


@dataclass(frozen=True)
class Gold:
    verdict: str
    category: str
    quotes: tuple[str, ...] = ()
    hard_negative: bool = False
    text: str = ""  # normalized message; used for grounding only, never written out


@dataclass
class Pred:
    verdict: str  # SCAM | SUSPICIOUS | SAFE | UNKNOWN
    category: str | None = None
    quotes: list[str] = field(default_factory=list)
    json_valid: bool | None = None  # strict §8.2; None = not applicable
    json_parsed: bool | None = None  # any JSON object; None = not applicable
    p_scam: float | None = None
    latency_ms: float | None = None
    error: str | None = None  # exception type name only


def is_hard_negative(meta: dict, verdict: str, category: str) -> bool:
    """Gold SAFE item flagged hard_negative or in a hard-negative category (§12.1, §10.4)."""
    return verdict == "SAFE" and (
        bool(meta.get("hard_negative")) or category in HARD_NEG_CATEGORIES
    )


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def scam_recall(golds: Sequence[Gold], preds: Sequence[Pred]) -> float | None:
    pairs = [(g, p) for g, p in zip(golds, preds, strict=True) if g.verdict == "SCAM"]
    return _ratio(sum(p.verdict in CAUGHT for _, p in pairs), len(pairs))


def strict_scam_recall(golds: Sequence[Gold], preds: Sequence[Pred]) -> float | None:
    pairs = [(g, p) for g, p in zip(golds, preds, strict=True) if g.verdict == "SCAM"]
    return _ratio(sum(p.verdict == "SCAM" for _, p in pairs), len(pairs))


def fpr_genuine(golds: Sequence[Gold], preds: Sequence[Pred]) -> float | None:
    pairs = [(g, p) for g, p in zip(golds, preds, strict=True) if g.verdict == "SAFE"]
    return _ratio(sum(p.verdict in CAUGHT for _, p in pairs), len(pairs))


def fpr_hard_negative(golds: Sequence[Gold], preds: Sequence[Pred]) -> float | None:
    pairs = [
        (g, p) for g, p in zip(golds, preds, strict=True) if g.verdict == "SAFE" and g.hard_negative
    ]
    return _ratio(sum(p.verdict in CAUGHT for _, p in pairs), len(pairs))


def per_class_f1(gold: Sequence[str], pred: Sequence[str]) -> dict[str, float]:
    """F1 per detector label present in gold or pred. Other predicted labels (UNKNOWN) are
    simply wrong: a false negative for the gold class, a false positive for none."""
    out = {}
    for c in LABELS:
        if c not in gold and c not in pred:
            continue
        tp = sum(g == c and p == c for g, p in zip(gold, pred, strict=True))
        fp = sum(g != c and p == c for g, p in zip(gold, pred, strict=True))
        fn = sum(g == c and p != c for g, p in zip(gold, pred, strict=True))
        out[c] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return out


def macro_f1(
    golds: Sequence[Gold], preds: Sequence[Pred], unknown_as: str | None = None
) -> float | None:
    gold = [g.verdict for g in golds]
    pred = [p.verdict if p.verdict in LABELS or unknown_as is None else unknown_as for p in preds]
    f1 = per_class_f1(gold, pred)
    return sum(f1.values()) / len(f1) if f1 else None


def category_accuracy(golds: Sequence[Gold], preds: Sequence[Pred]) -> float | None:
    pairs = [(g, p) for g, p in zip(golds, preds, strict=True) if g.verdict == "SCAM"]
    return _ratio(sum(p.category == g.category for g, p in pairs), len(pairs))


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def span_counts(gold_quotes: Sequence[str], pred_quotes: Sequence[str]) -> tuple[int, int, int]:
    """(overlap, predicted tokens, gold tokens) as multisets over all quotes of one item."""
    g = Counter(t for q in gold_quotes for t in tokens(q))
    p = Counter(t for q in pred_quotes for t in tokens(q))
    return sum((g & p).values()), sum(p.values()), sum(g.values())


def span_f1(golds: Sequence[Gold], preds: Sequence[Pred]) -> float | None:
    overlap = n_pred = n_gold = 0
    for g, p in zip(golds, preds, strict=True):
        if not g.quotes:
            continue
        o, np_, ng = span_counts(g.quotes, p.quotes)
        overlap, n_pred, n_gold = overlap + o, n_pred + np_, n_gold + ng
    if not n_gold:
        return None
    if not overlap:
        return 0.0
    precision, recall = overlap / n_pred, overlap / n_gold
    return 2 * precision * recall / (precision + recall)


def grounding_rate(golds: Sequence[Gold], preds: Sequence[Pred]) -> float | None:
    total = grounded = 0
    for g, p in zip(golds, preds, strict=True):
        for q in p.quotes:
            total += 1
            grounded += bool(q) and q in g.text
    return _ratio(grounded, total)


def _applicable_rate(values: Sequence[bool | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return _ratio(sum(vals), len(vals))


def json_validity(preds: Sequence[Pred]) -> float | None:
    return _applicable_rate([p.json_valid for p in preds])


def json_parse_rate(preds: Sequence[Pred]) -> float | None:
    return _applicable_rate([p.json_parsed for p in preds])


def unknown_rate(preds: Sequence[Pred]) -> float | None:
    return _ratio(sum(p.verdict not in LABELS for p in preds), len(preds))


def confusion(golds: Sequence[Gold], preds: Sequence[Pred]) -> dict[str, dict[str, int]]:
    """gold verdict -> predicted verdict -> count (UNKNOWN included as a column)."""
    out: dict[str, dict[str, int]] = {}
    for g, p in zip(golds, preds, strict=True):
        row = out.setdefault(g.verdict, {})
        row[p.verdict] = row.get(p.verdict, 0) + 1
    return out


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0..100)."""
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, math.ceil(q / 100 * len(s)) - 1))
    return s[k]


def latency(preds: Sequence[Pred]) -> dict[str, float | None]:
    ms = [p.latency_ms for p in preds if p.latency_ms is not None]
    return {"p50_ms": percentile(ms, 50), "p95_ms": percentile(ms, 95)}


# Headline metrics: name -> fn(golds, preds). Bootstrap CIs are computed for these.
HEADLINE = {
    "scam_recall": scam_recall,
    "scam_recall_strict": strict_scam_recall,
    "fpr_genuine": fpr_genuine,
    "fpr_hard_negative": fpr_hard_negative,
    "macro_f1": macro_f1,
    "category_accuracy": category_accuracy,
    "span_f1": span_f1,
}


def compute(golds: Sequence[Gold], preds: Sequence[Pred]) -> dict:
    """Every §12.1 metric for one system on one split (no CIs)."""
    if len(golds) != len(preds):
        raise ValueError("golds and preds differ in length")
    out: dict = {name: fn(golds, preds) for name, fn in HEADLINE.items()}
    out |= {
        "macro_f1_unknown_as_safe": macro_f1(golds, preds, unknown_as="SAFE"),
        "per_class_f1": per_class_f1([g.verdict for g in golds], [p.verdict for p in preds]),
        "grounding_rate": grounding_rate(golds, preds),
        "json_validity": json_validity(preds),
        "json_parse_rate": json_parse_rate(preds),
        "unknown_rate": unknown_rate(preds),
        "errors": sum(p.error is not None for p in preds),
        "latency": latency(preds),
        "confusion": confusion(golds, preds),
        "n": len(golds),
        "n_scam": sum(g.verdict == "SCAM" for g in golds),
        "n_suspicious": sum(g.verdict == "SUSPICIOUS" for g in golds),
        "n_genuine": sum(g.verdict == "SAFE" for g in golds),
        "n_hard_negative": sum(g.verdict == "SAFE" and g.hard_negative for g in golds),
    }
    return out


def selection_key(m: dict) -> tuple[float, float]:
    """Checkpoint selection (§11.1): dev macro-F1, then scam recall."""
    return (m.get("macro_f1") or 0.0, m.get("scam_recall") or 0.0)
