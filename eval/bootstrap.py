"""Bootstrap 95% CIs and McNemar test (PRD §12.2).

Percentile bootstrap over items with a fixed seed (deterministic). Resamples on which a
metric is undefined (e.g. no gold SCAM drawn) are skipped. McNemar uses the exact binomial
test on the discordant pairs (scipy), which is valid for small dev/test sets.
"""

from collections.abc import Callable, Sequence

import numpy as np
from scipy.stats import binomtest

from eval.metrics import HEADLINE, Gold, Pred

N_RESAMPLES = 1000
SEED = 20261003


def bootstrap_ci(
    golds: Sequence[Gold],
    preds: Sequence[Pred],
    metric: Callable[[Sequence[Gold], Sequence[Pred]], float | None],
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
    alpha: float = 0.05,
) -> tuple[float, float] | None:
    """(low, high) percentile CI, or None if the metric is undefined on (almost) every draw."""
    n = len(golds)
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        v = metric([golds[i] for i in idx], [preds[i] for i in idx])
        if v is not None:
            values.append(v)
    if len(values) < n_resamples // 2:
        return None
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def headline_cis(
    golds: Sequence[Gold], preds: Sequence[Pred], n_resamples: int = N_RESAMPLES, seed: int = SEED
) -> dict[str, list[float] | None]:
    out = {}
    for name, fn in HEADLINE.items():
        ci = bootstrap_ci(golds, preds, fn, n_resamples=n_resamples, seed=seed)
        out[name] = list(ci) if ci else None
    return out


def mcnemar(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict:
    """Exact McNemar test on paired correctness. b = A right / B wrong, c = A wrong / B right."""
    if len(correct_a) != len(correct_b):
        raise ValueError("paired sequences differ in length")
    b = sum(a and not bb for a, bb in zip(correct_a, correct_b, strict=True))
    c = sum(bb and not a for a, bb in zip(correct_a, correct_b, strict=True))
    p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
    return {"b_a_only": b, "c_b_only": c, "n": len(correct_a), "p_value": p}
