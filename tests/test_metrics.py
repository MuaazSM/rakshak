"""Eval metrics and bootstrap (PRD §12.1, §12.2). Synthetic values only."""

import pytest

from eval import metrics as M
from eval.bootstrap import bootstrap_ci, headline_cis, mcnemar
from eval.metrics import Gold, Pred

TEXT = "Your account will be blocked today. Click http://sbi-kyc.example now"


def golds():
    return [
        Gold("SCAM", "kyc_account_block", ("will be blocked today",), text=TEXT),
        Gold("SCAM", "digital_arrest", ("CBI officer",), text="I am a CBI officer"),
        Gold("SCAM", "lottery_prize", ("won Rs 25 lakh",), text="You won Rs 25 lakh"),
        Gold("SAFE", "genuine_otp", (), hard_negative=True, text="OTP is <OTP>"),
        Gold("SAFE", "personal", (), text="Dinner at 8?"),
        Gold("SUSPICIOUS", "other_scam", (), text="Is this your number?"),
    ]


def preds():
    return [
        Pred("SCAM", "kyc_account_block", ["will be blocked today"], json_valid=True),
        Pred("SUSPICIOUS", "other_scam", ["a CBI"], json_valid=True),
        Pred("SAFE", "personal", [], json_valid=False),
        Pred("SUSPICIOUS", "genuine_otp", ["not in text"], json_valid=True),
        Pred("SAFE", "personal", [], json_valid=True),
        Pred("UNKNOWN", None, [], json_valid=None),
    ]


def test_recall_counts_suspicious_as_caught_and_strict_does_not():
    assert M.scam_recall(golds(), preds()) == pytest.approx(2 / 3)
    assert M.strict_scam_recall(golds(), preds()) == pytest.approx(1 / 3)


def test_fpr_on_genuine_and_hard_negatives_ignores_gold_suspicious():
    assert M.fpr_genuine(golds(), preds()) == pytest.approx(1 / 2)
    assert M.fpr_hard_negative(golds(), preds()) == pytest.approx(1.0)


def test_fpr_none_without_genuine_items():
    g, p = golds()[:3], preds()[:3]
    assert M.fpr_genuine(g, p) is None
    assert M.fpr_hard_negative(g, p) is None


def test_hard_negative_definition():
    assert M.is_hard_negative({}, "SAFE", "genuine_otp")
    assert M.is_hard_negative({}, "SAFE", "transaction_alert")
    assert M.is_hard_negative({"hard_negative": True}, "SAFE", "delivery_update")
    assert not M.is_hard_negative({}, "SAFE", "delivery_update")
    assert not M.is_hard_negative({"hard_negative": True}, "SCAM", "otp_phishing")


def test_macro_f1_hand_computed():
    # SCAM: tp1 fp0 fn2 → 2/4; SUSPICIOUS: tp0 → 0; SAFE: tp1 fp1 fn1 → 2/4
    assert M.macro_f1(golds(), preds()) == pytest.approx((0.5 + 0.0 + 0.5) / 3)


def test_macro_f1_perfect_and_unknown_never_correct():
    g = [Gold("SCAM", "other_scam"), Gold("SAFE", "personal")]
    assert M.macro_f1(g, [Pred("SCAM"), Pred("SAFE")]) == 1.0
    assert M.macro_f1(g, [Pred("SCAM"), Pred("UNKNOWN")]) == pytest.approx(0.5)
    assert M.macro_f1(g, [Pred("SCAM"), Pred("UNKNOWN")], unknown_as="SAFE") == 1.0


def test_category_accuracy_on_true_scams_only():
    assert M.category_accuracy(golds(), preds()) == pytest.approx(1 / 3)


def test_span_f1_token_overlap():
    # item1: 4/4 overlap; item2: gold {cbi, officer}, pred {a, cbi} → overlap 1; item3: 0/4
    # overlap 5, pred tokens 6, gold tokens 4+2+4=10 → P 5/6, R 5/10
    p, r = 5 / 6, 5 / 10
    assert M.span_f1(golds(), preds()) == pytest.approx(2 * p * r / (p + r))


def test_span_tokens_handle_devanagari_and_case():
    assert M.tokens("आपका खाता BLOCK होगा") == ["आपका", "खाता", "block", "होगा"]
    assert M.span_counts(["Block HOGA"], ["block hoga"]) == (2, 2, 2)


def test_span_f1_none_without_gold_quotes():
    assert M.span_f1([Gold("SAFE", "personal")], [Pred("SAFE")]) is None


def test_grounding_rate_is_pre_filter():
    assert M.grounding_rate(golds(), preds()) == pytest.approx(2 / 3)
    assert M.grounding_rate([Gold("SAFE", "personal")], [Pred("SAFE")]) is None


def test_json_validity_skips_not_applicable():
    assert M.json_validity(preds()) == pytest.approx(4 / 5)
    assert M.json_validity([Pred("SCAM")]) is None


def test_compute_has_every_metric_and_counts():
    m = M.compute(golds(), preds())
    for k in (*M.HEADLINE, "grounding_rate", "json_validity", "latency", "confusion"):
        assert k in m
    assert (m["n"], m["n_scam"], m["n_genuine"], m["n_hard_negative"]) == (6, 3, 2, 1)
    assert m["confusion"]["SUSPICIOUS"] == {"UNKNOWN": 1}
    assert m["unknown_rate"] == pytest.approx(1 / 6)


def test_compute_rejects_length_mismatch():
    with pytest.raises(ValueError):
        M.compute(golds(), preds()[:2])


def test_percentile_nearest_rank():
    assert M.percentile([], 50) is None
    assert M.percentile(list(range(1, 21)), 95) == 19
    assert M.percentile([5.0], 95) == 5.0


def test_selection_key_prefers_macro_f1_then_recall():
    a = {"macro_f1": 0.8, "scam_recall": 0.9}
    b = {"macro_f1": 0.8, "scam_recall": 0.95}
    c = {"macro_f1": 0.7, "scam_recall": 1.0}
    assert max([a, b, c], key=M.selection_key) is b


def test_bootstrap_deterministic_and_brackets_point():
    g, p = golds() * 10, preds() * 10
    ci1 = bootstrap_ci(g, p, M.scam_recall)
    ci2 = bootstrap_ci(g, p, M.scam_recall)
    assert ci1 == ci2
    assert ci1[0] <= M.scam_recall(g, p) <= ci1[1]


def test_bootstrap_none_when_undefined():
    assert bootstrap_ci([Gold("SAFE", "personal")], [Pred("SAFE")], M.scam_recall) is None
    assert set(headline_cis(golds(), preds(), n_resamples=50)) == set(M.HEADLINE)


def test_mcnemar_exact():
    r = mcnemar([True] * 10 + [False] * 2, [False] * 10 + [True] * 2)
    assert (r["b_a_only"], r["c_b_only"]) == (10, 2)
    assert r["p_value"] == pytest.approx(0.0386, abs=1e-3)
    assert mcnemar([True, False], [True, False])["p_value"] == 1.0
    with pytest.raises(ValueError):
        mcnemar([True], [True, False])
