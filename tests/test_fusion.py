"""Fusion branches (PRD §7.4) and threshold loading."""

import json

import pytest

from hub import fusion
from hub.fusion import Thresholds, fuse
from tests.test_hub_env import FakeRules, hub_env, make_result  # noqa: F401

TH = Thresholds(0.80, 0.35, True)
TEXT = "Install http://sbi-kyc.in/app.apk now. Account will be blocked."
EVIDENCE = {
    "apk_link": ["http://sbi-kyc.in/app.apk"],
    "threat_lexicon": ["will be blocked"],
    "urgency_plus_payment": ["not in the text"],
    "asks_otp_or_pin": ["Account"],
}


def test_hard_rule_forces_scam_with_rule_flags_even_if_detector_safe():
    det = make_result("SAFE", p_scam=0.02, category="genuine_otp", flags=[])
    r = fuse(FakeRules.make(hard=["apk_link"], evidence=EVIDENCE), det, TEXT, TH)
    assert r.verdict == "SCAM"
    assert r.category == "malicious_apk"  # detector's safe category is not shown on a SCAM
    assert r.red_flags == [
        {"quote": "http://sbi-kyc.in/app.apk", "reason": "apk_link", "source": "rule"}
    ]
    assert r.p_scam == 0.02


def test_hard_rule_keeps_detector_scam_category():
    det = make_result("SCAM", 0.9, category="kyc_account_block")
    r = fuse(FakeRules.make(hard=["lookalike_domain"], evidence={}), det, TEXT, TH)
    assert (r.verdict, r.category, r.red_flags) == ("SCAM", "kyc_account_block", [])


def test_high_p_is_scam_with_model_flags():
    det = make_result("SCAM", 0.80)
    r = fuse(FakeRules.make(), det, TEXT, TH)
    assert r.verdict == "SCAM" and r.category == "kyc_account_block"
    assert r.red_flags == det.red_flags and r.p_scam == 0.80


def test_mid_p_is_suspicious_with_model_and_rule_flags():
    det = make_result("SUSPICIOUS", 0.5, category="other_scam")
    rules = FakeRules.make(soft=["threat_lexicon"], evidence=EVIDENCE)
    r = fuse(rules, det, TEXT, TH)
    assert r.verdict == "SUSPICIOUS"
    assert [f["source"] for f in r.red_flags] == ["model", "rule"]
    assert r.red_flags[1] == {
        "quote": "will be blocked",
        "reason": "threat_lexicon",
        "source": "rule",
    }


def test_p_exactly_at_t_low_is_suspicious():
    r = fuse(FakeRules.make(), make_result("SAFE", 0.35, category="personal"), TEXT, TH)
    assert r.verdict == "SUSPICIOUS" and r.category == "other_scam"


def test_two_soft_rules_escalate_a_safe_detector():
    rules = FakeRules.make(soft=["threat_lexicon", "asks_otp_or_pin"], evidence=EVIDENCE)
    r = fuse(rules, make_result("SAFE", 0.05, category="genuine_otp", flags=[]), TEXT, TH)
    assert r.verdict == "SUSPICIOUS"
    assert {f["reason"] for f in r.red_flags} == {"threat_lexicon", "asks_otp_or_pin"}


def test_one_soft_rule_and_low_p_is_safe():
    rules = FakeRules.make(soft=["threat_lexicon"], evidence=EVIDENCE)
    det = make_result("SAFE", 0.10, category="transaction_alert", flags=[])
    r = fuse(rules, det, TEXT, TH)
    assert (r.verdict, r.category, r.red_flags) == ("SAFE", "transaction_alert", [])


def test_rule_flags_skip_evidence_that_is_not_a_substring():
    rules = FakeRules.make(soft=["urgency_plus_payment", "threat_lexicon"], evidence=EVIDENCE)
    r = fuse(rules, make_result("SAFE", 0.5, category="personal", flags=[]), TEXT, TH)
    assert [f["reason"] for f in r.red_flags] == ["threat_lexicon"]


# --- detector unavailable / invalid: rules-only ----------------------------------


@pytest.mark.parametrize(
    ("hard", "soft", "verdict", "category"),
    [
        (["apk_link"], [], "SCAM", "malicious_apk"),
        (["upi_pin_to_receive"], [], "SCAM", "upi_collect_refund"),
        (["lookalike_domain"], [], "SCAM", "other_scam"),
        ([], ["threat_lexicon"], "SUSPICIOUS", "other_scam"),
        ([], [], "UNKNOWN", None),
    ],
)
def test_rules_only_mode(hard, soft, verdict, category):
    r = fuse(FakeRules.make(hard, soft, EVIDENCE), None, TEXT, TH)
    assert (r.verdict, r.category, r.p_scam) == (verdict, category, None)


def test_detector_down_is_never_safe():
    for soft in ([], ["threat_lexicon"], ["threat_lexicon", "asks_otp_or_pin"]):
        assert fuse(FakeRules.make(soft=soft, evidence=EVIDENCE), None, TEXT, TH).verdict != "SAFE"


def test_unknown_has_no_flags_or_category():
    r = fuse(FakeRules.make(), None, TEXT, TH)
    assert r.verdict == "UNKNOWN" and r.category is None and r.red_flags == []


# --- thresholds -----------------------------------------------------------------


def test_thresholds_default_uncalibrated(hub_env):  # noqa: F811
    th = fusion.load_thresholds()
    assert (th.t_high, th.t_low, th.calibrated) == (0.80, 0.35, False)


def test_thresholds_from_file(hub_env):  # noqa: F811
    f = hub_env / "t.json"
    f.write_text(json.dumps({"T_HIGH": 0.9, "T_LOW": 0.2}))
    assert fusion.load_thresholds(f) == Thresholds(0.9, 0.2, True)


def test_thresholds_from_settings_when_file_null(hub_env, monkeypatch):  # noqa: F811
    from hub import settings

    monkeypatch.setenv("T_HIGH", "0.7")
    monkeypatch.setenv("T_LOW", "0.3")
    settings.get_settings.cache_clear()
    f = hub_env / "t.json"
    f.write_text(json.dumps({"T_HIGH": None, "T_LOW": None}))
    assert fusion.load_thresholds(f) == Thresholds(0.7, 0.3, True)


def test_partial_thresholds_are_not_calibrated(hub_env):  # noqa: F811
    f = hub_env / "t.json"
    f.write_text(json.dumps({"T_HIGH": 0.9, "T_LOW": None}))
    th = fusion.load_thresholds(f)
    assert (th.t_high, th.t_low, th.calibrated) == (0.9, 0.35, False)


def test_thresholds_use_default_cache_path(hub_env):  # noqa: F811
    assert fusion.get_thresholds().calibrated is False


# --- §7.4 amended: SUSPICIOUS branch on risk = 1 - p_safe ---


def _with_p_safe(det, p_safe):
    det.p_safe = p_safe
    return det


def test_detector_suspicious_with_tiny_p_scam_is_suspicious_via_risk():
    det = _with_p_safe(make_result("SUSPICIOUS", 0.0002, category="other_scam"), 0.01)
    r = fuse(FakeRules.make(), det, TEXT, TH)
    assert r.verdict == "SUSPICIOUS" and r.p_scam == 0.0002


def test_low_risk_is_safe_and_high_p_scam_still_scam():
    det = _with_p_safe(make_result("SAFE", 0.01, category="personal", flags=[]), 0.98)
    assert fuse(FakeRules.make(), det, TEXT, TH).verdict == "SAFE"
    det = _with_p_safe(make_result("SCAM", 0.85), 0.0)
    assert fuse(FakeRules.make(), det, TEXT, TH).verdict == "SCAM"


def test_risk_falls_back_to_p_scam_without_p_safe():
    det = make_result("SUSPICIOUS", 0.0002, category="other_scam")
    assert det.p_safe is None
    assert fusion.risk(det) == 0.0002
    assert fuse(FakeRules.make(), det, TEXT, TH).verdict == "SAFE"
    assert fusion.risk(_with_p_safe(det, 0.25)) == 0.75
