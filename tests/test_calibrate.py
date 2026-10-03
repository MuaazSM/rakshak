"""Threshold search (PRD FR-36, §7.4). Synthetic p_scam arrays; no server, no model."""

import json

import pytest

from eval import calibrate as C
from hub.rules import RuleSignals


def rules(hard=(), soft=()):
    return RuleSignals(hard=list(hard), soft=list(soft), evidence={})


def cases(spec):
    """spec: list of (label, p_scam, n_soft=0, hard=False, p_safe=None)"""
    out = []
    for i, (label, p, *rest) in enumerate(spec):
        n_soft = rest[0] if rest else 0
        hard = rest[1] if len(rest) > 1 else False
        p_safe = rest[2] if len(rest) > 2 else None
        out.append(
            C.Case(
                id=f"x{i}",
                label=label,
                p_scam=p,
                rules=rules(
                    hard=["apk_link"] if hard else (),
                    soft=[f"s{j}" for j in range(n_soft)],
                ),
                p_safe=p_safe,
            )
        )
    return out


def test_separable_scores_take_midpoints_without_clamp():
    cs = cases(
        [("SCAM", p) for p in (0.9, 0.95, 0.99, 0.97, 0.92)]
        + [("SAFE", p) for p in (0.01, 0.02, 0.05, 0.1, 0.2)]
    )
    hi = C.search_t_high(cs)
    # edge 0.9 (lowest with no genuine SCAM); equivalent down to just above 0.2 → midpoint 0.55
    assert hi["edge"] == 0.9 and hi["interval"] == [0.2, 0.9]
    assert hi["value"] == pytest.approx(0.55) and hi["feasible"]
    lo = C.search_t_low(cs, hi["value"])
    # every scam is already SCAM; equivalent risk thresholds (0.2, 0.99] (no 1.0 sentinel)
    # → midpoint 0.595 → capped at 0.5 (never SAFE when P(SAFE) < 0.5); never clamped
    assert lo["interval"] == [0.2, 0.99] and lo["capped_from"] == pytest.approx(0.595)
    assert lo["value"] == C.T_LOW_CAP and "clamped_from" not in lo


def test_degenerate_dev_suspicious_mass_on_suspicious_label():
    """The v1 synthetic-dev case (§17 3 Oct 06:45): genuine p_scam ≤ 0.006, scams ≥ 0.263,
    detector-SUSPICIOUS items with p_scam 0.0002 but p_safe 0.01."""
    cs = cases(
        [("SAFE", 0.001, 0, False, 0.99)] * 39
        + [("SAFE", 0.006, 0, False, 0.99)]
        + [("SCAM", 0.263 + k * 0.03, 0, False, 0.0) for k in range(20)]
        + [("SUSPICIOUS", 0.0002, 0, False, 0.01)] * 10
    )
    hi = C.search_t_high(cs)
    assert hi["edge"] == 0.006  # the literal "lowest" T_HIGH
    assert hi["interval"] == [0.001, 0.263] and hi["value"] == pytest.approx(0.132)
    lo = C.search_t_low(cs, hi["value"])
    # gold SUSPICIOUS (risk 0.99) must be flagged too → edge 0.99, interval (0.01, 0.99]
    assert lo["feasible"] and lo["edge"] == pytest.approx(0.99)
    assert lo["value"] == pytest.approx(0.5) and "capped_from" not in lo
    v = C.verdicts(cs, hi["value"], lo["value"])
    assert v[:40] == ["SAFE"] * 40 and v[40:60] == ["SCAM"] * 20
    assert v[60:] == ["SUSPICIOUS"] * 10  # risk 0.99, not p_scam 0.0002


def test_t_high_respects_three_percent_exactly():
    # 100 genuine; 3 at p = 0.9 → 3% at T=0.9 is allowed, and so is everything above 0.0
    cs = cases([("SAFE", 0.0)] * 97 + [("SAFE", 0.9)] * 3 + [("SCAM", 0.95)] * 10)
    hi = C.search_t_high(cs)
    assert hi["edge"] == 0.9 and hi["interval"] == [0.0, 0.95]
    assert hi["value"] == pytest.approx(0.475)
    cs = cases([("SAFE", 0.0)] * 96 + [("SAFE", 0.9)] * 4 + [("SCAM", 0.95)] * 10)
    hi = C.search_t_high(cs)
    assert hi["edge"] == 0.95 and hi["value"] == pytest.approx(0.925)


def test_overlapping_distributions_meet_recall_with_low_threshold():
    scam = [0.95] * 45 + [0.5, 0.4, 0.3, 0.2, 0.1]  # 50 scams, 5 in the overlap
    safe = [0.05] * 40 + [0.15, 0.25, 0.35, 0.45, 0.55, 0.6, 0.7, 0.8, 0.85, 0.9]
    cs = cases([("SCAM", p) for p in scam] + [("SAFE", p) for p in safe])
    hi = C.search_t_high(cs)
    assert hi["edge"] == 0.9  # 0.9 alone is 1/50 = 2%; 0.85 would be 4%
    assert hi["interval"] == [0.85, 0.95] and hi["value"] == pytest.approx(0.9)
    lo = C.search_t_low(cs, hi["value"])
    # recall ≥ 0.98 of 50 → at most 1 miss → edge 0.2; 0.15 would flag one more genuine
    assert lo["edge"] == 0.2 and lo["recall_scam_or_suspicious_at_edge"] == pytest.approx(0.98)
    assert lo["interval"] == [0.15, 0.2] and lo["value"] == pytest.approx(0.175)


def test_soft_rules_count_toward_recall():
    spec = [("SCAM", 0.95)] * 48 + [("SAFE", 0.0)] * 20
    with_soft = cases(spec + [("SCAM", 0.01, 2)] * 2)
    lo = C.search_t_low(with_soft, 0.9)
    assert lo["feasible"] and lo["interval"] == [0.0, 0.95] and lo["value"] == 0.475
    without = cases(spec + [("SCAM", 0.01)] * 2)
    lo = C.search_t_low(without, 0.9)
    assert lo["edge"] == 0.01 and lo["value"] == pytest.approx(0.005)


def test_t_low_uses_risk_not_p_scam():
    # detector says SUSPICIOUS on scams: little SCAM mass, little SAFE mass
    cs = cases(
        [("SCAM", 0.001, 0, False, 0.02)] * 50
        + [("SAFE", 0.0005, 0, False, 0.995)] * 50
        + [("SCAM", 0.97, 0, False, 0.0)] * 50
    )
    lo = C.search_t_low(cs, 0.5)
    assert lo["edge"] == pytest.approx(0.98)  # risk of the SUSPICIOUS-labelled scams
    assert lo["interval"][0] == pytest.approx(0.005) and "clamped_from" not in lo
    v = C.verdicts(cs, 0.5, lo["value"])
    assert v[:50] == ["SUSPICIOUS"] * 50 and v[50:100] == ["SAFE"] * 50


def test_t_low_counts_gold_suspicious_and_ignores_sentinel():
    """Review 3 Oct (second pass, item 1): one gold SAFE item the detector calls SUSPICIOUS
    (risk 0.812) must not push T_LOW above the gold SUSPICIOUS items (risk 0.6)."""
    cs = cases(
        [("SCAM", 0.99, 0, False, 0.0)] * 20
        + [("SUSPICIOUS", 0.01, 0, False, 0.4)] * 10
        + [("SAFE", 0.0, 0, False, 0.999)] * 30
        + [("SAFE", 0.01, 0, False, 0.188)]
    )
    lo = C.search_t_low(cs, 0.5)
    assert lo["edge"] == pytest.approx(0.6) and lo["interval"][1] == pytest.approx(0.6)
    assert lo["interval"][0] == pytest.approx(0.001) and lo["value"] == pytest.approx(0.3005)
    v = C.verdicts(cs, 0.5, lo["value"])
    assert set(v[20:30]) == {"SUSPICIOUS"} and v[-1] == "SUSPICIOUS"


def test_t_low_needs_some_positive():
    assert C.search_t_low(cases([("SAFE", 0.1)] * 3), 0.9)["value"] is None
    assert C.search_t_low(cases([("SUSPICIOUS", 0.1, 0, False, 0.3)] * 3), 0.9)["feasible"]


def test_hard_rule_on_genuine_makes_t_high_infeasible():
    cs = cases([("SAFE", 0.0, 0, True)] * 5 + [("SAFE", 0.0)] * 5 + [("SCAM", 0.9)] * 5)
    hi = C.search_t_high(cs)
    assert hi["feasible"] is False and hi["value"] == 1.0


def test_no_genuine_or_no_scam_items():
    assert C.search_t_high(cases([("SCAM", 0.9)] * 3))["value"] is None
    assert C.search_t_low(cases([("SAFE", 0.1)] * 3), 0.9)["value"] is None


def test_unreachable_recall_uses_lowest_candidate():
    # half the scams have no detector output (invalid twice) and no rules → UNKNOWN
    cs = cases([("SCAM", None)] * 5 + [("SCAM", 0.2)] * 5 + [("SAFE", 0.1)] * 5)
    lo = C.search_t_low(cs, 0.9)
    assert lo["feasible"] is False and lo["value"] == 0.1
    assert lo["recall_scam_or_suspicious"] == pytest.approx(0.5)


def test_verdicts_use_fusion_rules_only_path_when_detector_missing():
    cs = cases([("SCAM", None, 1), ("SAFE", None), ("SCAM", 0.5, 0, True)])
    assert C.verdicts(cs, 0.9, 0.3) == ["SUSPICIOUS", "UNKNOWN", "SCAM"]


def test_refuses_test_split(tmp_path, capsys):
    assert C.main(["--split", "test"]) == 2
    t = tmp_path / "test.jsonl"
    t.write_text("")
    assert C.main(["--dev", str(t)]) == 2
    assert "dev only" in capsys.readouterr().err


def test_main_from_cache_writes_thresholds(tmp_path, monkeypatch):
    from hub.detector import chat_example
    from hub.schemas import DetectorOutput

    dev = tmp_path / "dev.jsonl"
    rows, cache = [], []
    for i in range(40):
        scam = i < 20
        out = DetectorOutput.model_validate(
            {
                "verdict": "SCAM" if scam else "SAFE",
                "category": "kyc_account_block" if scam else "personal",
                "red_flags": [],
            }
        )
        msgs = chat_example("sms", None, [], f"synthetic message {i}", out)
        rows.append({"messages": msgs, "meta": {"id": f"d{i}", "source": "synthetic"}})
        p = 0.9 + i / 1000 if scam else i / 1000
        cache.append({"id": f"d{i}", "label": out.verdict, "p_scam": p, "status": "ok"})
    dev.write_text("".join(json.dumps(r) + "\n" for r in rows))
    c = tmp_path / "cache.jsonl"
    c.write_text("".join(json.dumps(r) + "\n" for r in cache))
    out, rep = tmp_path / "thresholds.json", tmp_path / "report.json"
    rc = C.main(
        [
            "--dev",
            str(dev),
            "--from-cache",
            "--cache",
            str(c),
            "--out",
            str(out),
            "--report",
            str(rep),
        ]
    )
    assert rc == 0
    th = json.loads(out.read_text())
    # edge 0.9; equivalent down to just above the top genuine 0.039 → midpoint; T_LOW on risk
    assert th["T_HIGH"] == pytest.approx((0.039 + 0.9) / 2)
    assert th["T_LOW"] == pytest.approx((0.039 + 0.919) / 2)  # top observed risk, no sentinel
    assert th["split"] == "dev_synthetic" and th["provisional"] is True and th["n_dev"] == 40
    assert set(th) == {
        "T_HIGH",
        "T_LOW",
        "split",
        "provisional",
        "n_dev",
        "detector_version",
        "created_at",
    }
    m = json.loads(rep.read_text())["metrics_at_thresholds"]
    assert m["scam_recall"] == 1.0 and m["fpr_genuine"] == 0.0

    # hub.fusion reads the file despite the extra keys
    from hub.fusion import load_thresholds

    t = load_thresholds(out)
    assert t.calibrated and t.t_high == th["T_HIGH"] and t.t_low == th["T_LOW"]
