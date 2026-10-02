"""Threshold search (PRD FR-36, §7.4). Synthetic p_scam arrays; no server, no model."""

import json

import pytest

from eval import calibrate as C
from hub.rules import RuleSignals


def rules(hard=(), soft=()):
    return RuleSignals(hard=list(hard), soft=list(soft), evidence={})


def cases(spec):
    """spec: list of (label, p_scam, n_soft, hard?)"""
    out = []
    for i, (label, p, *rest) in enumerate(spec):
        n_soft = rest[0] if rest else 0
        hard = rest[1] if len(rest) > 1 else False
        out.append(
            C.Case(
                id=f"x{i}",
                label=label,
                p_scam=p,
                rules=rules(
                    hard=["apk_link"] if hard else (),
                    soft=[f"s{j}" for j in range(n_soft)],
                ),
            )
        )
    return out


def test_separable_scores_pick_tightest_thresholds():
    cs = cases(
        [("SCAM", p) for p in (0.9, 0.95, 0.99, 0.97, 0.92)]
        + [("SAFE", p) for p in (0.01, 0.02, 0.05, 0.1, 0.2)]
    )
    hi = C.search_t_high(cs)
    # lowest threshold with no genuine SCAM: the next lower observed p (0.2) is a SAFE item
    assert hi == {"value": 0.9, "feasible": True, "fpr_genuine_scam": 0.0}
    lo = C.search_t_low(cs, hi["value"])
    # every scam is already SCAM at T_HIGH, so recall holds at the top candidate → clamped
    assert lo["value"] == 0.9 and lo["clamped_from"] == 0.99 and lo["feasible"]


def test_all_scams_high_and_suspicious_band():
    cs = cases(
        [("SCAM", 0.99)] * 40
        + [("SCAM", 0.6)]  # needs the SUSPICIOUS band
        + [("SAFE", 0.0)] * 50
        + [("SAFE", 0.95)]  # 1/51 ≈ 2% FPR is allowed at 0.95…
        + [("SAFE", 0.97), ("SAFE", 0.98)]  # …but 3/53 is not
    )
    hi = C.search_t_high(cs)
    # 53 genuine: T=0.95 → 3/53, T=0.97 → 2/53 = 3.8% (too many), T=0.98 → 1/53 = 1.9%
    assert hi["value"] == 0.98 and hi["fpr_genuine_scam"] <= C.MAX_FPR_GENUINE
    lo = C.search_t_low(cs, hi["value"])
    # 40/41 = 0.976 < 0.98, so the 0.6 scam must be caught → T_LOW = 0.6
    assert lo == {"value": 0.6, "feasible": True, "scam_recall": 1.0}


def test_t_high_respects_three_percent_exactly():
    # 100 genuine; 3 with p = 0.9 → FPR 3% at T=0.9 is allowed; 4 would not be
    cs = cases([("SAFE", 0.0)] * 97 + [("SAFE", 0.9)] * 3 + [("SCAM", 0.95)] * 10)
    assert C.search_t_high(cs)["value"] == 0.9
    cs = cases([("SAFE", 0.0)] * 96 + [("SAFE", 0.9)] * 4 + [("SCAM", 0.95)] * 10)
    assert C.search_t_high(cs)["value"] == 0.95


def test_overlapping_distributions_meet_recall_with_low_threshold():
    scam = [0.95] * 45 + [0.5, 0.4, 0.3, 0.2, 0.1]  # 50 scams, 5 in the overlap
    safe = [0.05] * 40 + [0.15, 0.25, 0.35, 0.45, 0.55, 0.6, 0.7, 0.8, 0.85, 0.9]
    cs = cases([("SCAM", p) for p in scam] + [("SAFE", p) for p in safe])
    hi = C.search_t_high(cs)
    assert hi["value"] == 0.9  # 0.9 alone is 1/50 = 2%; 0.85 would be 4%
    lo = C.search_t_low(cs, hi["value"])
    # recall ≥ 0.98 of 50 → at most 1 miss → T_LOW must reach 0.2
    assert lo["value"] == 0.2 and lo["scam_recall"] == pytest.approx(0.98)


def test_soft_rules_count_toward_recall():
    # the 0.1 scam has 2 soft signals → SUSPICIOUS via rules regardless of T_LOW
    cs = cases([("SCAM", 0.95)] * 49 + [("SCAM", 0.1, 2)] + [("SAFE", 0.0)] * 20)
    lo = C.search_t_low(cs, 0.95)
    assert lo["feasible"] and lo["value"] == 0.95  # highest candidate, no clamp needed


def test_hard_rule_on_genuine_makes_t_high_infeasible():
    cs = cases([("SAFE", 0.0, 0, True)] * 5 + [("SAFE", 0.0)] * 5 + [("SCAM", 0.9)] * 5)
    hi = C.search_t_high(cs)
    assert hi["feasible"] is False and hi["value"] == 0.9


def test_no_genuine_or_no_scam_items():
    assert C.search_t_high(cases([("SCAM", 0.9)] * 3))["value"] is None
    assert C.search_t_low(cases([("SAFE", 0.1)] * 3), 0.9)["value"] is None


def test_unreachable_recall_uses_lowest_candidate():
    # one scam has no detector output (invalid twice) and no rules → UNKNOWN, never caught
    cs = cases([("SCAM", None)] * 5 + [("SCAM", 0.2)] * 5 + [("SAFE", 0.1)] * 5)
    lo = C.search_t_low(cs, 0.9)
    assert lo["feasible"] is False and lo["value"] == 0.1
    assert lo["scam_recall"] == pytest.approx(0.5)


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
    assert th["T_HIGH"] == 0.9 and th["T_LOW"] <= th["T_HIGH"]
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
