"""Export parity computation (PRD §11.4). No Tinker, no server."""

import json

from eval import parity as P


def row(v, cat=None, valid=True, error=None):
    return {"pred_verdict": v, "pred_category": cat, "json_valid": valid, "error": error}


def test_full_agreement_passes():
    ids = ["a", "b", "c"]
    ref = [row("SCAM", "kyc_account_block"), row("SAFE", "genuine_otp"), row("SUSPICIOUS")]
    rep = P.parity(ids, ref, [dict(r) for r in ref])
    assert rep["verdict_agreement"] == 1.0 and rep["pass"]
    assert rep["disagreements"] == [] and rep["category_agreement_on_agreed_scams"] == 1.0
    assert rep["confusion_tinker_x_gguf"]["SCAM"]["SCAM"] == 1


def test_agreement_threshold_and_disagreement_ids():
    ids = [f"i{k}" for k in range(20)]
    ref = [row("SCAM")] * 20
    served = [row("SCAM")] * 19 + [row("SAFE")]
    rep = P.parity(ids, ref, served)
    assert rep["verdict_agreement"] == 0.95 and rep["pass"]  # ≥ 95% passes
    assert rep["disagreements"] == ["i19"]
    served = [row("SCAM")] * 18 + [row("SAFE"), row("SUSPICIOUS")]
    assert not P.parity(ids, ref, served)["pass"]


def test_errors_and_json_validity_reported_separately():
    ids = ["a", "b", "c", "d"]
    ref = [row("SCAM"), row("SAFE"), row("SAFE", valid=False), row("SCAM")]
    served = [
        row("SCAM"),
        row("UNKNOWN", valid=None, error="ConnectError"),
        row("SAFE"),
        row("SCAM", "digital_arrest", valid=False),
    ]
    rep = P.parity(ids, ref, served)
    assert rep["verdict_agreement"] == 0.75
    assert rep["verdict_agreement_no_errors"] == 1.0
    assert rep["errors_gguf"] == 1 and rep["errors_tinker"] == 0
    assert rep["json_validity_tinker"] == 0.75
    assert rep["json_validity_gguf"] == 2 / 3
    assert rep["category_agreement_on_agreed_scams"] == 0.5
    assert rep["confusion_tinker_x_gguf"]["SAFE"]["UNKNOWN"] == 1


def test_reuse_rows_requires_full_coverage(tmp_path):
    f = tmp_path / "items.jsonl"
    f.write_text("".join(json.dumps({"id": i, **row("SAFE")}) + "\n" for i in ("a", "b")))
    assert [r["id"] for r in P.reuse_rows(f, ["b", "a"])] == ["b", "a"]
    assert P.reuse_rows(f, ["a", "c"]) is None
    assert P.reuse_rows(tmp_path / "missing.jsonl", ["a"]) is None


def test_best_epoch_items(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"best": {"epoch": 2, "checkpoint": "tinker://x/sampler_weights/e2"}})
    )
    ck, path = P.best_epoch_items(tmp_path)
    assert ck.endswith("e2") and path.name == "tuned_tinker_e2_dev.items.jsonl"


def test_refuses_non_dev(capsys):
    assert P.main(["--split", "test", "--checkpoint", "tinker://x"]) == 2
    assert "dev only" in capsys.readouterr().err
