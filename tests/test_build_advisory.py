"""Advisory eval slice builder (PRD §10.2): verdict derivation, redaction, never-test."""

import json

import pytest

from eval.run_eval import EvalRefused, split_path
from training import build_advisory as B
from training.redact import load_names


def test_verdict_derived_from_category():
    assert B.verdict_for("kyc_account_block") == "SCAM"
    assert B.verdict_for("digital_arrest") == "SCAM"
    assert B.verdict_for("genuine_otp") == "SAFE"
    assert B.verdict_for("govt_genuine") == "SAFE"
    with pytest.raises(ValueError):
        B.verdict_for("not_a_category")


def test_build_examples_renders_and_redacts():
    raw = [
        {
            "id": "pr-1",
            "text": "KYC expires today, verify now. OTP is 428193 do not share.",
            "sender": "VM-KYCUPD",
            "channel": "sms",
            "source": "public_report",
            "source_url": "https://rbi.org.in/x",
            "category_guess": "kyc_account_block",
        },
        {
            "id": "pr-2",
            "text": "Your parcel is out for delivery. Track at indiapost.gov.in",
            "sender": "AX-INDPST",
            "channel": "sms",
            "source": "public_report",
            "source_url": "https://pib.gov.in/y",
            "category_guess": "govt_genuine",
        },
    ]
    review = {"pr-1": {"id": "pr-1", "language_guess": "en"}}
    ex = B.build_examples(raw, review, load_names(None))
    assert [e["meta"]["verdict"] for e in ex] == ["SCAM", "SAFE"]
    assert all(e["meta"]["source"] == "public_report" for e in ex)
    assert all(e["meta"]["split"] == "advisory" and e["meta"]["auto_labeled"] for e in ex)
    # none are test-eligible, and the OTP digits were redacted out of the rendered text
    joined = json.dumps(ex, ensure_ascii=False)
    assert "428193" not in joined
    # system + user + assistant, assistant target carries the derived verdict
    scam = ex[0]["messages"]
    assert [m["role"] for m in scam] == ["system", "user", "assistant"]
    assert json.loads(scam[2]["content"])["verdict"] == "SCAM"


def test_advisory_split_resolves_and_is_not_test_guarded(tmp_path):
    p = split_path("advisory", final_eval=False, splits_dir=tmp_path)
    assert p == tmp_path / "advisory_eval.jsonl"
    # the test split is still guarded
    with pytest.raises(EvalRefused):
        split_path("test", final_eval=False, splits_dir=tmp_path)


def test_write_slice_and_lock(tmp_path):
    raw = [
        {
            "id": "pr-1",
            "text": "You won 25 lakh lottery, pay fee to claim at bit.ly/x",
            "sender": "+91 90000 11111",
            "channel": "whatsapp",
            "source": "public_report",
            "source_url": "https://example.news/z",
            "category_guess": "lottery_prize",
        }
    ]
    ex = B.build_examples(raw, {}, load_names(None))
    out = tmp_path / "advisory_eval.jsonl"
    lock = tmp_path / "splits.lock.json"
    lock.write_text(json.dumps({"test": None, "train": {"n": 1}}), encoding="utf-8")
    B.write_slice(ex, out)
    B.update_lock(out, ex, lock)
    data = json.loads(lock.read_text())
    assert data["advisory_eval"]["n"] == 1 and data["advisory_eval"]["provisional"] is True
    assert data["advisory_eval"]["auto_labeled"] is True
    assert data["test"] is None and data["train"] == {"n": 1}  # untouched
