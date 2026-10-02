"""Test-set freezing (PRD FR-32, §10.5). Synthetic items only."""

import hashlib
import json

import pytest

from training import freeze_test
from training.freeze_test import main

ITEMS = [
    {
        "id": "t1",
        "text": "Your  SBI a/c <ACCT> will be blocked today.​ Update at http://sbi-kyc.example",
        "sender": "VK-SBIUPD",
        "channel": "sms",
        "source": "family_real",
    },
    {
        "id": "t2",
        "text": "आपका OTP <OTP> है। किसी के साथ साझा न करें।",
        "sender": "AD-SBIOTP",
        "channel": "sms",
        "source": "own_inbox",
    },
    {"id": "n1", "text": "Lottery! Pay Rs 5000", "channel": "sms", "source": "family_real"},
]


def label(id_, verdict, category, flags, language, source, phone, is_test, obfuscated=False):
    return {
        "id": id_,
        "file": "mom.jsonl",
        "verdict": verdict,
        "category": category,
        "red_flags": flags,
        "language": language,
        "source": source,
        "source_phone": phone,
        "obfuscated": obfuscated,
        "is_test": is_test,
        "seed_group": id_,
    }


LABELS = [
    label(
        "t1",
        "SCAM",
        "kyc_account_block",
        [{"quote": "will be blocked today", "reason": "urgency_deadline"}],
        "en",
        "family_real",
        "mom",
        True,
    ),
    label("t2", "SAFE", "genuine_otp", [], "hi", "own_inbox", "own", True),
    label("n1", "SCAM", "lottery_prize", [], "en", "family_real", "dad", False),  # not test
]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(freeze_test, "rule_signals_fn", lambda: None)
    red = tmp_path / "redacted"
    red.mkdir()
    (red / "mom.jsonl").write_text(
        "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in ITEMS), "utf-8"
    )
    labels = red / "_labels.jsonl"
    labels.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in LABELS), "utf-8")
    out = tmp_path / "splits"
    args = ["--redacted", str(red), "--labels", str(labels), "--out", str(out)]
    return args, out, labels


def test_freeze_writes_chat_format_and_lock(setup, capsys):
    args, out, _ = setup
    assert main(args) == 0
    lines = (out / "test.jsonl").read_text("utf-8").splitlines()
    assert len(lines) == 2  # only TEST items
    ex = json.loads(lines[0])
    assert ex["meta"]["id"] == "t1"
    user = ex["messages"][1]["content"]
    assert user == (
        "CHANNEL: sms\nSENDER: VK-SBIUPD (registered)\nRULE_SIGNALS: []\nMESSAGE:\n"
        "Your SBI a/c <ACCT> will be blocked today. Update at http://sbi-kyc.example"
    )
    assert json.loads(ex["messages"][2]["content"])["verdict"] == "SCAM"

    lock = json.loads((out / "splits.lock.json").read_text("utf-8"))["test"]
    assert lock["sha256"] == hashlib.sha256((out / "test.jsonl").read_bytes()).hexdigest()
    assert lock["n"] == 2 and lock["path"] == "test.jsonl"
    assert lock["counts"]["verdict"] == {"SAFE": 1, "SCAM": 1}
    assert lock["counts"]["language"] == {"en": 1, "hi": 1}
    assert lock["counts"]["source"] == {"family_real": 1, "own_inbox": 1}
    assert lock["counts"]["source_phone"] == {"mom": 1, "own": 1}
    assert lock["counts"]["obfuscated"] == {"false": 2}
    assert lock["rule_signals"].startswith("none")

    printed = capsys.readouterr().out
    for fragment in ("blocked", "SBI a/c", "OTP", "साझा"):
        assert fragment not in printed  # counts only


def test_refuses_overwrite_without_force(setup, capsys):
    args, out, _ = setup
    assert main(args) == 0
    sha = hashlib.sha256((out / "test.jsonl").read_bytes()).hexdigest()
    assert main(args) == 1
    assert "refusing" in capsys.readouterr().err
    assert hashlib.sha256((out / "test.jsonl").read_bytes()).hexdigest() == sha
    assert main([*args, "--force"]) == 0


def test_force_drops_stale_train_dev_entries(setup):
    args, out, _ = setup
    assert main(args) == 0
    lock_path = out / "splits.lock.json"
    lock = json.loads(lock_path.read_text("utf-8"))
    lock["train"] = {"sha256": "x"}
    lock["dev"] = {"sha256": "y"}
    lock_path.write_text(json.dumps(lock), "utf-8")
    assert main([*args, "--force"]) == 0
    assert set(json.loads(lock_path.read_text("utf-8"))) == {"test"}


def test_rule_signals_used_when_rules_exist(setup, monkeypatch):
    args, out, _ = setup
    monkeypatch.setattr(
        freeze_test,
        "rule_signals_fn",
        lambda: lambda text, sender, channel: ["threat_lexicon"] if "blocked" in text else [],
    )
    assert main(args) == 0
    first = json.loads((out / "test.jsonl").read_text("utf-8").splitlines()[0])
    assert "RULE_SIGNALS: [threat_lexicon]" in first["messages"][1]["content"]
    assert (
        json.loads((out / "splits.lock.json").read_text("utf-8"))["test"]["rule_signals"]
        == "hub.rules.rule_signals"
    )


def test_stale_quote_is_rejected_without_leaking_text(setup, capsys):
    args, out, labels = setup
    bad = [*LABELS]
    bad[0] = {**bad[0], "red_flags": [{"quote": "blocked tomorrow", "reason": "urgency_deadline"}]}
    labels.write_text("".join(json.dumps(x) + "\n" for x in bad), "utf-8")
    assert main(args) == 2
    err = capsys.readouterr().err
    assert "'t1'" in err and "blocked" not in err
    assert not (out / "test.jsonl").exists()


def test_non_real_item_cannot_be_test(setup):
    args, _, labels = setup
    bad = [*LABELS, label("n1", "SCAM", "lottery_prize", [], "en", "advisory", None, True)]
    labels.write_text("".join(json.dumps(x) + "\n" for x in bad), "utf-8")
    assert main(args) == 2
