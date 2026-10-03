"""Dataset builder: group split, dedupe, dropout on train only, lock file, test-set safety."""

import json

import pytest

from training import build_dataset as bd

SCAM_FLAGS = [{"quote": "will be blocked", "reason": "urgency_deadline"}]


def item(i, group, text, verdict="SCAM", category="kyc_account_block", flags=None):
    return {
        "id": f"syn-{i}",
        "text": text,
        "sender": "+91 98100 22334" if verdict != "SAFE" else "AX-HDFCBK",
        "channel": "sms",
        "verdict": verdict,
        "category": category,
        "red_flags": (SCAM_FLAGS if verdict != "SAFE" else []) if flags is None else flags,
        "language": "en",
        "source": "synthetic",
        "source_phone": None,
        "obfuscated": False,
        "seed_group": group,
        "hard_negative": verdict == "SAFE",
    }


def distinct_text(tag: str, n: int) -> str:
    # far apart in char 5-gram space so MinHash never merges them
    words = [f"{tag}{n}{chr(97 + (n * k) % 26)}{k * n}x" for k in range(1, 9)]
    return "Your account will be blocked " + " ".join(words)


def pool():
    items = []
    n = 0
    for gi, (verdict, cat) in enumerate(
        [("SCAM", "kyc_account_block"), ("SCAM", "digital_arrest"), ("SAFE", "genuine_otp")]
    ):
        for g in range(3):  # three seed groups per stratum
            for _ in range(6):
                n += 1
                text = distinct_text(f"g{gi}{g}", n)
                if verdict == "SAFE":
                    text = f"OTP is <OTP>. Do not share it. ref {text[30:]}"
                items.append(item(n, f"seed-{gi}{g}", text, verdict, cat))
    return items


def test_split_by_seed_group_is_disjoint_and_caps_dev():
    train, dev, dev_groups = bd.split_groups(pool(), dev_per_stratum=4, dev_max_per_group=4, seed=1)
    assert {i["seed_group"] for i in train}.isdisjoint({i["seed_group"] for i in dev})
    assert len(dev_groups) == 3  # one group per stratum was enough for 4 items
    assert all(sum(1 for i in dev if i["seed_group"] == g) <= 4 for g in dev_groups)
    # the unused rest of a dev group is discarded, not sent to train
    assert not any(i["seed_group"] in dev_groups for i in train)
    # deterministic
    assert bd.split_groups(pool(), dev_per_stratum=4, dev_max_per_group=4, seed=1)[2] == dev_groups


def test_single_group_stratum_stays_in_train():
    only = [item(i, "seed-x", distinct_text("x", i)) for i in range(1, 5)]
    train, dev, _ = bd.split_groups(only, dev_per_stratum=8, dev_max_per_group=8, seed=1)
    assert len(train) == 4 and dev == []


def test_minhash_dedupe_and_reference():
    base = "Your account will be blocked today. Update KYC at http://hdfc-kyc.example.in now"
    a, b = item(1, "g1", base), item(2, "g2", base + "!")
    c = item(3, "g3", "Congratulations you won a lottery of ten lakh rupees, pay the fee")
    kept = bd.dedupe([a, b, c])
    assert [i["id"] for i in kept] == ["syn-1", "syn-3"]
    ref = bd.index_of([a])
    assert [i["id"] for i in bd.dedupe([b, c], ref=ref)] == ["syn-3"]


def test_build_dropout_only_on_train_and_deterministic():
    train, dev, stats = bd.build(
        pool(), test_index=None, dev_per_stratum=4, dev_max_per_group=4, dropout=0.5, seed=7
    )
    assert train and dev
    assert {e["meta"]["split"] for e in train} == {"train"}
    assert {e["meta"]["split"] for e in dev} == {"dev"}

    def empty(e):
        return "RULE_SIGNALS: []" in e["messages"][1]["content"]

    # dropout 0.5 over many train items: some but not all are blanked
    train_empty = sum(empty(e) for e in train)
    assert 0 < train_empty < len(train)
    # with dropout=1.0 every train example is blanked, dev never is by dropout
    train1, dev1, _ = bd.build(
        pool(), test_index=None, dev_per_stratum=4, dev_max_per_group=4, dropout=1.0, seed=7
    )
    assert all(empty(e) for e in train1)
    dev_signal_free = [e for e in dev1 if empty(e)]
    dev_ref = [e for e in dev if empty(e)]
    assert len(dev_signal_free) == len(dev_ref)  # dev identical regardless of dropout rate
    again, _, _ = bd.build(
        pool(), test_index=None, dev_per_stratum=4, dev_max_per_group=4, dropout=0.5, seed=7
    )
    assert [e["meta"]["id"] for e in again] == [e["meta"]["id"] for e in train]
    assert stats["pool"] == len(pool())


def test_render_format_and_meta_keys():
    ex = bd.render(item(1, "g", distinct_text("r", 1)), "dev", dropout=0.0, seed=1)
    roles = [m["role"] for m in ex["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert ex["messages"][1]["content"].startswith(
        "CHANNEL: sms\nSENDER: +91 98100 22334 (unregistered)"
    )
    target = json.loads(ex["messages"][2]["content"])
    assert list(target) == ["verdict", "category", "red_flags"]
    assert set(ex["meta"]) == {
        "id", "split", "verdict", "category", "language", "source", "source_phone",
        "obfuscated", "seed_group", "hard_negative",
    }  # fmt: skip


def test_invalid_items_are_dropped():
    bad_quote = item(
        1, "g", "Hello there friend", flags=[{"quote": "nope", "reason": "urgency_deadline"}]
    )
    bad_reason = item(
        2, "g", "Your account will be blocked", flags=[{"quote": "will be blocked", "reason": "x"}]
    )
    no_flags = item(3, "g", "Your account will be blocked", flags=[])
    for bad in (bad_quote, bad_reason, no_flags):
        assert bd.valid_item(bad) is None
    assert bd.valid_item(item(4, "g", "Your account will be blocked")) is not None


def test_leakage_check_drops_test_matches(tmp_path):
    leaked = distinct_text("leak", 99)
    p = [*pool(), item(900, "seed-leak", leaked)]
    test = tmp_path / "test.jsonl"
    user = f"CHANNEL: sms\nSENDER: x (unknown)\nRULE_SIGNALS: []\nMESSAGE:\n{leaked}"
    test.write_text(json.dumps({"messages": [{}, {"content": user}, {}], "meta": {}}) + "\n")
    idx = bd.load_test_index(test)
    train, dev, stats = bd.build(p, test_index=idx, dev_per_stratum=4, dev_max_per_group=4)
    assert stats["after_leakage"] == len(p) - 1
    assert "syn-900" not in {e["meta"]["id"] for e in train + dev}


def run_main(tmp_path, extra=()):
    syn = tmp_path / "syn"
    syn.mkdir(exist_ok=True)
    rows = [
        {
            "id": i["id"], "text": i["text"], "sender": i["sender"], "channel": "sms",
            "verdict": i["verdict"], "category": i["category"], "red_flags": i["red_flags"],
            "language": "en",
            "meta": {"source": "synthetic", "seed_group": i["seed_group"], "obfuscated": False,
                     "hard_negative": i["hard_negative"]},
        }
        for i in pool()
    ]  # fmt: skip
    (syn / "batch1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (syn / "review_batch1.jsonl").write_text(json.dumps(rows[0]) + "\n")  # must be ignored
    out = tmp_path / "splits"
    argv = ["--synthetic", str(syn), "--redacted", str(tmp_path / "none"), "--out", str(out),
            "--seeds", str(tmp_path / "noseeds.jsonl"), "--dev-per-stratum", "4",
            "--dev-max-per-group", "4", *extra]  # fmt: skip
    return bd.main(argv), out


def test_main_provisional_lock_without_test(tmp_path, capsys):
    rc, out = run_main(tmp_path)
    assert rc == 0
    assert "WARNING" in capsys.readouterr().err
    assert not (out / "test.jsonl").exists()
    lock = json.loads((out / "splits.lock.json").read_text())
    assert lock["test"] is None and lock["provisional"] is True
    import hashlib

    for name in ("train", "dev"):
        data = (out / f"{name}.jsonl").read_bytes()
        assert lock[name]["sha256"] == hashlib.sha256(data).hexdigest()
        assert lock[name]["n"] == len(data.splitlines())


def test_main_preserves_existing_test_entry(tmp_path):
    out = tmp_path / "splits"
    out.mkdir()
    leaked = distinct_text("zz", 5)
    user = f"CHANNEL: sms\nSENDER: x (unknown)\nRULE_SIGNALS: []\nMESSAGE:\n{leaked}"
    (out / "test.jsonl").write_text(json.dumps({"messages": [{}, {"content": user}, {}]}) + "\n")
    (out / "splits.lock.json").write_text(json.dumps({"test": {"sha256": "abc", "n": 1}}))
    before = (out / "test.jsonl").read_bytes()
    rc, _ = run_main(tmp_path)
    lock = json.loads((out / "splits.lock.json").read_text())
    assert rc == 0 and lock["test"] == {"sha256": "abc", "n": 1} and lock["provisional"] is False
    assert (out / "test.jsonl").read_bytes() == before


def test_refuses_to_write_test_jsonl(tmp_path):
    with pytest.raises(bd.BuildError):
        bd.write_split(tmp_path / "test.jsonl", b"{}\n")
    assert not (tmp_path / "test.jsonl").exists()


def test_fill_placeholders_consistent_in_text_and_quotes():
    import random

    it = item(
        1, "g", "Code <OTP> for A/c XX<ACCT>. Ref <ACCT>. Share it now", "SAFE", "genuine_otp"
    )
    out = bd.fill_placeholders(it, random.Random(1))
    assert "<" not in out["text"] and out["text"].startswith("Code ")
    code = out["text"].split()[1]
    assert code.isdigit() and 4 <= len(code) <= 6
    assert out["text"].split("XX")[1][:4].isdigit()
    scam = item(
        2, "g", "Send <OTP> now", flags=[{"quote": "Send <OTP>", "reason": "asks_otp_or_pin"}]
    )
    got = bd.fill_placeholders(scam, random.Random(2))
    assert got["red_flags"][0]["quote"] in got["text"] and "<" not in got["red_flags"][0]["quote"]
    cut = item(
        3, "g", "Use <OTP> today", flags=[{"quote": "OTP> today", "reason": "urgency_deadline"}]
    )
    assert bd.fill_placeholders(cut, random.Random(3)) is None  # only flag cut a placeholder


def test_diversify_is_deterministic_and_keeps_quotes_valid():
    items = [item(i, f"g{i}", distinct_text("d", i)) for i in range(1, 301)]
    a = [bd.diversify(it, 5) for it in items]
    assert a == [bd.diversify(it, 5) for it in items]
    assert all(x["red_flags"][0]["quote"] in x["text"] for x in a)
    shots = sum(x["channel"] == "screenshot" for x in a)
    assert 0.08 < shots / len(a) < 0.25
    with_ref = sum(len(x["text"]) > len(y["text"]) for x, y in zip(a, items, strict=True))
    assert 0.2 < with_ref / len(a) < 0.5
    safe = [item(i, "s", "Hi mama, home safe", "SAFE", "personal") for i in range(400, 700)]
    out = [bd.diversify(it, 5) for it in safe]
    assert {x["sender"] is None for x in out} == {True, False}
    assert any(x["sender"] and x["sender"].startswith("+91") for x in out)


def test_build_leaves_no_placeholders_and_reports_shortcuts():
    p = pool()
    for it in p:
        if it["verdict"] == "SAFE":
            it["text"] += " A/c XX<ACCT> code <OTP>"
    train, dev, _ = bd.build(p, test_index=None, dev_per_stratum=4, dev_max_per_group=4)
    assert not any(
        "<OTP>" in e["messages"][1]["content"] or "<ACCT>" in e["messages"][1]["content"]
        for e in train + dev
    )
    rep = bd.shortcut_report(train)
    assert set(rep) == {"SAFE", "SCAM"} and rep["SAFE"]["digit_run"] > 0.9
