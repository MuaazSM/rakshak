"""Eval harness: test-split guard, results files, few-shot selection (PRD FR-35, §12, A.4)."""

import json

import pytest

from eval import baselines, run_eval
from eval.metrics import Pred
from eval.run_eval import EvalRefused, build_result, format_table, split_path, write_result
from eval.systems import item_from_row
from tests.test_eval_systems import OTP_TEXT, SAFE_OUT, SCAM_OUT, SCAM_TEXT, row


def test_test_split_refused_without_flag(tmp_path):
    (tmp_path / "test.jsonl").write_text("{}\n")
    with pytest.raises(EvalRefused):
        split_path("test", final_eval=False, splits_dir=tmp_path)


def test_test_split_refused_when_missing_even_with_flag(tmp_path):
    with pytest.raises(EvalRefused):
        split_path("test", final_eval=True, splits_dir=tmp_path)


def test_test_split_allowed_only_with_flag_and_file(tmp_path):
    (tmp_path / "test.jsonl").write_text("{}\n")
    assert split_path("test", final_eval=True, splits_dir=tmp_path).name == "test.jsonl"
    assert split_path("dev", final_eval=False, splits_dir=tmp_path).name == "dev.jsonl"
    with pytest.raises(EvalRefused):
        split_path("holdout", final_eval=False, splits_dir=tmp_path)


def test_main_refuses_test_split(capsys):
    assert run_eval.main(["--systems", "rules_only", "--split", "test"]) == 2
    assert "refused" in capsys.readouterr().err


def _items():
    rows = [
        row("s1", SCAM_TEXT, SCAM_OUT, language="en", obfuscated=False),
        row("o1", OTP_TEXT, SAFE_OUT, language="hinglish", obfuscated=True),
    ]
    return [item_from_row(r) for r in rows]


def test_result_is_provisional_on_synthetic_and_has_no_text(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "ROOT", tmp_path)
    split = tmp_path / "dev.jsonl"
    split.write_text("x\n")
    items = _items()
    preds = [
        Pred("SCAM", "malicious_apk", ["will be blocked today"], json_valid=True, latency_ms=5),
        Pred("SAFE", "genuine_otp", [], json_valid=True, latency_ms=7),
    ]
    res = build_result("demo", "dev", split, items, preds)
    assert res["provisional"] is True and res["split"] == "dev_synthetic"
    assert res["metrics"]["macro_f1"] == 1.0
    assert set(res["slices"]) == {"language", "category", "obfuscated", "source"}
    assert res["slices"]["language"]["hinglish"]["n"] == 1
    assert set(res["ci95"]) >= {"scam_recall", "fpr_genuine", "macro_f1"}
    out = write_result(res, items, preds, tmp_path / "results")
    assert out.name == "demo_dev.json"
    blob = out.read_text() + (tmp_path / "results" / "demo_dev.items.jsonl").read_text()
    for secret in ("blocked today", "OTP is", "sbi-help"):
        assert secret not in blob
    item = json.loads((tmp_path / "results" / "demo_dev.items.jsonl").read_text().splitlines()[0])
    assert item["n_grounded"] == 1 and item["pred_verdict"] == "SCAM"
    table = format_table([res])
    assert table.startswith("PROVISIONAL (synthetic dev)")
    assert "| demo | 2 |" in table


def test_not_provisional_when_real_items_present():
    items = _items()
    items[0].meta["source"] = "own_inbox"
    assert not run_eval.is_provisional(items)


def test_git_sha_reads_files(tmp_path):
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "refs" / "heads" / "main").write_text("abc123\n")
    assert run_eval.git_sha(tmp_path) == "abc123"
    (git / "refs" / "heads" / "main").unlink()
    (git / "packed-refs").write_text("# pack\ndef456 refs/heads/main\n")
    assert run_eval.git_sha(tmp_path) == "def456"
    assert run_eval.git_sha(tmp_path / "nope") is None


def _train_items():
    out = []
    scam_cats = ["kyc_account_block", "digital_arrest", "lottery_prize"]
    for i, (cat, lang) in enumerate(zip(scam_cats, ["en", "hinglish", "hi"], strict=True)):
        o = {"verdict": "SCAM", "category": cat, "red_flags": []}
        out.append(row(f"scam{i}", f"scam text {i}", o, language=lang))
    for i, cat in enumerate(["genuine_otp", "transaction_alert", "genuine_otp"]):
        out.append(row(f"hn{i}", f"genuine {i}", {**SAFE_OUT, "category": cat}))
    out.append(
        row(
            "sus0",
            "is this your number?",
            {**SAFE_OUT, "verdict": "SUSPICIOUS", "category": "other_scam"},
        )
    )
    out.append(row("per0", "dinner at 8?", {**SAFE_OUT, "category": "personal"}))
    return [item_from_row(r) for r in out]


def test_fewshot_selection_follows_a4():
    sel = baselines.select_fewshot(_train_items())
    kinds = [k for k, _ in sel]
    assert kinds == list(baselines.ORDER)
    scams = [it for k, it in sel if k == "scam"]
    assert scams[0].gold.category != scams[1].gold.category
    assert scams[0].meta["language"] != scams[1].meta["language"]
    hn = sorted(it.gold.category for k, it in sel if k == "hard_negative")
    assert hn == ["genuine_otp", "transaction_alert"]
    assert len({it.id for _, it in sel}) == 6
    assert baselines.select_fewshot(_train_items()) == sel  # deterministic


def test_fewshot_selection_fails_loudly_when_kind_missing():
    items = [it for it in _train_items() if it.gold.verdict != "SUSPICIOUS"]
    with pytest.raises(ValueError, match="suspicious"):
        baselines.select_fewshot(items)


def test_build_fewshot_refuses_test_file(tmp_path):
    with pytest.raises(ValueError):
        baselines.build_fewshot(tmp_path / "test.jsonl", tmp_path / "f.jsonl")


def test_build_fewshot_writes_user_assistant_pairs(tmp_path):
    train = tmp_path / "train.jsonl"
    rows = [it.messages for it in _train_items()]
    train.write_text(
        "".join(
            json.dumps({"messages": m, "meta": it.meta}) + "\n"
            for m, it in zip(rows, _train_items(), strict=True)
        )
    )
    out = tmp_path / "fewshot.jsonl"
    lines = baselines.build_fewshot(train, out)
    assert len(lines) == 6
    first = json.loads(out.read_text().splitlines()[0])
    assert [m["role"] for m in first["messages"]] == ["user", "assistant"]
