"""Training script helpers (PRD FR-33, §11). No Tinker calls."""

from pathlib import Path

import pytest

from training import train_tinker as T


def test_refuses_test_split(tmp_path):
    test = tmp_path / "test.jsonl"
    test.write_text("{}\n")
    with pytest.raises(T.TrainError):
        T.guard_split(test, "train")
    with pytest.raises(T.TrainError):
        T.guard_split(T.SPLITS / "test.jsonl", "dev")


def test_refuses_missing_file(tmp_path):
    with pytest.raises(T.TrainError):
        T.guard_split(tmp_path / "train.jsonl", "train")


def test_main_refuses_test_split_before_any_api_call(tmp_path, capsys):
    test = tmp_path / "test.jsonl"
    test.write_text("{}\n")
    assert T.main(["--train", str(test), "--dry-run"]) == 2
    assert "test split" in capsys.readouterr().err


def test_select_rows_seeded_and_limited():
    rows = [{"i": i} for i in range(100)]
    a = T.select_rows(rows, 10, seed=0)
    assert a == T.select_rows(rows, 10, seed=0)
    assert len(a) == 10 and a != rows[:10]
    assert len(T.select_rows(rows, None, seed=0)) == 100


def test_epoch_batches_cover_all_and_keep_partial():
    b = T.epoch_batches(50, 32, epoch=0, seed=0)
    assert [len(x) for x in b] == [32, 18]
    assert sorted(i for x in b for i in x) == list(range(50))
    assert T.epoch_batches(50, 32, 1, 0) != b  # reshuffled per epoch
    assert T.epoch_batches(50, 32, 0, 0) == b  # deterministic


def test_estimate_cost_matches_prd_budget_arithmetic():
    # PRD §11.2: ≈ 2,000 ex × 450 tok × 3 epochs ≈ 2.7 M tokens × $0.737/M ≈ $2
    est = T.estimate_cost(2000 * 450, 3, 0, 0, 0)
    assert est["train_tokens"] == 2_700_000
    assert est["train_usd"] == pytest.approx(1.99, abs=0.01)
    assert est["total_usd"] == est["train_usd"]
    est = T.estimate_cost(1000, 1, 3000, 1000, 2)
    assert est["dev_eval_prompt_tokens"] == 6000 and est["dev_eval_sample_tokens"] == 2000
    assert est["total_usd"] == pytest.approx((1000 + 8000) * 0.737 / 1e6, abs=1e-4)


def test_defaults_match_prd_11_1():
    import argparse

    seen = {}

    def fake_train(args: argparse.Namespace) -> int:
        seen.update(vars(args))
        return 0

    orig = T.train
    T.train = fake_train
    try:
        assert T.main([]) == 0
    finally:
        T.train = orig
    assert (seen["epochs"], seen["rank"], seen["batch"], seen["max_len"]) == (3, 32, 32, 1024)
    assert seen["lr"] is None  # → hyperparam_utils.get_lr(base)
    assert seen["train"] == Path(T.SPLITS / "train.jsonl")
