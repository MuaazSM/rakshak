"""Run summary for notes/failures.md (P4.2). Synthetic fixture files only."""

import json

from eval import error_analysis


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_summary_per_run_epoch_and_suspicious_groups(tmp_path):
    dev = tmp_path / "dev.jsonl"
    _write(
        dev, [{"meta": {"id": f"i{k}", "seed_group": "g1" if k < 2 else "g2"}} for k in range(3)]
    )
    run = tmp_path / "runs" / "20261003-000000-full-lr1x"
    run.mkdir(parents=True)
    dev_metrics = dict.fromkeys(error_analysis.KEEP, 0.5)
    (run / "config.json").write_text(
        json.dumps(
            {
                "run_id": run.name,
                "status": "done",
                "hyperparams": {"lr": 1e-4, "epochs": 1},
                "data": {"dev": {"sha256": "d"}, "train": {"sha256": "t"}},
            }
        )
    )
    (run / "metrics.json").write_text(
        json.dumps(
            {
                "provisional": True,
                "split": "dev_synthetic",
                "best": {"epoch": 1},
                "epochs": [{"epoch": 1, "train_loss_mean": 0.1, "dev": dev_metrics}],
            }
        )
    )
    _write(
        run / "tuned_tinker_e1_dev.items.jsonl",
        [
            {"id": "i0", "gold_verdict": "SUSPICIOUS", "pred_verdict": "SAFE"},
            {"id": "i1", "gold_verdict": "SUSPICIOUS", "pred_verdict": "SUSPICIOUS"},
            {"id": "i2", "gold_verdict": "SCAM", "pred_verdict": "SCAM"},
        ],
    )
    (tmp_path / "runs" / "20261003-000000-full-x-crashed").mkdir()
    s = error_analysis.build(tmp_path / "runs", "*-full-*", dev)
    assert s["provisional"] is True and s["split"] == "dev_synthetic"
    [r] = s["runs"]
    assert r["best_epoch"] == 1 and r["lr"] == 1e-4
    [e] = r["epochs"]
    assert e["macro_f1"] == 0.5
    assert e["suspicious_pred_by_seed_group"] == {"g1": {"SAFE": 1, "SUSPICIOUS": 1}}


def test_refuses_test_split(tmp_path):
    assert error_analysis.main(["--dev", str(tmp_path / "test.jsonl")]) == 2
