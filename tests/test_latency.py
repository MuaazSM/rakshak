"""Latency script (PRD §12.1, NFR-2) against a mocked hub. Synthetic items, no network."""

import json

import httpx

from eval import latency
from eval.systems import item_from_row
from tests.test_eval_systems import OTP_TEXT, SAFE_OUT, SCAM_OUT, SCAM_TEXT, row

TIMINGS = {"normalize": 1, "perceive": 0, "rules": 2, "detect": 800, "fuse": 0, "explain": 1500}


def _items(n=8):
    rows = [
        row(f"i{k}", SCAM_TEXT, SCAM_OUT)
        if k % 2
        else row(f"i{k}", OTP_TEXT, SAFE_OUT, sender=None)
        for k in range(n)
    ]
    return [item_from_row(r) for r in rows]


def _hub(seen, fail_on=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={"status": "ok", "model_versions": {"detector": "rakshak-detector-v1-q4km"}},
            )
        body = json.loads(request.content)
        seen.append(body)
        if fail_on and body["text"] == fail_on:
            return httpx.Response(500, json={})
        verdict = "SCAM" if "apk" in body["text"] else "SAFE"
        return httpx.Response(
            200, json={"verdict": verdict, "timings_ms": {**TIMINGS, "total": 2303}}
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_warmup_then_sequential_measured_requests(capsys):
    items, seen = _items(), []
    records, hl = latency.run(items, 4, "http://hub", "mom", _hub(seen))
    assert len(seen) == 3 + 4
    measured = seen[3:]
    assert [b["text"] for b in measured] == [it.text for it in items[:4]]
    assert all(b["parent_id"] == "mom" for b in seen)
    assert measured[0] == {"parent_id": "mom", "text": OTP_TEXT, "channel": "sms"}
    assert measured[1]["sender"] == "VK-SBIUPD" and measured[1]["channel"] == "sms"
    assert [r["verdict"] for r in records] == ["SAFE", "SCAM", "SAFE", "SCAM"]
    assert hl["model_versions"]["detector"] == "rakshak-detector-v1-q4km"
    out = capsys.readouterr().out
    assert "blocked" not in out and "OTP is" not in out  # never prints message text


def test_summary_and_result_file(tmp_path, monkeypatch):
    monkeypatch.setattr(latency, "hardware", lambda: "MacBook Pro M3 Pro 18 GB")
    items, seen = _items(), []
    records, hl = latency.run(items, 4, "http://hub", "mom", _hub(seen, fail_on=SCAM_TEXT))
    dev = tmp_path / "dev.jsonl"
    dev.write_text("x\n")
    res = latency.build_result(records, hl, items[:4], dev, "http://hub", 3)
    assert res["provisional"] is True and res["split"] == "dev_synthetic"
    assert res["n"] == 4 and res["errors"] == {"HTTPStatusError": 2}
    assert res["detector_version"] == "rakshak-detector-v1-q4km"
    assert res["hardware"] == "MacBook Pro M3 Pro 18 GB"
    assert res["nodes"]["detect"]["p50_ms"] == 800 and res["nodes"]["explain"]["p95_ms"] == 1500
    assert res["nodes"]["total"]["n"] == 2
    assert res["end_to_end_wall"]["n"] == 2 and res["end_to_end_wall"]["p50_ms"] > 0
    assert set(res["end_to_end_by_verdict"]) == {"SAFE"}
    blob = json.dumps(res)
    assert "blocked" not in blob and "OTP is" not in blob


def test_main_refuses_test_split(tmp_path, capsys):
    assert latency.main(["--dev", str(tmp_path / "test.jsonl")]) == 2


def test_hardware_falls_back_without_sysctl(monkeypatch):
    monkeypatch.setattr(latency, "_run", lambda cmd: None)
    assert latency.hardware()
