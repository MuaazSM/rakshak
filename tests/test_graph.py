"""LangGraph pipeline end to end with fake detector, explainer, perception and alerts."""

import pytest

from hub import alerts, db, detector, explainer, graph, perception
from tests.test_hub_env import anyio_backend, hub_env, make_result  # noqa: F401

pytestmark = pytest.mark.anyio

SCAM_TEXT = "Dear customer your account will be blocked today. Share your OTP now."
OTP_TEXT = "123456 is your OTP for login. Do not share it with anyone. - HDFC Bank"
APK_TEXT = "Your KYC expired. Install the app from http://sbi-kyc-update.in/sbi.apk today."


class Fakes:
    def __init__(self):
        self.detect_calls: list[tuple] = []
        self.explain_calls: list[tuple] = []
        self.alerts: list[tuple] = []
        self.result = make_result("SCAM", 0.97)
        self.detect_error: Exception | None = None
        self.explain_error: Exception | None = None


@pytest.fixture
def fakes(hub_env, monkeypatch):  # noqa: F811
    f = Fakes()

    async def fake_detect(channel, sender, signals, text, *, client=None):
        f.detect_calls.append((channel, sender, list(signals), text))
        if f.detect_error:
            raise f.detect_error
        return f.result

    async def fake_explain(verdict, parent, lang, input_text, *, client=None):
        f.explain_calls.append((dict(verdict), parent.id, lang, input_text))
        if f.explain_error:
            raise f.explain_error
        return f"EXPLAIN {verdict['verdict']} {lang}"

    async def fake_notify(parent, category, when, *, client=None):
        f.alerts.append((parent.id, category))
        return True

    monkeypatch.setattr(detector, "detect", fake_detect)
    monkeypatch.setattr(explainer, "explain", fake_explain)
    monkeypatch.setattr(alerts, "notify_scam", fake_notify)
    db.init_db()
    return f


async def test_scam_end_to_end(fakes):
    v = await graph.run_check(
        parent_id="mom", text=SCAM_TEXT, channel="sms", sender="+91 98765 43210"
    )
    assert v.verdict == "SCAM" and v.category == "kyc_account_block" and v.p_scam == 0.97
    assert v.parent_id == "mom" and v.language == "en" and v.explanation == "EXPLAIN SCAM en"
    assert v.event_id.startswith("evt_")
    assert [f.quote for f in v.red_flags] == ["account will be blocked"]
    assert v.model_versions == {"detector": "rakshak-detector-v1-q4km", "gemma": "gemma4:e2b"}
    for node in ("normalize", "perceive", "rules", "detect", "fuse", "explain", "total"):
        assert node in v.timings_ms
    channel, sender, signals, text = fakes.detect_calls[0]
    assert (channel, sender) == ("sms", "+91 98765 43210") and text == SCAM_TEXT
    assert isinstance(signals, list)
    assert fakes.alerts == [("mom", "kyc_account_block")]
    stored = db.get_verdict(v.event_id)
    assert stored is not None and stored["verdict"] == "SCAM" and stored["parent_id"] == "mom"


async def test_safe_message_no_alert(fakes):
    fakes.result = make_result("SAFE", 0.02, category="genuine_otp", flags=[])
    v = await graph.run_check(parent_id="dad", text=OTP_TEXT, channel="sms", sender="AX-HDFCBK")
    assert v.verdict == "SAFE" and v.category == "genuine_otp" and v.red_flags == []
    assert fakes.alerts == []


async def test_suspicious_no_alert(fakes):
    fakes.result = make_result("SUSPICIOUS", 0.5, category="other_scam")
    v = await graph.run_check(parent_id="mom", text=SCAM_TEXT, channel="whatsapp")
    assert v.verdict == "SUSPICIOUS" and fakes.alerts == []


async def test_detector_down_is_unknown_never_safe(fakes):
    fakes.detect_error = detector.DetectorUnavailable("ConnectError")
    v = await graph.run_check(parent_id="mom", text="Hi, are you coming for dinner?", channel="sms")
    assert v.verdict == "UNKNOWN" and v.category is None and v.p_scam is None
    assert v.explanation == "EXPLAIN UNKNOWN en"
    assert fakes.alerts == []
    assert db.get_verdict(v.event_id)["verdict"] == "UNKNOWN"


async def test_detector_down_errors_recorded_by_type(fakes):
    fakes.detect_error = detector.DetectorUnavailable("ConnectError")
    final = await graph.get_graph().ainvoke(
        {"event_id": "evt_x", "parent_id": "mom", "channel": "sms", "raw_text": SCAM_TEXT}
    )
    assert "detect:DetectorUnavailable" in final["errors"]
    assert all(SCAM_TEXT not in e for e in final["errors"])


async def test_detector_invalid_json_falls_back_to_rules_only(fakes):
    fakes.result = None
    v = await graph.run_check(parent_id="mom", text=APK_TEXT, channel="sms")
    assert v.verdict == "SCAM" and v.category == "malicious_apk" and v.p_scam is None
    assert v.red_flags and all(f.source == "rule" for f in v.red_flags)
    assert all(f.quote in APK_TEXT for f in v.red_flags)
    assert fakes.alerts == [("mom", "malicious_apk")]


async def test_hard_rule_overrides_safe_detector(fakes):
    fakes.result = make_result("SAFE", 0.01, category="personal", flags=[])
    v = await graph.run_check(parent_id="mom", text=APK_TEXT, channel="sms")
    assert v.verdict == "SCAM" and v.category == "malicious_apk"


async def test_image_path_uses_perception_and_drops_media(fakes, monkeypatch):
    seen = {}

    async def fake_image(image, mime, *, client=None):
        seen["args"] = (image, mime)
        return "AX-HDFCBK", SCAM_TEXT

    monkeypatch.setattr(perception, "transcribe_image", fake_image)
    v = await graph.run_check(
        parent_id="mom", channel="sms", image=b"\x89PNG-bytes", image_mime="image/png"
    )
    assert seen["args"] == (b"\x89PNG-bytes", "image/png")
    channel, sender, _, text = fakes.detect_calls[0]
    assert (channel, sender, text) == ("screenshot", "AX-HDFCBK", SCAM_TEXT)
    assert v.verdict == "SCAM"
    final = await graph.get_graph().ainvoke(
        {
            "event_id": "evt_y",
            "parent_id": "mom",
            "channel": "screenshot",
            "image": b"raw",
            "raw_text": None,
        }
    )
    assert final.get("image") is None and final.get("audio") is None
    assert db.get_verdict(v.event_id) is not None


async def test_explicit_sender_beats_transcribed_sender(fakes, monkeypatch):
    async def fake_image(image, mime, *, client=None):
        return "OTHER", SCAM_TEXT

    monkeypatch.setattr(perception, "transcribe_image", fake_image)
    await graph.run_check(parent_id="mom", channel="sms", sender="AX-ME", image=b"i")
    assert fakes.detect_calls[0][1] == "AX-ME"


async def test_audio_path_is_call_description(fakes, monkeypatch):
    async def fake_audio(audio, mime, *, client=None):
        assert (audio, mime) == (b"m4a", "audio/mp4")
        return "Someone called saying my account will be blocked and asked for my OTP"

    monkeypatch.setattr(perception, "transcribe_audio", fake_audio)
    v = await graph.run_check(parent_id="dad", channel="sms", audio=b"m4a", audio_mime="audio/mp4")
    assert fakes.detect_calls[0][0] == "call_description" and v.verdict == "SCAM"


async def test_perception_failure_is_unknown_and_skips_detector(fakes, monkeypatch):
    async def boom(image, mime, *, client=None):
        raise perception.PerceptionError("unreadable")

    monkeypatch.setattr(perception, "transcribe_image", boom)
    v = await graph.run_check(parent_id="mom", channel="sms", image=b"junk")
    assert v.verdict == "UNKNOWN" and v.p_scam is None and v.category is None
    assert fakes.detect_calls == [] and fakes.alerts == []
    assert v.explanation == "EXPLAIN UNKNOWN en"


async def test_perception_unexpected_error_is_also_unknown(fakes, monkeypatch):
    async def boom(audio, mime, *, client=None):
        raise RuntimeError("whisper crashed")

    monkeypatch.setattr(perception, "transcribe_audio", boom)
    v = await graph.run_check(parent_id="mom", channel="sms", audio=b"x")
    assert v.verdict == "UNKNOWN"


async def test_language_follows_profile_and_override(fakes):
    await graph.run_check(parent_id="mom", text=SCAM_TEXT, channel="sms")
    assert fakes.explain_calls[-1][2] == "en"
    v = await graph.run_check(parent_id="mom", text=SCAM_TEXT, channel="sms", lang="hi")
    assert fakes.explain_calls[-1][2] == "hi" and v.language == "hi"


async def test_explainer_receives_verdict_json_and_text(fakes):
    await graph.run_check(parent_id="mom", text=SCAM_TEXT, channel="sms")
    verdict, parent, _, input_text = fakes.explain_calls[0]
    assert set(verdict) == {"verdict", "category", "red_flags"} and parent == "mom"
    assert input_text == SCAM_TEXT


async def test_explainer_crash_falls_back_to_template(fakes):
    fakes.explain_error = RuntimeError("gemma down")
    v = await graph.run_check(parent_id="mom", text=SCAM_TEXT, channel="sms")
    assert v.verdict == "SCAM" and v.explanation and not v.explanation.startswith("EXPLAIN")


async def test_alert_failure_does_not_break_check(fakes, monkeypatch):
    async def boom(parent, category, when, *, client=None):
        raise RuntimeError("ntfy down")

    monkeypatch.setattr(alerts, "notify_scam", boom)
    v = await graph.run_check(parent_id="mom", text=SCAM_TEXT, channel="sms")
    assert v.verdict == "SCAM" and db.get_verdict(v.event_id) is not None


async def test_keep_media_writes_file_only_when_enabled(fakes, hub_env, monkeypatch):  # noqa: F811
    from hub import settings

    async def fake_image(image, mime, *, client=None):
        return None, SCAM_TEXT

    monkeypatch.setattr(perception, "transcribe_image", fake_image)
    await graph.run_check(parent_id="mom", channel="sms", image=b"IMG")
    assert not (hub_env / "var" / "media").exists()
    monkeypatch.setenv("KEEP_MEDIA", "true")
    settings.get_settings.cache_clear()
    v = await graph.run_check(parent_id="mom", channel="sms", image=b"IMG")
    assert (hub_env / "var" / "media" / f"{v.event_id}.img").read_bytes() == b"IMG"


async def test_bad_requests(fakes):
    with pytest.raises(graph.UnknownParent):
        await graph.run_check(parent_id="uncle", text="hi", channel="sms")
    with pytest.raises(graph.EmptyInput):
        await graph.run_check(parent_id="mom", text="   ", channel="sms")
    assert fakes.detect_calls == []


async def test_event_ids_unique(fakes):
    ids = [graph.new_event_id() for _ in range(50)]
    assert len(set(ids)) == 50
