"""PRD §13, §7.4, FR-3, FR-11, FR-13, FR-16, FR-17: what the hub does when things go wrong.

Real app, real graph, rules, fusion, perception, explainer and alerts code; only the network
is faked (one httpx mock transport for llama-server, Ollama, Gemma audio and ntfy). The rule
that runs through every test: a failure may make the answer more cautious or say "couldn't
check", but it must never produce SAFE.
"""

import io
import json
import shutil
import wave

import httpx
import pytest
from fastapi.testclient import TestClient

from hub import app as hub_app
from hub import explainer, perception
from hub.settings import get_settings
from tests.test_hub_env import hub_env, lp_entry  # noqa: F401

BENIGN = "Hi, are you coming over for dinner tonight?"
SCAM = "Dear customer your account will be blocked today. Share your OTP now."
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32  # sniffs as PNG but is not a decodable image
HEIC = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic" + b"0" * 64
SCAM_EXPLANATION = "This is a scam. Do not share any code with anyone. Call Muaaz."
COULDNT_CHECK = "couldn't check"


class Net:
    """Behaviour switches for the fake network, plus a log of what was called."""

    def __init__(self):
        self.detector = "ok"  # ok | down | http500 | not_json | bad_schema | retry_ok | timeout
        self.ollama = "ok"  # ok | down | empty | foreign_url
        self.audio = "ok"  # ok | down
        self.ntfy = "ok"  # ok | down
        self.verdict = "SCAM"
        self.image_text = SCAM
        self.calls: list[httpx.Request] = []

    def to(self, port: int | None = None, host: str | None = None) -> list[httpx.Request]:
        return [
            r
            for r in self.calls
            if (port is None or r.url.port == port) and (host is None or r.url.host == host)
        ]


def _detector_response(verdict: str) -> httpx.Response:
    content = {
        "verdict": verdict,
        "category": "kyc_account_block" if verdict == "SCAM" else "personal",
        "red_flags": [],
    }
    rest = ',"category":"x","red_flags":[]}'
    lp = {
        "content": [
            lp_entry('{"verdict":"', {'{"verdict":"': 1.0}),
            lp_entry(verdict, {"SCAM": 0.97, "SUSPICIOUS": 0.02, "SAFE": 0.01})
            if verdict == "SCAM"
            else lp_entry(verdict, {"SCAM": 0.01, "SUSPICIOUS": 0.01, "SAFE": 0.98}),
            lp_entry(rest, {rest: 1.0}),
        ]
    }
    choice = {"message": {"content": json.dumps(content)}, "logprobs": lp}
    return httpx.Response(200, json={"choices": [choice]})


def _handler(net: Net):
    def handle(request: httpx.Request) -> httpx.Response:
        net.calls.append(request)
        host, port = request.url.host, request.url.port
        if host == "ntfy.sh":
            if net.ntfy == "down":
                raise httpx.ConnectError("ntfy unreachable")
            return httpx.Response(200)
        if port == 8081:
            n = len(net.to(8081))
            mode = net.detector
            if mode == "down":
                raise httpx.ConnectError("llama-server down")
            if mode == "timeout":
                raise httpx.ReadTimeout("slow")
            if mode == "http500":
                return httpx.Response(500, text="boom")
            if mode == "bad_schema":
                return httpx.Response(200, json={"unexpected": True})
            if mode == "not_json" or (mode == "retry_ok" and n == 1):
                msg = {"message": {"content": "Sure! Here is my answer: not json"}}
                return httpx.Response(200, json={"choices": [msg]})
            return _detector_response(net.verdict)
        if port == 8082:
            if net.audio == "down":
                raise httpx.ConnectError("audio server down")
            return httpx.Response(200, json={"choices": [{"message": {"content": SCAM}}]})
        if port == 11434:
            if net.ollama == "down":
                raise httpx.ConnectError("ollama down")
            body = json.loads(request.content)
            if body["messages"][0].get("images"):
                return httpx.Response(
                    200,
                    json={"message": {"content": f"SENDER: unknown\nMESSAGE:\n{net.image_text}"}},
                )
            text = {
                "ok": SCAM_EXPLANATION,
                "empty": "",
                "foreign_url": "This is a scam. Open http://evil.example now and call 9999999999.",
            }[net.ollama]
            return httpx.Response(200, json={"message": {"content": text}})
        raise AssertionError(f"unexpected request to {request.url}")

    return handle


@pytest.fixture
def net(hub_env, monkeypatch):  # noqa: F811
    monkeypatch.setenv("NTFY_TOPIC", "rakshak-test-topic-not-real")
    monkeypatch.setenv("GEMMA_AUDIO_URL", "http://127.0.0.1:8082/v1")
    monkeypatch.setenv("OCR_FALLBACK", "gemma")
    monkeypatch.setenv("ASR_FALLBACK", "gemma")
    get_settings.cache_clear()
    n = Net()
    transport = httpx.MockTransport(_handler(n))
    real = httpx.AsyncClient

    class Mocked(real):  # type: ignore[misc, valid-type]
        def __init__(self, *a, **k):
            k["transport"] = transport
            super().__init__(*a, **k)

    monkeypatch.setattr(httpx, "AsyncClient", Mocked)
    return n


@pytest.fixture
def client(net, hub_env):  # noqa: F811
    with TestClient(hub_app.create_app(dist_dir=hub_env / "no-dist")) as c:
        yield c


def _check(client, text=BENIGN, parent="mom", **extra):
    return client.post(
        "/api/check", json={"parent_id": parent, "text": text, "channel": "sms", **extra}
    )


def _wav(seconds: float, rate: int = 8000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(rate)
        w.writeframes(b"\x80" * int(rate * seconds))
    return buf.getvalue()


def _assert_couldnt_check(v: dict) -> None:
    assert v["verdict"] == "UNKNOWN" and v["category"] is None and v["p_scam"] is None
    assert v["red_flags"] == []
    assert COULDNT_CHECK in v["explanation"].lower()
    assert v["explanation"] == explainer.template("UNKNOWN", None, v["language"], "Muaaz")


# --- detector unavailable ---------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["down", "http500", "timeout", "bad_schema"])
def test_detector_unavailable_is_unknown_never_safe(client, net, mode):
    net.detector = mode
    r = _check(client)
    assert r.status_code == 200
    _assert_couldnt_check(r.json())
    assert net.to(host="ntfy.sh") == []  # no alert for a check that didn't happen
    assert net.to(11434) == []  # UNKNOWN uses the fixed template, Gemma isn't asked


def test_detector_unavailable_for_image_and_share_target_still_unknown(client, net):
    net.detector = "down"
    net.image_text = BENIGN
    r = client.post(
        "/api/share", data={"parent_id": "dad"}, files={"image": ("s.png", PNG, "image/png")}
    )
    _assert_couldnt_check(r.json())
    r = client.post("/share?parent=mom", data={"text": BENIGN}, follow_redirects=False)
    assert r.status_code == 303
    v = client.get(r.headers["location"].replace("/v/", "/api/verdict/")).json()
    _assert_couldnt_check(v)


def test_detector_unavailable_hard_rule_still_flags_scam(client, net):
    """Rules-only mode (PRD §7.4): a hard rule is a SCAM even without the model."""
    net.detector = "down"
    text = "Your KYC expired. Install the app from http://sbi-kyc-update.in/sbi.apk today."
    v = _check(client, text).json()
    assert v["verdict"] == "SCAM" and v["category"] == "malicious_apk" and v["p_scam"] is None
    assert all(f["source"] == "rule" for f in v["red_flags"])


def test_detector_crash_in_node_degrades_to_unknown(client, monkeypatch):
    """Any unexpected exception inside a node falls back toward caution."""
    from hub import detector

    async def boom(*a, **k):
        raise RuntimeError(BENIGN)  # message must not leak anywhere

    monkeypatch.setattr(detector, "detect", boom)
    r = _check(client)
    assert r.status_code == 200
    _assert_couldnt_check(r.json())


# --- detector invalid JSON ---------------------------------------------------------------------


def test_invalid_json_twice_uses_rules_only_and_retries_constrained(client, net):
    net.detector = "not_json"
    v = _check(client).json()
    _assert_couldnt_check(v)
    calls = net.to(8081)
    assert len(calls) == 2
    first, second = (json.loads(c.content) for c in calls)
    assert "response_format" not in first
    assert second["response_format"]["type"] == "json_schema"  # FR-13 constrained retry


def test_invalid_json_twice_with_hard_rule_is_scam(client, net):
    net.detector = "not_json"
    v = _check(client, "Your KYC expired. Install http://sbi-kyc-update.in/sbi.apk today.").json()
    assert v["verdict"] == "SCAM" and v["p_scam"] is None


def test_invalid_json_once_then_valid_uses_the_retry(client, net):
    net.detector = "retry_ok"
    v = _check(client, SCAM).json()
    assert v["verdict"] == "SCAM" and v["p_scam"] is not None
    assert len(net.to(8081)) == 2


# --- Ollama (explainer) down -------------------------------------------------------------------


def test_ollama_down_uses_template_explanation(client, net):
    net.ollama = "down"
    v = _check(client, SCAM).json()
    assert v["verdict"] == "SCAM"  # the verdict is untouched by the explainer
    assert v["explanation"] == explainer.template("SCAM", v["category"], "en", "Muaaz")
    assert "scam" in v["explanation"].lower()
    assert len(net.to(host="ntfy.sh")) == 1  # alert still goes out


@pytest.mark.parametrize("bad", ["empty", "foreign_url"])
def test_bad_explainer_output_falls_back_to_template(client, net, bad):
    net.ollama = bad
    v = _check(client, SCAM).json()
    assert v["verdict"] == "SCAM"
    assert v["explanation"] == explainer.template("SCAM", v["category"], "en", "Muaaz")
    assert "evil.example" not in v["explanation"]


def test_ollama_down_safe_verdict_keeps_safe_template(client, net):
    net.verdict = "SAFE"
    net.ollama = "down"
    v = _check(client, "Your order has been delivered. Thanks for shopping.").json()
    assert v["verdict"] == "SAFE" and "normal message" in v["explanation"].lower()


# --- ntfy down ---------------------------------------------------------------------------------


def test_ntfy_down_does_not_break_the_check(client, net):
    net.ntfy = "down"
    r = _check(client, SCAM)
    assert r.status_code == 200 and r.json()["verdict"] == "SCAM"
    assert client.get(f"/api/verdict/{r.json()['event_id']}").json()["verdict"] == "SCAM"


# --- empty and bad requests --------------------------------------------------------------------


def test_empty_input_is_rejected_on_every_endpoint(client, net):
    assert _check(client, "").status_code == 422
    assert client.post("/api/check", json={"parent_id": "mom", "channel": "sms"}).status_code == 422
    assert client.post("/api/share", data={"parent_id": "dad"}).status_code == 422
    assert client.post("/api/share", data={"parent_id": "dad", "text": "  \n "}).status_code == 422
    assert client.post("/share?parent=mom", data={"title": " "}).status_code == 422
    assert client.post("/share?parent=mom").status_code == 422
    empty_img = {"image": ("s.png", b"", "image/png")}
    assert client.post("/api/share", data={"parent_id": "dad"}, files=empty_img).status_code == 422
    r = client.post("/voice", data={"parent_id": "mom"}, files={"audio": ("a.webm", b"")})
    assert r.status_code == 422
    assert client.post("/voice", data={"parent_id": "mom"}).status_code == 422
    assert net.calls == []  # nothing reached any model


def test_unknown_or_missing_parent_is_400(client, net):
    assert _check(client, parent="uncle").status_code == 400
    assert client.post("/api/share", data={"parent_id": "uncle", "text": "x"}).status_code == 400
    assert client.post("/api/share", data={"text": "x"}).status_code == 400
    assert client.post("/share?parent=uncle", data={"text": "x"}).status_code == 400
    assert client.post("/share", data={"text": "x"}).status_code == 400
    audio = {"audio": ("a.webm", b"x", "audio/webm")}
    assert client.post("/voice", data={"parent_id": "uncle"}, files=audio).status_code == 400
    assert client.post("/voice", files=audio).status_code == 400
    assert client.patch("/api/parents/uncle", json={"language": "en"}).status_code == 400
    body = _check(client, parent="uncle").json()
    assert body["error"] == "unknown_parent" and "uncle" not in json.dumps(body)
    assert net.calls == []


def test_bad_language_is_rejected(client, net):
    assert _check(client, lang="mr").status_code == 422
    r = client.post("/api/share", data={"parent_id": "dad", "text": "hi", "lang": "fr"})
    assert r.status_code == 422


# --- audio limits ------------------------------------------------------------------------------


def test_audio_over_2mb_is_413(client, net):
    big = b"0" * (hub_app.MAX_AUDIO_BYTES + 1)
    r = client.post("/voice", data={"parent_id": "mom"}, files={"audio": ("a.webm", big)})
    assert r.status_code == 413 and r.json()["error"] == "audio_too_large"
    assert net.calls == []


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe needed to measure duration")
def test_audio_over_60_seconds_is_413_with_real_ffprobe(client, net):
    data = _wav(61)
    assert len(data) < hub_app.MAX_AUDIO_BYTES  # rejected for its length, not its size
    r = client.post(
        "/voice", data={"parent_id": "mom"}, files={"audio": ("a.wav", data, "audio/wav")}
    )
    assert r.status_code == 413 and r.json()["error"] == "audio_too_long"
    assert net.calls == []


@pytest.mark.skipif(
    shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None,
    reason="ffmpeg/ffprobe needed",
)
def test_audio_under_60_seconds_is_checked(client, net):
    r = client.post(
        "/voice", data={"parent_id": "mom"}, files={"audio": ("a.wav", _wav(3), "audio/wav")}
    )
    assert r.status_code == 200 and r.json()["verdict"] == "SCAM"
    assert len(net.to(8082)) == 1


# --- perception failures: never SAFE -----------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "data", "mime"),
    [
        ("random bytes", bytes(range(256)) * 4, "image/png"),
        ("heic", HEIC, "image/heic"),
        ("pdf", b"%PDF-1.7\n" + b"0" * 64, "application/pdf"),
        ("png magic, wrong mime", PNG, "text/plain"),
    ],
)
def test_unsupported_image_is_unknown_not_safe(client, net, name, data, mime):
    net.verdict = "SAFE"  # even a detector that would say SAFE must not get to see it
    r = client.post("/api/share", data={"parent_id": "dad"}, files={"image": ("x", data, mime)})
    assert r.status_code == 200, name
    _assert_couldnt_check(r.json())
    assert net.to(8081) == [] and net.to(11434) == [], name  # nothing read, nothing detected
    assert net.to(host="ntfy.sh") == []


def test_image_with_no_text_is_unknown_not_safe(client, net):
    """Gemma sees nothing and Tesseract fails on the undecodable file."""
    net.image_text = ""
    net.verdict = "SAFE"
    r = client.post(
        "/api/share", data={"parent_id": "dad"}, files={"image": ("s.png", PNG, "image/png")}
    )
    assert r.status_code == 200
    _assert_couldnt_check(r.json())
    assert net.to(8081) == []


def test_ollama_down_for_screenshot_falls_back_then_unknown(client, net):
    net.ollama = "down"  # Gemma fails -> Tesseract -> cannot decode the fake PNG
    net.verdict = "SAFE"
    r = client.post(
        "/api/share", data={"parent_id": "dad"}, files={"image": ("s.png", PNG, "image/png")}
    )
    assert r.status_code == 200
    _assert_couldnt_check(r.json())


def test_screenshot_text_survives_ollama_down_via_tesseract(client, net, monkeypatch):
    """Gemma is down but Tesseract reads the picture: the check goes ahead as normal."""

    monkeypatch.setattr(perception, "_tesseract", lambda image: SCAM)
    net.ollama = "down"
    r = client.post(
        "/api/share", data={"parent_id": "dad"}, files={"image": ("s.png", PNG, "image/png")}
    )
    v = r.json()
    assert v["verdict"] == "SCAM"
    assert v["explanation"] == explainer.template("SCAM", v["category"], "en", "Muaaz")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg needed")
def test_undecodable_audio_is_unknown_not_safe(client, net):
    net.verdict = "SAFE"
    r = client.post(
        "/voice",
        data={"parent_id": "mom"},
        files={"audio": ("a.webm", b"not audio at all" * 20, "audio/webm")},
    )
    assert r.status_code == 200
    _assert_couldnt_check(r.json())
    assert net.to(8081) == []


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg needed")
def test_audio_server_down_and_no_whisper_is_unknown(client, net, monkeypatch):
    def no_whisper(wav):
        raise perception.PerceptionError("mlx-whisper is not installed")

    monkeypatch.setattr(perception, "_whisper", no_whisper)
    net.audio = "down"
    net.verdict = "SAFE"
    r = client.post(
        "/voice", data={"parent_id": "mom"}, files={"audio": ("a.wav", _wav(2), "audio/wav")}
    )
    assert r.status_code == 200
    _assert_couldnt_check(r.json())
    assert net.to(8081) == []


def test_image_over_limit_is_413(client, net):
    big = PNG + b"0" * hub_app.MAX_IMAGE_BYTES
    r = client.post(
        "/api/share", data={"parent_id": "dad"}, files={"image": ("s.png", big, "image/png")}
    )
    assert r.status_code == 413 and r.json()["error"] == "image_too_large"
    assert net.calls == []
