"""PRD §13 / NFR-1 canary test: message content never leaves the Mac except to local inference.

A fake message containing a canary string is checked end to end through the real FastAPI app
(real graph, rules, fusion, perception, explainer, alerts and tracing code) for /api/check,
/api/share (text and image), /voice and /share. Everything outside the process is faked:

- Sentry is initialised through `tracing.init_tracing` with a DSN-shaped fake value and an
  in-memory transport that records every serialized envelope;
- ntfy is on (fake topic) and every httpx client the hub creates uses one mock transport that
  answers for Ollama, llama-server (detector, with logprobs and red flags quoting the canary),
  the Gemma audio server and ntfy, and records each request (URL, headers, body);
- all loggers are captured at DEBUG, and so are stdout/stderr.

Asserted: the canary is in no Sentry payload, no ntfy request, no log record and no console
output; the only outbound hosts are OLLAMA_HOST, DETECTOR_URL, GEMMA_AUDIO_URL (all loopback)
and the ntfy server; nothing else opens a socket.
"""

import json
import logging
import shutil
import socket
import wave
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx
import pytest
import sentry_sdk
from fastapi.testclient import TestClient
from sentry_sdk.transport import Transport

from hub import app as hub_app
from hub import tracing
from hub.settings import get_settings
from tests.test_hub_env import hub_env, lp_entry  # noqa: F401

CANARY = "CANARY-7f3a9c1e-privacy-check"
SENTRY_DSN = "https://0123456789abcdef0123456789abcdef@o4500000000.ingest.sentry.io/4500000001"
NTFY_TOPIC = "rakshak-test-topic-not-real"
NTFY_SERVER = "https://ntfy.sh"
LOCAL_HOSTS = {"127.0.0.1", "localhost"}
SCAM_TEXT = f"Dear customer your account will be blocked today. Share your OTP now. Ref {CANARY}"
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32
# Explanation Gemma "returns": has the verdict word, no URL, no number outside the input.
EXPLANATION = "This is a scam. Do not share any code with anyone. Call Muaaz."


class _CaptureTransport(Transport):
    """Sentry transport that keeps every envelope (headers and items) as serialized text."""

    def __init__(self, sink: list[str], items: list[dict], options=None):
        super().__init__(options)
        self.sink = sink
        self.items = items

    def capture_envelope(self, envelope):
        self.sink.append(envelope.serialize().decode("utf-8", "replace"))
        self.items.extend(i.payload.json for i in envelope.items if i.payload.json is not None)


@dataclass
class Harness:
    client: TestClient
    caplog: pytest.LogCaptureFixture
    sentry: list[str] = field(default_factory=list)  # serialized envelopes
    sentry_items: list[dict] = field(default_factory=list)  # parsed events / transactions
    down: bool = False  # local inference servers refuse connections
    outbound: list[httpx.Request] = field(default_factory=list)
    sockets: list[str] = field(default_factory=list)

    def to_host(self, host: str) -> list[httpx.Request]:
        return [r for r in self.outbound if r.url.host == host]

    def by_port(self, port: int) -> list[httpx.Request]:
        return [r for r in self.outbound if r.url.port == port]


def _logprobs(verdict: str) -> dict:
    rest = '","category":"kyc_account_block","red_flags":[]}'
    return {
        "content": [
            lp_entry('{"verdict":"', {'{"verdict":"': 1.0}),
            lp_entry(verdict, {"SCAM": 0.97, "SUSPICIOUS": 0.02, "SAFE": 0.01}),
            lp_entry(rest, {rest: 1.0}),
        ]
    }


def _detector_reply() -> dict:
    content = {
        "verdict": "SCAM",
        "category": "kyc_account_block",
        # exact substrings of the input, so the grounding filter keeps them
        "red_flags": [
            {"quote": f"Ref {CANARY}", "reason": "urgency_deadline"},
            {"quote": "Share your OTP now", "reason": "asks_otp_or_pin"},
        ],
    }
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": json.dumps(content)},
                "logprobs": _logprobs("SCAM"),
            }
        ]
    }


def _handler(h: Harness, audio_port: int):
    def handle(request: httpx.Request) -> httpx.Response:
        h.outbound.append(request)
        host, port, path = request.url.host, request.url.port, request.url.path
        if host == "ntfy.sh":
            return httpx.Response(200, json={"id": "x"})
        if h.down and host in LOCAL_HOSTS:
            raise httpx.ConnectError(f"refused {request.url}")
        if host in LOCAL_HOSTS and port == 11434 and path == "/api/chat":
            body = json.loads(request.content)
            if body["messages"][0].get("images"):  # perception (Appendix A.2 output format)
                content = f"SENDER: AX-HDFCBK\nMESSAGE:\n{SCAM_TEXT}"
            else:  # explainer
                content = EXPLANATION
            return httpx.Response(200, json={"message": {"role": "assistant", "content": content}})
        if host in LOCAL_HOSTS and port == 8081 and path == "/v1/chat/completions":
            return httpx.Response(200, json=_detector_reply())
        if host in LOCAL_HOSTS and port == audio_port and path == "/v1/chat/completions":
            reply = {"choices": [{"message": {"content": SCAM_TEXT}}]}
            return httpx.Response(200, json=reply)
        return httpx.Response(599, json={"error": "unexpected destination"})

    return handle


@pytest.fixture
def harness(hub_env, monkeypatch, caplog, capsys):  # noqa: F811
    monkeypatch.setenv("RAKSHAK_CANARY", CANARY)
    monkeypatch.setenv("NTFY_TOPIC", NTFY_TOPIC)
    monkeypatch.setenv("NTFY_SERVER", NTFY_SERVER)
    monkeypatch.setenv("SENTRY_DSN", SENTRY_DSN)
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    monkeypatch.setenv("DETECTOR_URL", "http://127.0.0.1:8081/v1")
    monkeypatch.setenv("GEMMA_AUDIO_URL", "http://127.0.0.1:8082/v1")
    monkeypatch.setenv("OCR_FALLBACK", "gemma")
    monkeypatch.setenv("ASR_FALLBACK", "gemma")
    get_settings.cache_clear()
    assert get_settings().rakshak_canary == CANARY

    h = Harness(client=None, caplog=caplog)  # type: ignore[arg-type]
    transport = httpx.MockTransport(_handler(h, 8082))

    # Narrowest hook for outbound HTTP: every hub module creates `httpx.AsyncClient()` when no
    # client is passed in, so give that class a mock transport.
    real_client = httpx.AsyncClient

    class MockedClient(real_client):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", MockedClient)

    # Sentry: keep the real init (DSN, scrubbers, sampling) but swap in an in-memory transport.
    real_init = sentry_sdk.init

    def init_with_capture(*args, **kwargs):
        kwargs["transport"] = _CaptureTransport(h.sentry, h.sentry_items)
        return real_init(*args, **kwargs)

    monkeypatch.setattr(tracing.sentry_sdk, "init", init_with_capture)

    # Safety net: no code path may open a real network connection.
    def refuse(self, address, *a, **k):
        if isinstance(address, tuple):
            h.sockets.append(f"{address[0]}:{address[1]}")
            raise OSError("network disabled in privacy test")
        return None

    def refuse_lookup(host, *a, **k):
        h.sockets.append(f"dns:{host}")
        raise socket.gaierror("network disabled in privacy test")

    real_connect = socket.socket.connect
    monkeypatch.setattr(
        socket.socket,
        "connect",
        lambda self, address: refuse(self, address) or real_connect(self, address),
    )
    monkeypatch.setattr(socket, "getaddrinfo", refuse_lookup)

    caplog.set_level(logging.DEBUG)  # root logger: every library and hub logger
    for name in ("httpx", "httpcore", "sentry_sdk.errors", "hub"):
        logging.getLogger(name).setLevel(logging.DEBUG)

    with TestClient(hub_app.create_app(dist_dir=hub_env / "no-dist")) as c:
        h.client = c
        assert sentry_sdk.is_initialized()
        yield h

    client = sentry_sdk.get_client()
    client.close()
    sentry_sdk.get_global_scope().set_client(None)
    get_settings.cache_clear()


def _wav(seconds: float = 1.0) -> bytes:
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))
    return buf.getvalue()


def _variants(s: str) -> list[str]:
    """The canary as it might appear after casing or URL-quoting."""
    return [s, s.lower(), s.upper(), s.replace("-", "%2D"), s.replace(" ", "+")]


def _contains_canary(blob: str) -> bool:
    low = blob.lower()
    return any(v.lower() in low for v in _variants(CANARY))


def _netloc(r: httpx.Request) -> str:
    port = f":{r.url.port}" if r.url.port else ""
    return f"{r.url.host}{port}"


def _request_dump(r: httpx.Request) -> str:
    headers = " ".join(f"{k}={v}" for k, v in r.headers.items())
    return f"{r.method} {r.url} {headers} {r.content.decode('utf-8', 'replace')}"


def _log_dump(h: Harness) -> str:
    out = []
    for rec in h.caplog.records:
        out.append(rec.getMessage())
        out.append(repr(rec.args))
        out.append(h.caplog.handler.format(rec))
        if rec.exc_info:
            out.append(repr(rec.exc_info[1]))
    return "\n".join(out)


def assert_private(h: Harness, capsys, *, expect_alert: bool = True) -> None:
    ntfy = h.to_host("ntfy.sh")
    local = [r for r in h.outbound if r.url.host in LOCAL_HOSTS]

    # Sanity: the test really exercised the pipeline, so the absence checks mean something.
    assert any(CANARY in r.content.decode() for r in h.by_port(8081)), "detector never saw canary"
    assert any(r.url.path == "/api/chat" for r in h.by_port(11434)), "explainer never called"
    assert h.sentry, "Sentry transport received nothing"
    txns = [i for i in h.sentry_items if i.get("type") == "transaction"]
    assert txns, "no Sentry transaction captured"
    spans = {sp.get("description") or sp.get("name") for t in txns for sp in t["spans"]}
    assert {"normalize", "rules", "detect", "fuse", "explain", "notify_store"} <= spans, spans
    if expect_alert:
        assert len(ntfy) == 1, "SCAM verdict should push exactly one ntfy alert"

    # 1. Sentry: events, transactions, spans and breadcrumbs (whole serialized envelopes).
    for env in h.sentry:
        assert not _contains_canary(env), "canary in a Sentry envelope"
        assert "Share your OTP" not in env and "account will be blocked" not in env
        assert EXPLANATION not in env

    # 2. ntfy: URL, headers and body carry no content.
    for r in ntfy:
        assert not _contains_canary(_request_dump(r)), "canary in an ntfy request"
        body = r.content.decode()
        assert "Share your OTP" not in body and EXPLANATION not in body
        assert str(r.url) == f"{NTFY_SERVER}/{NTFY_TOPIC}"
        assert body.endswith("Call them.") and "likely SCAM" in body

    # 3. Logs (all loggers, DEBUG) and console output.
    logs = _log_dump(h)
    assert logs, "no log records captured"
    assert not _contains_canary(logs), "canary in a log record"
    assert "Share your OTP" not in logs and "account will be blocked" not in logs
    assert EXPLANATION not in logs
    console = capsys.readouterr()
    assert not _contains_canary(console.out + console.err), "canary on stdout/stderr"

    # 4. Outbound hosts: only the configured local inference servers and ntfy (+ Sentry, which
    #    went through the in-memory transport above and never reached httpx).
    s = get_settings()
    allowed = {
        urlsplit(s.ollama_host).netloc,
        urlsplit(s.detector_url).netloc,
        urlsplit(s.gemma_audio_url or "").netloc,
        urlsplit(s.ntfy_server).netloc,
    }
    hosts = {_netloc(r) for r in h.outbound}
    assert hosts <= allowed, f"unexpected outbound hosts: {hosts - allowed}"
    inference = (s.ollama_host, s.detector_url, s.gemma_audio_url)
    assert {urlsplit(u).hostname for u in inference} <= LOCAL_HOSTS
    assert local and all(r.url.scheme == "http" for r in local)
    assert not h.sockets, f"real network access attempted: {h.sockets}"


# --- the five entry points -----------------------------------------------------------------


def test_canary_api_check(harness, capsys):
    r = harness.client.post(
        "/api/check", json={"parent_id": "mom", "text": SCAM_TEXT, "channel": "sms"}
    )
    assert r.status_code == 200 and r.json()["verdict"] == "SCAM"
    assert_private(harness, capsys)


def test_canary_api_share_text(harness, capsys):
    r = harness.client.post("/api/share", data={"parent_id": "dad", "text": SCAM_TEXT})
    assert r.status_code == 200 and r.json()["verdict"] == "SCAM"
    assert_private(harness, capsys)


def test_canary_api_share_image(harness, capsys):
    r = harness.client.post(
        "/api/share", data={"parent_id": "dad"}, files={"image": ("s.png", PNG, "image/png")}
    )
    assert r.status_code == 200 and r.json()["verdict"] == "SCAM"
    # The screenshot went to the local Ollama only (never anywhere else).
    assert any(b"images" in r.content for r in harness.by_port(11434))
    assert_private(harness, capsys)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg needed to decode audio")
def test_canary_voice(harness, capsys):
    r = harness.client.post(
        "/voice",
        data={"parent_id": "dad"},
        files={"audio": ("note.wav", _wav(), "audio/wav")},
    )
    assert r.status_code == 200 and r.json()["verdict"] == "SCAM"
    assert harness.by_port(8082), "Gemma audio server was not used"
    assert_private(harness, capsys)


def test_canary_share_target(harness, capsys):
    r = harness.client.post(
        "/share?parent=mom",
        data={"title": "Bank", "text": SCAM_TEXT, "url": "http://x.example/a"},
        follow_redirects=False,
    )
    assert r.status_code == 303 and r.headers["location"].startswith("/v/evt_")
    assert_private(harness, capsys)


def test_canary_stays_out_of_logs_and_traces_when_inference_is_down(harness, capsys):
    """Failure paths log error types only: detector and Ollama refuse connections."""
    harness.down = True
    r = harness.client.post(
        "/api/check", json={"parent_id": "mom", "text": SCAM_TEXT, "channel": "sms"}
    )
    assert r.status_code == 200 and r.json()["verdict"] != "SAFE"
    assert harness.sentry, "Sentry transport received nothing"
    assert not _contains_canary(_log_dump(harness))
    for env in harness.sentry:
        assert not _contains_canary(env)
    for req in harness.to_host("ntfy.sh"):
        assert not _contains_canary(_request_dump(req))
    console = capsys.readouterr()
    assert not _contains_canary(console.out + console.err)
    assert not harness.sockets
