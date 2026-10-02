"""PRD FR-17, §9.2: ntfy body is exact, carries no message text, and is a no-op without a topic."""

import asyncio
from datetime import datetime

import httpx

from hub import alerts
from hub.settings import Parent, Settings

MOM = Parent(id="mom", name="Asha", age=60, language="en", device="android")
WHEN = datetime(2026, 10, 3, 14, 5)


def _use(monkeypatch, **kw):
    s = Settings(_env_file=None, **kw)
    monkeypatch.setattr(alerts, "get_settings", lambda: s)


def _client(seen: list, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_body_and_title_exact(monkeypatch):
    _use(monkeypatch, ntfy_server="https://ntfy.example", ntfy_topic="rakshak-test-topic")
    seen: list[httpx.Request] = []
    ok = asyncio.run(alerts.notify_scam(MOM, "kyc_account_block", WHEN, client=_client(seen)))
    assert ok is True
    (req,) = seen
    assert req.method == "POST"
    assert str(req.url) == "https://ntfy.example/rakshak-test-topic"
    assert req.headers["Title"] == "Rakshak alert"
    assert req.content.decode() == "Asha got a likely SCAM (kyc_account_block) at 14:05. Call them."


def test_no_message_content_in_request(monkeypatch):
    _use(monkeypatch, ntfy_topic="t")
    seen: list[httpx.Request] = []
    asyncio.run(alerts.notify_scam(MOM, "other_scam", WHEN, client=_client(seen)))
    assert seen[0].content.decode() == alerts.alert_body(MOM, "other_scam", WHEN)


def test_noop_without_topic(monkeypatch):
    _use(monkeypatch, ntfy_topic=None)
    seen: list[httpx.Request] = []
    assert asyncio.run(alerts.notify_scam(MOM, "other_scam", WHEN, client=_client(seen))) is False
    assert seen == []


def test_errors_are_swallowed(monkeypatch):
    _use(monkeypatch, ntfy_topic="t")
    assert asyncio.run(alerts.notify_scam(MOM, "x", WHEN, client=_client([], status=500))) is False

    def boom(request):
        raise httpx.ConnectError("down")

    c = httpx.AsyncClient(transport=httpx.MockTransport(boom))
    assert asyncio.run(alerts.notify_scam(MOM, "x", WHEN, client=c)) is False
