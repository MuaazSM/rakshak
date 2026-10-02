"""GET /status: metadata-only hub page (PRD §13, §18; frame D2). No live services."""

import re
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from hub import db
from hub.app import create_app
from tests.test_hub_env import hub_env  # noqa: F401

MARKER = "ZQX-UNIQUE-MESSAGE-MARKER"
EXPLAIN = "ZQX-UNIQUE-EXPLANATION-MARKER"
QUOTE = "ZQX-UNIQUE-QUOTE-MARKER"


@pytest.fixture
def client(hub_env, monkeypatch, tmp_path):  # noqa: F811
    async def fake_ping(_client, url):
        return None if not url else True

    monkeypatch.setattr("hub.app._ping", fake_ping)
    with TestClient(create_app(dist_dir=tmp_path / "no-dist")) as c:
        yield c


def seed(
    event_id: str,
    parent: str,
    verdict: str,
    category: str | None,
    *,
    at: datetime | None = None,
    timings: dict | None = None,
) -> None:
    when = (at or datetime.now(UTC)).astimezone(UTC).isoformat(timespec="seconds")
    db.save_event(event_id, parent, "sms", f"{MARKER} {event_id}", "AX-SENDER", created_at=when)
    db.save_verdict(
        {
            "event_id": event_id,
            "verdict": verdict,
            "category": category,
            "p_scam": 0.9,
            "red_flags": [{"quote": QUOTE, "reason": "urgency_deadline", "source": "model"}],
            "explanation": f"{EXPLAIN} {event_id}",
            "language": "en",
            "timings_ms": timings or {"detect": 800, "total": 1500},
            "model_versions": {},
        }
    )


def parent_section(html: str, parent_id: str) -> str:
    m = re.search(
        rf'<section class="card parent" data-parent="{parent_id}">(.*?)</section>', html, re.S
    )
    assert m, f"no card for {parent_id}"
    return m.group(1)


def tile_counts(section: str) -> list[int]:
    return [int(n) for n in re.findall(r'<span class="tile-n">(\d+)</span>', section)]


def test_empty_db_renders(client):
    r = client.get("/status")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert r.headers["cache-control"] == "no-store"
    assert "Home computer is online" in r.text
    assert "Mom \u00b7 today" in r.text and "Dad \u00b7 today" in r.text
    assert "Nothing waiting" in r.text
    assert tile_counts(parent_section(r.text, "mom")) == [0, 0, 0, 0]


def test_no_message_text_explanation_or_quotes(client):
    seed("evt_1", "mom", "SCAM", "kyc_account_block")
    seed("evt_2", "dad", "SAFE", "genuine_otp")
    seed("evt_3", "mom", "SUSPICIOUS", None)
    html = client.get("/status").text
    for secret in (MARKER, EXPLAIN, QUOTE, "AX-SENDER", "urgency_deadline"):
        assert secret not in html
    assert "kyc_account_block" in html  # category is metadata and is allowed


def test_counts_per_parent_and_times(client):
    seed("evt_a", "mom", "SCAM", "kyc_account_block")
    seed("evt_b", "mom", "SCAM", "lottery_prize")
    seed("evt_c", "mom", "SAFE", "personal")
    seed("evt_d", "dad", "UNKNOWN", None)
    seed("evt_e", "dad", "SUSPICIOUS", None)
    seed("evt_old", "mom", "SCAM", "other_scam", at=datetime.now(UTC) - timedelta(days=2))
    html = client.get("/status").text
    mom, dad = parent_section(html, "mom"), parent_section(html, "dad")
    assert tile_counts(mom) == [2, 0, 1, 0]  # SCAM, CAREFUL, NORMAL, COULDN'T
    assert tile_counts(dad) == [0, 1, 0, 1]
    assert "other_scam" not in mom  # yesterday's check is not "today"
    assert len(re.findall(r'class="row ev"', mom)) == 3
    assert re.search(r"\d\d:\d\d", mom)


def test_feedback_queue_buttons_and_post(client):
    seed("evt_f1", "mom", "SCAM", "kyc_account_block")
    seed("evt_f2", "dad", "SAFE", "genuine_otp")
    html = client.get("/status").text
    for eid in ("evt_f1", "evt_f2"):
        assert f'data-event="{eid}" data-correct="true"' in html
        assert f'data-event="{eid}" data-correct="false"' in html
    assert 'id="fb-n">2<' in html
    assert "/api/feedback/" in html  # inline script posts here

    assert client.post("/api/feedback/evt_f1", json={"correct": True}).status_code == 204
    html = client.get("/status").text
    assert 'data-event="evt_f1"' not in html
    assert 'data-event="evt_f2"' in html
    assert 'id="fb-n">1<' in html


def test_everything_is_escaped(client):
    seed("evt_x", "mom", "SCAM", "<img src=x onerror=alert(1)>")
    seed("evt_y", "ghost", "SAFE", "<script>alert(2)</script>")
    db.save_event("evt_z", '"><u>zz', "sms", "t", None)
    db.save_verdict(
        {
            "event_id": "evt_z",
            "verdict": "SAFE",
            "category": None,
            "red_flags": [],
            "explanation": "e",
            "language": "en",
        }
    )
    html = client.get("/status").text
    assert "<img src=x" not in html and "<script>alert(2)" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert '"><u>zz' not in html


def test_degraded_banner_and_latency(client, monkeypatch):
    seed("evt_t", "mom", "SAFE", "personal", timings={"detect": 900, "total": 2000})
    seed(
        "evt_m", "mom", "SAFE", "personal",
        timings={"perceive": 3000, "detect": 1000, "explain": 1200, "total": 7000},
    )  # fmt: skip

    async def down(_client, url):
        return False if url else None

    monkeypatch.setattr("hub.app._ping", down)
    html = client.get("/status").text
    assert "a model isn\u2019t ready" in html
    assert "not responding" in html
    assert "2.0 s" in html and "7.0 s" in html  # text p95, image/voice p95
    assert "/ 6.0" in html and "/ 12.0" in html
