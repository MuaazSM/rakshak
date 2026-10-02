"""SQLite store (PRD §8.4, FR-18)."""

import sqlite3

from hub import db
from tests.test_hub_env import hub_env  # noqa: F401


def _verdict(event_id: str, verdict: str = "SCAM") -> dict:
    return {
        "event_id": event_id,
        "verdict": verdict,
        "category": "digital_arrest",
        "p_scam": 0.97,
        "red_flags": [{"quote": "arrest warrant", "reason": "threat_or_arrest", "source": "model"}],
        "explanation": "This is a scam.",
        "language": "en",
        "timings_ms": {"detect": 800, "total": 1500},
        "model_versions": {"detector": "d1", "gemma": "g1"},
    }


def test_init_creates_dirs_and_tables(hub_env):  # noqa: F811
    db.init_db()
    path = hub_env / "var" / "rakshak.db"
    assert path.is_file()
    names = {r[0] for r in sqlite3.connect(path).execute("SELECT name FROM sqlite_master")}
    assert {"events", "verdicts", "feedback"} <= names
    db.init_db()  # idempotent


def test_event_verdict_roundtrip(hub_env):  # noqa: F811
    db.init_db()
    db.save_event("evt_1", "mom", "sms", "hello text", "AX-BANK")
    db.save_verdict(_verdict("evt_1"))
    got = db.get_verdict("evt_1")
    assert got == {**_verdict("evt_1"), "parent_id": "mom"}
    assert db.get_verdict("evt_missing") is None
    assert db.event_exists("evt_1") and not db.event_exists("evt_2")


def test_feedback_and_pending(hub_env):  # noqa: F811
    db.init_db()
    for i in (1, 2, 3):
        db.save_event(
            f"evt_{i}", "dad", "sms", "t", None, created_at=f"2026-10-03T10:00:0{i}+00:00"
        )
        db.save_verdict(_verdict(f"evt_{i}"))
    db.save_feedback("evt_2", True)
    db.save_feedback("evt_3", False, "SAFE", "was my bank")
    pending = db.pending_feedback(10)
    assert [p["event_id"] for p in pending] == ["evt_1"]
    row = (
        sqlite3.connect(hub_env / "var" / "rakshak.db")
        .execute("SELECT correct, true_verdict, note FROM feedback WHERE event_id='evt_3'")
        .fetchone()
    )
    assert row == (0, "SAFE", "was my bank")


def test_list_events_since_is_metadata_only(hub_env):  # noqa: F811
    db.init_db()
    db.save_event(
        "evt_old", "mom", "sms", "SECRETTEXT", "S", created_at="2026-10-01T09:00:00+00:00"
    )
    db.save_event(
        "evt_new", "mom", "whatsapp", "SECRETTEXT", "S", created_at="2026-10-03T09:00:00+00:00"
    )
    db.save_verdict(_verdict("evt_new"))
    rows = db.list_events_since("2026-10-02")
    assert [r["event_id"] for r in rows] == ["evt_new"]
    assert rows[0]["verdict"] == "SCAM" and rows[0]["timings_ms"]["total"] == 1500
    blob = repr(rows)
    assert (
        "SECRETTEXT" not in blob and "This is a scam" not in blob and "arrest warrant" not in blob
    )


def test_recent_timings(hub_env):  # noqa: F811
    db.init_db()
    db.save_event("evt_1", "mom", "sms", "t", None)
    db.save_verdict(_verdict("evt_1"))
    assert db.recent_timings(5) == [{"detect": 800, "total": 1500}]
