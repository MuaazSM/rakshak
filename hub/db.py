"""SQLite store for events, verdicts, feedback (PRD FR-18, §8.4).

`events.text` stays on the Mac; the DB file is git-ignored. Functions that feed the status
page (`list_events_since`) return metadata only, never message text or explanations.
Each call opens a short-lived connection to `settings.db_path`, so tests can point the
setting at a temp file.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path

from hub.settings import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY, parent_id TEXT, created_at TEXT, channel TEXT, text TEXT, sender TEXT);
CREATE TABLE IF NOT EXISTS verdicts (
    event_id TEXT PRIMARY KEY, verdict TEXT, category TEXT, p_scam REAL, red_flags TEXT,
    explanation TEXT, language TEXT, timings_ms TEXT, model_versions TEXT);
CREATE TABLE IF NOT EXISTS feedback (
    event_id TEXT, created_at TEXT, correct INTEGER, true_verdict TEXT, note TEXT);
CREATE INDEX IF NOT EXISTS idx_events_created ON events (created_at);
"""


def _path() -> Path:
    return Path(get_settings().db_path)


@contextmanager
def _connect():
    conn = sqlite3.connect(_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def init_db() -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as c:
        c.executescript(SCHEMA)


def save_event(
    event_id: str,
    parent_id: str,
    channel: str,
    text: str | None,
    sender: str | None,
    created_at: str | None = None,
) -> None:
    with _connect() as c:
        c.execute(
            "INSERT OR REPLACE INTO events (id, parent_id, created_at, channel, text, sender)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (event_id, parent_id, created_at or _now(), channel, text, sender),
        )


def save_verdict(verdict: dict) -> None:
    """`verdict` is a `Verdict.model_dump()`."""
    with _connect() as c:
        c.execute(
            "INSERT OR REPLACE INTO verdicts (event_id, verdict, category, p_scam, red_flags,"
            " explanation, language, timings_ms, model_versions) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                verdict["event_id"],
                verdict["verdict"],
                verdict.get("category"),
                verdict.get("p_scam"),
                json.dumps(verdict.get("red_flags", []), ensure_ascii=False),
                verdict["explanation"],
                verdict["language"],
                json.dumps(verdict.get("timings_ms", {})),
                json.dumps(verdict.get("model_versions", {})),
            ),
        )


def get_verdict(event_id: str) -> dict | None:
    """Verdict dict (§8.1 shape) or None if unknown."""
    with _connect() as c:
        row = c.execute(
            "SELECT v.*, e.parent_id FROM verdicts v JOIN events e ON e.id = v.event_id"
            " WHERE v.event_id = ?",
            (event_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "event_id": row["event_id"],
        "verdict": row["verdict"],
        "category": row["category"],
        "p_scam": row["p_scam"],
        "red_flags": json.loads(row["red_flags"] or "[]"),
        "explanation": row["explanation"],
        "language": row["language"],
        "parent_id": row["parent_id"],
        "timings_ms": json.loads(row["timings_ms"] or "{}"),
        "model_versions": json.loads(row["model_versions"] or "{}"),
    }


def event_exists(event_id: str) -> bool:
    with _connect() as c:
        return c.execute("SELECT 1 FROM events WHERE id = ?", (event_id,)).fetchone() is not None


def save_feedback(
    event_id: str, correct: bool, true_verdict: str | None = None, note: str | None = None
) -> None:
    with _connect() as c:
        c.execute(
            "INSERT INTO feedback (event_id, created_at, correct, true_verdict, note)"
            " VALUES (?, ?, ?, ?, ?)",
            (event_id, _now(), int(correct), true_verdict, note),
        )


def list_events_since(since: date | datetime | str) -> list[dict]:
    """Metadata for the status page, oldest first: id, parent_id, created_at, channel,
    verdict, category, p_scam, timings_ms. Never message text, quotes or explanations."""
    ts = since.isoformat() if hasattr(since, "isoformat") else str(since)
    with _connect() as c:
        rows = c.execute(
            "SELECT e.id, e.parent_id, e.created_at, e.channel, v.verdict, v.category,"
            " v.p_scam, v.timings_ms FROM events e LEFT JOIN verdicts v ON v.event_id = e.id"
            " WHERE e.created_at >= ? ORDER BY e.created_at, e.id",
            (ts,),
        ).fetchall()
    return [
        {
            "event_id": r["id"],
            "parent_id": r["parent_id"],
            "created_at": r["created_at"],
            "channel": r["channel"],
            "verdict": r["verdict"],
            "category": r["category"],
            "p_scam": r["p_scam"],
            "timings_ms": json.loads(r["timings_ms"] or "{}"),
        }
        for r in rows
    ]


def recent_timings(limit: int = 50) -> list[dict]:
    """Timings of the most recent verdicts (for /health p95)."""
    with _connect() as c:
        rows = c.execute(
            "SELECT v.timings_ms FROM verdicts v JOIN events e ON e.id = v.event_id"
            " ORDER BY e.created_at DESC, e.id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [json.loads(r["timings_ms"] or "{}") for r in rows]


def pending_feedback(limit: int = 20) -> list[dict]:
    """Recent verdict events that have no feedback yet (metadata only), newest first."""
    with _connect() as c:
        rows = c.execute(
            "SELECT e.id, e.parent_id, e.created_at, e.channel, v.verdict, v.category"
            " FROM events e JOIN verdicts v ON v.event_id = e.id"
            " WHERE NOT EXISTS (SELECT 1 FROM feedback f WHERE f.event_id = e.id)"
            " ORDER BY e.created_at DESC, e.id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {
            "event_id": r["id"],
            "parent_id": r["parent_id"],
            "created_at": r["created_at"],
            "channel": r["channel"],
            "verdict": r["verdict"],
            "category": r["category"],
        }
        for r in rows
    ]
