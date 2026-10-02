"""Hub status page, GET /status (PRD §2, §5.3, §9.3, §13, §18; design frame D2).

Metadata only: times, parent names, verdict words and categories, counts and latencies.
Message text, explanations and red-flag quotes are never read here (`db.list_events_since`
and `db.pending_feedback` do not return them) and everything rendered is HTML-escaped.
The hub is reached over the tailnet only, so there is no auth; responses are `no-store`.
"""

import asyncio
import platform
import shutil
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from string import Template

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from hub import db, fusion
from hub.settings import Parent, get_parents, get_settings

router = APIRouter()

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
TARGET_TEXT_S = 6.0  # NFR-2: text check p95
TARGET_MEDIA_S = 12.0  # NFR-2: screenshot / voice check p95
TIMING_WINDOW = 50  # same window as /health

# verdict -> (css class, word, icon). PRD §18.3: raised hand, warning, check, phone.
_VERDICTS = {
    "SCAM": ("scam", "SCAM", "hand"),
    "SUSPICIOUS": ("careful", "CAREFUL", "alert"),
    "SAFE": ("normal", "NORMAL", "check"),
    "UNKNOWN": ("unknown", "COULDN’T", "phone"),
}
_ORDER = ("SCAM", "SUSPICIOUS", "SAFE", "UNKNOWN")

_ICONS = {  # Lucide outline paths (2px stroke), as in the design export's icons/
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "hand": (
        '<path d="M18 11V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2"/>'
        '<path d="M14 10V4a2 2 0 0 0-2-2a2 2 0 0 0-2 2v2"/>'
        '<path d="M10 10.5V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2v8"/>'
        '<path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6'
        'a2 2 0 0 1 2.83-2.82L7 15"/>'
    ),
    "alert": (
        '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/>'
        '<path d="M12 9v4"/><path d="M12 17h.01"/>'
    ),
    "phone": (
        '<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6'
        " 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7"
        " 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45"
        ' 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/>'
    ),
}


def _icon(name: str, size: int) -> str:
    return (
        f'<svg class="ic" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none"'
        ' stroke="currentColor" stroke-width="2" stroke-linecap="round"'
        f' stroke-linejoin="round" aria-hidden="true">{_ICONS[name]}</svg>'
    )


def _vinfo(verdict: str | None) -> tuple[str, str, str]:
    return _VERDICTS.get(verdict or "UNKNOWN", _VERDICTS["UNKNOWN"])


# --- small pure helpers -------------------------------------------------------------


def _percentile(values: list[int], q: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _secs(ms: int | None) -> str:
    return "—" if ms is None else f"{ms / 1000:.2f} s"


def _local(created_at: str | None) -> datetime | None:
    try:
        dt = datetime.fromisoformat(created_at or "")
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone()


def _clock(created_at: str | None, today: datetime | None = None) -> str:
    dt = _local(created_at)
    if dt is None:
        return "—"
    if today is not None and dt.date() != today.date():
        return dt.strftime("%b %-d %H:%M")
    return dt.strftime("%H:%M")


def _start_of_today(now: datetime) -> str:
    """Local midnight as a UTC ISO string comparable with `events.created_at`."""
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight.astimezone(UTC).isoformat(timespec="seconds")


def _node_ms(timings: list[dict], *nodes: str) -> list[int]:
    """Latencies of the given nodes; 0 means the node was skipped for that check."""
    return [v for t in timings for n in nodes if (v := t.get(n)) and v > 0]


def split_totals(timings: list[dict]) -> tuple[list[int], list[int]]:
    """(text totals, image/voice totals): a check that ran `perceive` was image or voice."""
    text = [t["total"] for t in timings if "total" in t and not t.get("perceive")]
    media = [t["total"] for t in timings if "total" in t and t.get("perceive")]
    return text, media


# --- HTML fragments (everything escaped) ----------------------------------------------


def _p95_block(label: str, values: list[int], target: float) -> str:
    p95 = _percentile(values, 0.95)
    num = "—" if p95 is None else f"{p95 / 1000:.1f} s"
    return (
        f'<div class="stat"><span>{escape(label)}</span><span class="stat-num">{escape(num)}'
        f' <span class="stat-of">/ {target:.1f}</span></span></div>'
    )


def _status_cell(up: bool | None) -> str:
    if up is None:
        return '<span class="st st-unknown">not set up</span>'
    if up:
        return f'<span class="st st-normal">{_icon("check", 14)}ready</span>'
    return f'<span class="st st-careful">{_icon("alert", 14)}not responding</span>'


def _model_row(
    title: str, ident: str, role: str, runtime: str, up: bool | None, ms: list[int]
) -> str:
    return (
        '<div class="row models-row">'
        f'<div class="cell name"><b>{escape(title)}</b><span class="muted">{escape(ident)}</span>'
        "</div>"
        f'<span class="cell">{escape(role)}</span><span class="cell">{escape(runtime)}</span>'
        f'<span class="cell">{_status_cell(up)}</span>'
        f'<span class="cell num">{_secs(_percentile(ms, 0.5))}</span>'
        f'<span class="cell num">{_secs(_percentile(ms, 0.95))}</span>'
        "</div>"
    )


def _kv(key: str, value: str, cls: str = "") -> str:
    return (
        f'<div class="kv"><span class="muted">{escape(key)}</span>'
        f'<span class="{cls}">{escape(value)}</span></div>'
    )


def _verdict_word(verdict: str | None) -> str:
    cls, word, icon = _vinfo(verdict)
    return f'<span class="vw v-{cls}">{_icon(icon, 14)}{escape(word)}</span>'


def _parent_card(parent: Parent, events: list[dict], today: datetime) -> str:
    counts = {v: 0 for v in _ORDER}
    for e in events:
        counts[e["verdict"] if e["verdict"] in counts else "UNKNOWN"] += 1
    tiles = "".join(
        f'<div class="tile v-{_vinfo(v)[0]}"><span class="tile-n">{counts[v]}</span>'
        f'<span class="tile-l">{escape(_vinfo(v)[1])}</span></div>'
        for v in _ORDER
    )
    rows = (
        "".join(
            f'<div class="row ev"><span class="muted">{_clock(e["created_at"])}</span>'
            f"{_verdict_word(e['verdict'])}"
            f"<span>{escape(e['category'] or chr(0x2014))}</span></div>"
            for e in reversed(events)
        )
        or '<div class="empty">No checks yet today.</div>'
    )
    device = "ios shortcut" if parent.device == "ios" else parent.device
    return (
        f'<section class="card parent" data-parent="{escape(parent.id)}">'
        f'<h2>{escape(parent.name)} · today<span class="head-note">{escape(device)}</span>'
        f'</h2><div class="tiles">{tiles}</div><div class="list mono">{rows}</div></section>'
    )


def _feedback_row(item: dict, today: datetime) -> str:
    parent = get_parents().get(item["parent_id"])
    who = parent.id if parent else item["parent_id"] or "?"
    eid = escape(item["event_id"], quote=True)
    return (
        f'<div class="row fb-row" data-event="{eid}">'
        f'<span class="muted">{escape(_clock(item["created_at"], today))}</span>'
        f"<span>{escape(who)}</span>{_verdict_word(item['verdict'])}"
        f'<span class="fb-cat">{escape(item["category"] or chr(0x2014))}</span>'
        '<span class="fb-btns">'
        f'<button type="button" class="pill yes" data-event="{eid}" data-correct="true">Yes</button>'
        f'<button type="button" class="pill" data-event="{eid}" data-correct="false">No</button>'
        "</span></div>"
    )


# --- page assembly ------------------------------------------------------------------


async def _ping_all() -> tuple[bool | None, bool | None, bool | None]:
    """Same probes as /health (detector, Ollama, Gemma audio), in-process."""
    from hub.app import _ping  # lazy: hub.app imports this module while it is being built

    s = get_settings()
    base = s.detector_url.rstrip("/").removesuffix("/v1")
    audio = s.gemma_audio_url.rstrip("/").removesuffix("/v1") if s.gemma_audio_url else None
    async with httpx.AsyncClient(timeout=2.0) as c:
        det, oll, aud = await asyncio.gather(
            _ping(c, f"{base}/health"),
            _ping(c, f"{s.ollama_host.rstrip('/')}/api/tags"),
            _ping(c, f"{audio}/health" if audio else None),
        )
    return det, oll, aud


def _disk_free_gb() -> int | None:
    path = Path(get_settings().db_path).resolve()
    for candidate in (path, *path.parents):
        if candidate.exists():
            try:
                return int(shutil.disk_usage(candidate).free / 1e9)
            except OSError:
                return None
    return None


def _port(url: str) -> str:
    return url.rstrip("/").rsplit(":", 1)[-1].split("/")[0]


def render_status(
    *,
    now: datetime,
    hostname: str,
    up: tuple[bool | None, bool | None, bool | None],
    timings: list[dict],
    events: list[dict],
    pending: list[dict],
) -> str:
    """Build the page from plain data (no I/O) so it can be tested directly."""
    s = get_settings()
    th = fusion.get_thresholds()
    det, oll, aud = up
    degraded = not (det and oll and aud is not False)
    text_totals, media_totals = split_totals(timings)

    models = [
        _model_row(
            "Gemma 4 E2B",
            s.gemma_model,
            "perceive · explain",
            f"ollama :{_port(s.ollama_host)}",
            oll,
            _node_ms(timings, "perceive", "explain"),
        ),
        _model_row(
            "Rakshak detector",
            s.detector_version,
            "detect",
            f"llama-server :{_port(s.detector_url)}",
            det,
            _node_ms(timings, "detect"),
        ),
    ]
    if s.gemma_audio_url:
        models.append(
            _model_row(
                "Gemma 4 E2B audio",
                s.gemma_model,
                "listen",
                f"llama-server :{_port(s.gemma_audio_url)}",
                aud,
                [],
            )
        )

    by_parent: dict[str, list[dict]] = {}
    for e in events:
        by_parent.setdefault(e["parent_id"], []).append(e)
    parents = list(get_parents().values())
    cards = "".join(_parent_card(p, by_parent.get(p.id, []), now) for p in parents)
    scams_today = sum(1 for e in events if e["verdict"] == "SCAM")

    outbound = "".join(
        [
            _kv("ntfy alerts", "on" if s.ntfy_topic else "off"),
            _kv("sentry tracing", "on" if s.sentry_dsn else "off"),
            _kv("scam alerts today", str(scams_today)),
            _kv(
                "thresholds",
                f"T_HIGH {th.t_high:.2f} · T_LOW {th.t_low:.2f}"
                + ("" if th.calibrated else " (defaults)"),
            ),
        ]
    )
    fb_rows = "".join(_feedback_row(p, now) for p in pending) or (
        '<div class="empty" id="fb-empty">Nothing waiting. Every check has an answer.</div>'
    )
    disk = _disk_free_gb()
    system = "macOS" if platform.system() == "Darwin" else platform.system()
    machine = " · ".join(
        [f"{system} {platform.machine()}"]
        + ([f"disk {disk} GB free"] if disk is not None else [])
        + [f"{len(events)} checks today"]
    )
    if degraded:
        banner = ("careful", "alert", "Home computer is online, a model isn’t ready")
    else:
        banner = ("normal", "check", "Home computer is online")

    return Template((TEMPLATE_DIR / "status.html").read_text("utf-8")).substitute(
        mark_svg=(TEMPLATE_DIR / "status_mark.svg").read_text("utf-8"),
        hostname=escape(hostname),
        refreshed=now.strftime("%H:%M:%S"),
        banner_class=banner[0],
        banner_icon=_icon(banner[1], 22),
        banner_title=escape(banner[2]),
        machine_line=escape(machine),
        p95_text=_p95_block("text check p95", text_totals, TARGET_TEXT_S),
        p95_media=_p95_block("image · voice p95", media_totals, TARGET_MEDIA_S),
        models_rows="".join(models),
        outbound_rows=outbound,
        parent_cards=cards,
        parent_count=str(max(len(parents), 1)),
        feedback_count=str(len(pending)),
        feedback_rows=fb_rows,
    )


@router.get("/status", response_class=HTMLResponse, include_in_schema=False)
async def status_page(request: Request) -> HTMLResponse:
    now = datetime.now().astimezone()
    await asyncio.to_thread(db.init_db)  # idempotent; the page works before the first check
    up, timings, events, pending = await asyncio.gather(
        _ping_all(),
        asyncio.to_thread(db.recent_timings, TIMING_WINDOW),
        asyncio.to_thread(db.list_events_since, _start_of_today(now)),
        asyncio.to_thread(db.pending_feedback, 20),
    )
    html = render_status(
        now=now,
        hostname=request.url.hostname or "localhost",
        up=up,
        timings=timings,
        events=events,
        pending=pending,
    )
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})
