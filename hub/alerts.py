"""ntfy alerts to the son on SCAM, metadata only (PRD FR-17, §9.2).

The payload is exactly `"{parent.name} got a likely SCAM ({category}) at {HH:MM}. Call them."`
under the title "Rakshak alert". No message content, ever. Errors are swallowed.
"""

import logging
from datetime import datetime

import httpx

from hub.settings import Parent, get_settings

log = logging.getLogger(__name__)

TITLE = "Rakshak alert"
TIMEOUT_S = 10.0


def alert_body(parent: Parent, category: str | None, when: datetime) -> str:
    return (
        f"{parent.name} got a likely SCAM ({category or 'other_scam'}) at {when:%H:%M}. Call them."
    )


async def notify_scam(
    parent: Parent,
    category: str | None,
    when: datetime,
    *,
    client: httpx.AsyncClient | None = None,
) -> bool:
    """Publish to ntfy. False when NTFY_TOPIC is unset or the push failed; never raises."""
    s = get_settings()
    topic = (s.ntfy_topic or "").strip().strip("/")
    if not topic:
        return False
    own = client is None
    http = client or httpx.AsyncClient()
    try:
        r = await http.post(
            f"{s.ntfy_server.rstrip('/')}/{topic}",
            content=alert_body(parent, category, when).encode("utf-8"),
            headers={"Title": TITLE, "Priority": "high"},
            timeout=TIMEOUT_S,
        )
        r.raise_for_status()
        return True
    except Exception as e:  # the check result matters more than the push
        log.warning("ntfy alert failed (%s)", type(e).__name__)  # never log the topic
        return False
    finally:
        if own:
            await http.aclose()
