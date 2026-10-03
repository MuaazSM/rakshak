"""Sentry setup: one transaction per check, one span per node, PII scrubbing (PRD FR-19, §9.3).

Nothing is sent unless SENTRY_DSN is set; without it every helper here is a no-op. Payloads
carry metadata only: `scrub` strips every content-bearing key at any depth, and `node_span`
accepts only the allow-listed attributes below.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import sentry_sdk

# PRD §9.3: keys removed recursively from every event, transaction, span and breadcrumb.
FORBIDDEN_KEYS = frozenset(
    {"text", "raw_text", "message", "explanation", "quote", "red_flags", "audio", "image"}
)

# PRD §9.3 span attributes (node name is the span name). Anything else is dropped.
ALLOWED_ATTRS = frozenset(
    {
        "model",
        "model_name",
        "model_version",
        "input_tokens",
        "output_tokens",
        "latency_ms",
        "verdict",
        "category",
        "p_scam",
        "rule_signals",  # signal names only
        "retries",
        "retry_count",
        "error_type",
    }
)
_MAX_STR = 80


# Dropped from stack frames: local variables can hold message text.
_FRAME_KEYS = frozenset({"vars"})
# Free-text fields that can echo input (e.g. a pydantic ValidationError includes the input):
# blanked, not removed, so the event shape stays valid.
_LOGENTRY_BLANK = ("formatted",)
_LOGENTRY_DROP = ("params",)


def scrub(event: Any, hint: Any = None) -> Any:
    """Sentry before_send / before_send_transaction / before_breadcrumb hook.

    Removes the PRD §9.3 keys at any depth, then blanks the string fields that carry
    formatted text: exception values, logentry formatted text, and (via the key list)
    message / breadcrumb message. Exception type/module and frame function/line stay.
    """
    out = _scrub(event)
    if isinstance(out, dict):
        _blank_text_fields(out)
    return out


def _blank_text_fields(event: dict) -> None:
    exc = event.get("exception")
    if isinstance(exc, dict):
        for item in exc.get("values") or []:
            if isinstance(item, dict) and "value" in item:
                item["value"] = ""
    log_entry = event.get("logentry")
    if isinstance(log_entry, dict):
        for k in _LOGENTRY_BLANK:
            if k in log_entry:
                log_entry[k] = ""
        for k in _LOGENTRY_DROP:
            log_entry.pop(k, None)


def _scrub(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: _scrub(v)
            for k, v in obj.items()
            if not (isinstance(k, str) and (k.lower() in FORBIDDEN_KEYS or k in _FRAME_KEYS))
        }
    if isinstance(obj, (list, tuple)):
        return [_scrub(v) for v in obj]
    return obj


def init_tracing(settings: Any) -> None:
    """Initialise Sentry iff `settings.sentry_dsn` is set."""
    if not settings.sentry_dsn:
        return
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        send_default_pii=False,
        traces_sample_rate=1.0,
        environment=settings.sentry_env,
        before_send=scrub,
        before_send_transaction=scrub,
        before_breadcrumb=scrub,
        include_local_variables=False,  # frame locals can hold message text
        max_request_body_size="never",
    )


def _safe_value(v: Any) -> Any:
    if isinstance(v, bool | int | float):
        return v
    if isinstance(v, str):
        return v[:_MAX_STR]
    if isinstance(v, (list, tuple)) and all(isinstance(i, str) for i in v):
        return [i[:_MAX_STR] for i in v]
    return None


def allowed_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Keep only PRD §9.3 attributes with plain scalar (or list-of-name) values."""
    out = {}
    for k, v in attrs.items():
        if k in ALLOWED_ATTRS and (safe := _safe_value(v)) is not None:
            out[k] = safe
    return out


@contextmanager
def check_transaction(parent_id: str, channel: str) -> Iterator[Any]:
    """One transaction per check. No-op without a DSN."""
    if not sentry_sdk.is_initialized():
        yield None
        return
    with sentry_sdk.start_transaction(op="check", name="check") as txn:
        txn.set_tag("parent_id", parent_id)
        txn.set_tag("channel", channel)
        yield txn


@contextmanager
def node_span(name: str, **attrs: Any) -> Iterator[Any]:
    """One span per graph node with only the allowed attributes. No-op without a DSN.

    The yielded span (None when disabled) has `set_data`, but prefer `set_attrs` below so
    late attributes (token counts, verdict) go through the same allowlist.
    """
    if not sentry_sdk.is_initialized():
        yield None
        return
    with sentry_sdk.start_span(op="node", name=name) as span:
        span.set_data("node", name)
        for k, v in allowed_attrs(attrs).items():
            span.set_data(k, v)
        yield span


def set_attrs(span: Any, **attrs: Any) -> None:
    """Add allow-listed attributes to a span from `node_span` (None-safe)."""
    if span is None:
        return
    for k, v in allowed_attrs(attrs).items():
        span.set_data(k, v)
