"""PRD FR-19, §9.3: scrubber removes forbidden keys at any depth; spans carry allowed attrs only."""

import pytest
import sentry_sdk
from sentry_sdk.transport import Transport

from hub import tracing
from hub.settings import Settings

FORBIDDEN = sorted(tracing.FORBIDDEN_KEYS)
CANARY = "CANARY-xyz-message-body"


def test_forbidden_keys_match_prd():
    assert {
        "text",
        "raw_text",
        "message",
        "explanation",
        "quote",
        "red_flags",
        "audio",
        "image",
    } == tracing.FORBIDDEN_KEYS


@pytest.mark.parametrize("key", FORBIDDEN)
def test_scrub_removes_key_at_any_depth(key):
    event = {
        key: CANARY,
        "keep": 1,
        "extra": {key: CANARY, "deep": {"list": [{key: CANARY, "ok": "x"}, [{key: CANARY}]]}},
        "breadcrumbs": {"values": [{"category": "log", key: CANARY}]},
        "request": {"data": {key: CANARY}},
        "spans": [{"data": {key: CANARY, "node": "detect"}}],
        "contexts": {"c": ({key: CANARY},)},
    }
    out = tracing.scrub(event, {})
    assert CANARY not in repr(out)
    assert out["keep"] == 1
    assert out["spans"][0]["data"] == {"node": "detect"}
    assert out["extra"]["deep"]["list"][0] == {"ok": "x"}
    assert event["keep"] == 1 and key in event  # input is not mutated


def test_scrub_keeps_other_keys_and_non_dicts():
    assert tracing.scrub({"a": [1, "two", None, {"b": 3.5}]}) == {"a": [1, "two", None, {"b": 3.5}]}


def test_allowed_attrs_drops_everything_else():
    got = tracing.allowed_attrs(
        {
            "model": "gemma4:e2b",
            "model_version": "v1",
            "input_tokens": 10,
            "output_tokens": 5,
            "latency_ms": 12,
            "verdict": "SCAM",
            "category": "other_scam",
            "p_scam": 0.97,
            "rule_signals": ["lookalike_domain"],
            "retries": 1,
            "error_type": "HTTPError",
            "text": CANARY,
            "quote": CANARY,
            "explanation": CANARY,
            "anything_else": "x",
            "bad_type": {"text": CANARY},
        }
    )
    assert set(got) == {
        "model",
        "model_version",
        "input_tokens",
        "output_tokens",
        "latency_ms",
        "verdict",
        "category",
        "p_scam",
        "rule_signals",
        "retries",
        "error_type",
    }


def test_no_ops_without_dsn():
    assert not sentry_sdk.is_initialized()
    tracing.init_tracing(Settings(_env_file=None, sentry_dsn=None))
    assert not sentry_sdk.is_initialized()
    with (
        tracing.check_transaction("mom", "sms"),
        tracing.node_span("normalize", text=CANARY, verdict="SAFE") as span,
    ):
        tracing.set_attrs(span, latency_ms=3)


class _Capture(Transport):
    def __init__(self, sink):
        super().__init__()
        self.sink = sink

    def capture_envelope(self, envelope):
        for item in envelope.items:
            if item.payload.json is not None:
                self.sink.append(item.payload.json)


def test_end_to_end_payload_is_clean():
    events = []
    sentry_sdk.init(
        dsn="http://public@localhost:9/1",
        transport=_Capture(events),
        send_default_pii=False,
        traces_sample_rate=1.0,
        before_send=tracing.scrub,
        before_send_transaction=tracing.scrub,
        include_local_variables=False,
    )
    try:
        with tracing.check_transaction("mom", "sms"):
            with tracing.node_span("detect", verdict="SCAM", text=CANARY, quote=CANARY) as span:
                tracing.set_attrs(span, p_scam=0.9, explanation=CANARY, image=CANARY)
            sentry_sdk.capture_message(CANARY)  # sent as "message": removed by the scrubber
    finally:
        sentry_sdk.get_client().close()
        sentry_sdk.get_global_scope().set_client(None)
    assert events, "transport saw no events"
    blob = repr(events)
    assert CANARY not in blob
    txn = next(e for e in events if e.get("type") == "transaction")
    data = txn["spans"][0]["data"]
    assert data["verdict"] == "SCAM" and data["p_scam"] == 0.9 and data["node"] == "detect"
    assert "text" not in data and "quote" not in data and "explanation" not in data


def test_scrub_blanks_exception_and_logentry_text():
    event = {
        "exception": {
            "values": [
                {
                    "type": "ValidationError",
                    "module": "pydantic",
                    "value": f"input_value='{CANARY}'",
                    "stacktrace": {
                        "frames": [
                            {"function": "node", "lineno": 7, "vars": {"x": CANARY}},
                        ]
                    },
                }
            ]
        },
        "logentry": {"message": "bad %s", "formatted": f"bad {CANARY}", "params": [CANARY]},
        "breadcrumbs": {"values": [{"type": "log", "message": CANARY, "category": "x"}]},
    }
    out = tracing.scrub(event)
    assert CANARY not in repr(out)
    (exc,) = out["exception"]["values"]
    assert exc["type"] == "ValidationError" and exc["module"] == "pydantic" and exc["value"] == ""
    assert exc["stacktrace"]["frames"] == [{"function": "node", "lineno": 7}]
    assert out["logentry"] == {"formatted": ""}
    assert out["breadcrumbs"]["values"] == [{"type": "log", "category": "x"}]


def test_exception_and_log_text_never_reach_the_transport():
    import logging

    from hub.schemas import DetectorOutput

    events = []
    sentry_sdk.init(
        dsn="http://public@localhost:9/1",
        transport=_Capture(events),
        traces_sample_rate=1.0,
        before_send=tracing.scrub,
        before_send_transaction=tracing.scrub,
        before_breadcrumb=tracing.scrub,
        include_local_variables=False,
    )
    log = logging.getLogger("canary-test")
    try:
        with pytest.raises(ValueError), tracing.check_transaction("mom", "sms"):
            log.warning("checking %s", CANARY)  # breadcrumb
            with tracing.node_span("detect"):
                try:
                    DetectorOutput.model_validate({"verdict": CANARY})
                except ValueError as e:  # pydantic's message echoes the input
                    assert CANARY in str(e)
                    sentry_sdk.capture_exception(e)
                    log.error("detector failed for %s", CANARY)  # logentry event
                    raise
    finally:
        sentry_sdk.get_client().close()
        sentry_sdk.get_global_scope().set_client(None)
    assert events
    assert CANARY not in repr(events)
    assert any("exception" in e for e in events)
