"""Detector client: parsing, constrained retry, None/unavailable, grounding, p_scam (FR-13/14)."""

import json
import math

import httpx
import pytest

from hub import detector
from tests.test_hub_env import anyio_backend, hub_env, lp_entry  # noqa: F401

pytestmark = pytest.mark.anyio

TEXT = "Your account will be blocked today. Share OTP now."


def _content(verdict="SCAM", flags=None, category="kyc_account_block") -> str:
    flags = (
        flags if flags is not None else [{"quote": "will be blocked", "reason": "urgency_deadline"}]
    )
    return json.dumps(
        {"verdict": verdict, "category": category, "red_flags": flags}, separators=(",", ":")
    )


def _completion(content: str, logprobs: list[dict] | None = None) -> dict:
    choice: dict = {"message": {"role": "assistant", "content": content}}
    if logprobs is not None:
        choice["logprobs"] = {"content": logprobs}
    return {"choices": [choice]}


def _scam_logprobs(p_scam=0.9) -> list[dict]:
    rest = (1 - p_scam) / 2
    return [
        lp_entry('{"verdict":"', {'{"verdict":"': 1.0}),
        lp_entry("SCAM", {"SCAM": p_scam, "SAFE": rest, "SUSPICIOUS": rest}),
        lp_entry('"', {'"': 1.0}),
    ]


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# --- p_scam_from_logprobs ---------------------------------------------------


def test_p_scam_single_token_labels():
    p = detector.p_scam_from_logprobs(_scam_logprobs(0.9))
    assert p == pytest.approx(0.9)


def test_label_probs_sum_to_one_and_match_p_scam():
    lp = [
        lp_entry('{"verdict":"', {'{"verdict":"': 1.0}),
        lp_entry("SAFE", {"SAFE": 0.6, "SCAM": 0.1, "SUSPICIOUS": 0.1}),
    ]
    probs = detector.label_probs_from_logprobs(lp)
    assert probs == pytest.approx({"SCAM": 0.125, "SUSPICIOUS": 0.125, "SAFE": 0.75})
    assert detector.p_scam_from_logprobs(lp) == pytest.approx(probs["SCAM"])
    assert detector.label_probs_from_logprobs(None) is None


def test_p_scam_single_token_safe_sampled():
    lp = [
        lp_entry('{"verdict":"', {'{"verdict":"': 1.0}),
        lp_entry("SAFE", {"SAFE": 0.7, "SCAM": 0.2, "SUSPICIOUS": 0.1}),
    ]
    assert detector.p_scam_from_logprobs(lp) == pytest.approx(0.2)


def test_p_scam_renormalizes_over_labels():
    lp = [
        lp_entry('{"verdict":"', {'{"verdict":"': 1.0}),
        lp_entry("SCAM", {"SCAM": 0.45, "SAFE": 0.15, "SUSPICIOUS": 0.1, "UN": 0.3}),
    ]
    assert detector.p_scam_from_logprobs(lp) == pytest.approx(0.45 / 0.7)


def test_p_scam_shared_prefix_tokenization():
    # "S" is shared by all labels; they diverge on the next token.
    lp = [
        lp_entry('{"verdict":"', {'{"verdict":"': 1.0}),
        lp_entry("S", {"S": 0.90, "SC": 0.05, "SA": 0.03, "UN": 0.02}),
        lp_entry("CAM", {"CAM": 0.6, "U": 0.3, "AFE": 0.1}),
        lp_entry('"', {'"': 1.0}),
    ]
    scam, safe, susp = 0.05 + 0.9 * 0.6, 0.03 + 0.9 * 0.1, 0.9 * 0.3
    assert detector.p_scam_from_logprobs(lp) == pytest.approx(scam / (scam + safe + susp))


def test_p_scam_two_level_shared_prefix():
    # "SU" shared by SUSPICIOUS only; "S" shared by all: walk two positions.
    lp = [
        lp_entry('{"verdict":"', {'{"verdict":"': 1.0}),
        lp_entry("S", {"S": 0.99, "UN": 0.01}),
        lp_entry("U", {"U": 0.8, "CAM": 0.15, "AFE": 0.05}),
        lp_entry("SPICIOUS", {"SPICIOUS": 1.0}),
    ]
    assert detector.p_scam_from_logprobs(lp) == pytest.approx(0.15 / (0.15 + 0.05 + 0.8))


def test_p_scam_key_quote_and_value_in_one_token():
    # The tokenizer glued the opening quote to the first label piece.
    lp = [
        lp_entry('{"verdict":"S', {'{"verdict":"S': 0.9, '{"verdict":"SC': 0.1}),
        lp_entry("CAM", {"CAM": 0.5, "AFE": 0.4, "U": 0.1}),
    ]
    scam, safe, susp = 0.1 + 0.9 * 0.5, 0.9 * 0.4, 0.9 * 0.1
    assert detector.p_scam_from_logprobs(lp) == pytest.approx(scam / (scam + safe + susp))


def test_glued_quote_token_aware_credits_suspicious():
    # Qwen pieces: SCAM = SC|AM, SUSPICIOUS = S|US|PIC|IOUS; here the quote is glued to the
    # first piece. `"S` belongs to SUSPICIOUS only (review: was dropped → P(SCAM) 0.923).
    lp = [
        lp_entry('{"verdict":', {'{"verdict":': 1.0}),
        lp_entry('"SC', {'"SC': 0.60, '"S': 0.35, '"SAFE': 0.05}),
        lp_entry("AM", {"AM": 1.0}),
    ]
    probs = detector.label_probs_from_logprobs(lp, detector.STATIC_LABEL_TOKENS)
    assert probs == pytest.approx({"SCAM": 0.60, "SUSPICIOUS": 0.35, "SAFE": 0.05})


def test_glued_quote_suspicious_sampled_walks_pieces():
    lp = [
        lp_entry('{"verdict":', {'{"verdict":': 1.0}),
        lp_entry('"S', {'"S': 0.7, '"SC': 0.2, '"SAFE': 0.1}),
        lp_entry("US", {"US": 1.0}),
        lp_entry("PIC", {"PIC": 1.0}),
    ]
    probs = detector.label_probs_from_logprobs(lp, detector.STATIC_LABEL_TOKENS)
    assert probs == pytest.approx({"SCAM": 0.2, "SUSPICIOUS": 0.7, "SAFE": 0.1})


def test_p_scam_spaced_json_like_base_model():
    lp = [
        lp_entry('{"', {'{"': 1.0}),
        lp_entry("verdict", {"verdict": 1.0}),
        lp_entry('":', {'":': 1.0}),
        lp_entry(' "', {' "': 1.0}),
        lp_entry("SC", {"SC": 0.90, "SA": 0.05, "S": 0.01, "UN": 0.04}),
        lp_entry("AM", {"AM": 1.0}),
    ]
    assert detector.p_scam_from_logprobs(lp) == pytest.approx(0.90 / 0.95)


def test_p_scam_none_without_logprobs_or_verdict_key():
    assert detector.p_scam_from_logprobs(None) is None
    assert detector.p_scam_from_logprobs([]) is None
    assert detector.p_scam_from_logprobs([lp_entry("hello", {"hello": 1.0})]) is None


def test_p_scam_handles_missing_alternatives_list():
    lp = [
        lp_entry('{"verdict":"', {'{"verdict":"': 1.0}),
        {"token": "SCAM", "logprob": math.log(0.9)},
    ]
    assert detector.p_scam_from_logprobs(lp) == pytest.approx(1.0)


# --- grounding --------------------------------------------------------------


def test_grounding_filter_drops_ungrounded_quotes():
    out = detector.DetectorOutput.model_validate(
        json.loads(
            _content(
                flags=[
                    {"quote": "will be blocked", "reason": "urgency_deadline"},
                    {"quote": "invented quote", "reason": "asks_payment"},
                ]
            )
        )
    )
    kept, rate = detector.ground_red_flags(out, TEXT)
    assert [f["quote"] for f in kept] == ["will be blocked"]
    assert kept[0]["source"] == "model"
    assert rate == 0.5


def test_grounding_rate_none_without_flags():
    out = detector.DetectorOutput.model_validate(json.loads(_content("SAFE", [], "personal")))
    assert detector.ground_red_flags(out, TEXT) == ([], None)


# --- detect() ---------------------------------------------------------------


async def test_detect_happy_path_request_shape(hub_env):  # noqa: F811
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_completion(_content(), _scam_logprobs(0.93)))

    res = await detector.detect(
        "sms", "AX-HDFCBK", ["urgency_plus_payment"], TEXT, client=_client(handler)
    )
    assert res is not None and res.retries == 0
    assert res.output.verdict == "SCAM"
    assert res.p_scam == pytest.approx(0.93)
    assert res.p_safe == pytest.approx(0.035)
    assert res.grounding_rate == 1.0
    assert res.red_flags == [
        {"quote": "will be blocked", "reason": "urgency_deadline", "source": "model"}
    ]
    body = seen[0]
    assert len(seen) == 1
    assert body["temperature"] == 0 and body["logprobs"] is True and body["top_logprobs"] >= 10
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "response_format" not in body
    user = body["messages"][1]["content"]
    assert user.startswith("CHANNEL: sms\nSENDER: AX-HDFCBK (registered)")
    assert "RULE_SIGNALS: [urgency_plus_payment]" in user and TEXT in user


async def test_detect_retries_once_with_json_schema(hub_env):  # noqa: F811
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(200, json=_completion("Sure! Here is my answer"))
        return httpx.Response(200, json=_completion(_content(), _scam_logprobs()))

    res = await detector.detect("sms", None, [], TEXT, client=_client(handler))
    assert res is not None and res.retries == 1
    assert "response_format" not in bodies[0]
    rf = bodies[1]["response_format"]
    assert rf["type"] == "json_schema" and "verdict" in rf["json_schema"]["schema"]["properties"]


async def test_detect_schema_violation_counts_as_invalid(hub_env):  # noqa: F811
    bad = json.dumps({"verdict": "MAYBE", "category": "x", "red_flags": []})
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json=_completion(bad))

    assert await detector.detect("sms", None, [], TEXT, client=_client(handler)) is None
    assert len(calls) == 2  # original + one constrained retry, then give up


async def test_detect_none_after_two_invalid(hub_env):  # noqa: F811
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion("not json"))

    assert await detector.detect("sms", None, [], TEXT, client=_client(handler)) is None


async def test_detect_unavailable_on_connection_error(hub_env):  # noqa: F811
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(detector.DetectorUnavailable):
        await detector.detect("sms", None, [], TEXT, client=_client(handler))


async def test_detect_unavailable_on_timeout_and_5xx(hub_env):  # noqa: F811
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    with pytest.raises(detector.DetectorUnavailable):
        await detector.detect("sms", None, [], TEXT, client=_client(timeout))
    with pytest.raises(detector.DetectorUnavailable):
        await detector.detect(
            "sms", None, [], TEXT, client=_client(lambda r: httpx.Response(503, text="loading"))
        )


async def test_detect_falls_back_to_label_without_logprobs(hub_env):  # noqa: F811
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion(_content("SAFE", [], "personal")))

    res = await detector.detect("sms", None, [], TEXT, client=_client(handler))
    assert res is not None and res.p_scam == 0.0 and res.grounding_rate is None
    assert res.p_safe == 1.0


@pytest.mark.parametrize(
    ("verdict", "p_scam", "p_safe"),
    [("SCAM", 1.0, 0.0), ("SUSPICIOUS", 0.0, 0.0), ("SAFE", 0.0, 1.0)],
)
async def test_detect_label_fallback_p_safe(hub_env, verdict, p_scam, p_safe):  # noqa: F811
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion(_content(verdict, [], "other_scam")))

    res = await detector.detect("sms", None, [], TEXT, client=_client(handler))
    assert res is not None and (res.p_scam, res.p_safe) == (p_scam, p_safe)


async def test_detect_p_safe_when_suspicious_sampled(hub_env):  # noqa: F811
    lp = [
        lp_entry('{"verdict":"', {'{"verdict":"': 1.0}),
        lp_entry("SUSPICIOUS", {"SUSPICIOUS": 0.9898, "SAFE": 0.01, "SCAM": 0.0002}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion(_content("SUSPICIOUS", [], "other_scam"), lp))

    res = await detector.detect("sms", None, [], TEXT, client=_client(handler))
    assert res.p_scam == pytest.approx(0.0002) and res.p_safe == pytest.approx(0.01)


async def test_detect_grounds_against_normalized_text(hub_env):  # noqa: F811
    # Raw text has a zero-width space and double spaces; the quote matches the normalized form.
    raw = "Your  account​ will be   blocked today."

    def handler(request: httpx.Request) -> httpx.Response:
        flags = [{"quote": "account will be blocked", "reason": "urgency_deadline"}]
        return httpx.Response(200, json=_completion(_content(flags=flags), _scam_logprobs()))

    res = await detector.detect("sms", None, [], raw, client=_client(handler))
    assert res is not None and res.grounding_rate == 1.0
