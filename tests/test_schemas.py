import json
import typing

import pytest
from pydantic import ValidationError

from hub.schemas import CheckRequest, CheckState, DetectorOutput, Feedback, Verdict

# PRD §8.1 example, with the enum placeholders resolved to concrete values.
VERDICT_EXAMPLE = """
{
  "event_id": "evt_01J...",
  "verdict": "SCAM",
  "category": "digital_arrest",
  "p_scam": 0.97,
  "red_flags": [{"quote": "will be blocked today", "reason": "urgency_deadline", "source": "model"}],
  "explanation": "यह धोखा है। ...",
  "language": "hi",
  "parent_id": "mom",
  "timings_ms": {"normalize": 2, "perceive": 0, "rules": 1, "detect": 840, "fuse": 0, "explain": 1900},
  "model_versions": {"detector": "rakshak-detector-v1-q4km", "gemma": "gemma4:e2b"}
}
"""

# PRD §8.2 example detector output.
DETECTOR_EXAMPLE = (
    '{"verdict":"SCAM","category":"kyc_account_block",'
    '"red_flags":[{"quote":"...","reason":"urgency_deadline"}]}'
)


def test_verdict_round_trip():
    data = json.loads(VERDICT_EXAMPLE)
    v = Verdict.model_validate(data)
    dumped = v.model_dump(mode="json")
    assert dumped == data
    assert list(dumped) == list(data)  # key order matches PRD
    assert Verdict.model_validate_json(v.model_dump_json()) == v


@pytest.mark.parametrize("label", ["SCAM", "SUSPICIOUS", "SAFE", "UNKNOWN"])
def test_verdict_labels(label):
    data = json.loads(VERDICT_EXAMPLE) | {"verdict": label}
    assert Verdict.model_validate(data).verdict == label


@pytest.mark.parametrize(
    "patch",
    [
        {"verdict": "MAYBE"},
        {"language": "mr"},
        {"category": "not_a_category"},
        {"red_flags": [{"quote": "x", "reason": "y", "source": "llm"}]},
        {"extra_field": 1},
    ],
)
def test_verdict_rejects_invalid(patch):
    with pytest.raises(ValidationError):
        Verdict.model_validate(json.loads(VERDICT_EXAMPLE) | patch)


def test_detector_output_round_trip():
    d = DetectorOutput.model_validate_json(DETECTOR_EXAMPLE)
    assert d.model_dump_json() == DETECTOR_EXAMPLE  # same keys, same order


def test_detector_output_rejects_unknown_verdict_and_reason():
    with pytest.raises(ValidationError):
        DetectorOutput.model_validate_json(DETECTOR_EXAMPLE.replace('"SCAM"', '"UNKNOWN"'))
    with pytest.raises(ValidationError):
        DetectorOutput.model_validate_json(DETECTOR_EXAMPLE.replace("urgency_deadline", "vibes"))


def test_check_request_and_feedback():
    req = CheckRequest.model_validate({"parent_id": "dad", "text": "hi", "channel": "sms"})
    assert req.lang is None and req.sender is None
    with pytest.raises(ValidationError):
        CheckRequest.model_validate({"parent_id": "dad", "text": "hi", "channel": "email"})
    fb = Feedback.model_validate({"correct": False, "true_verdict": "SAFE"})
    assert fb.note is None


def test_check_state_keys_match_prd():
    assert list(typing.get_type_hints(CheckState)) == [
        "event_id", "parent_id", "channel", "raw_text", "image", "audio", "text", "urls",
        "phones", "upi_handles", "sender", "rule_signals", "detector", "p_scam", "verdict",
        "red_flags", "grounding_rate", "explanation", "language", "timings_ms", "errors",
    ]  # fmt: skip
