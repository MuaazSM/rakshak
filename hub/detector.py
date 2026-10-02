"""Detector client for llama-server: prompt, JSON parse/retry, p_scam from logprobs, grounding filter (PRD FR-13, FR-14, §8.2).

`user_message` and `target_json` define the §8.2 I/O format; the dataset builders use them
too, so training and inference inputs are identical.
"""

import json

from hub import prompts
from hub.normalize import sender_status
from hub.schemas import DetectorOutput


def user_message(channel: str, sender: str | None, rule_signals: list[str], text: str) -> str:
    """§8.2 user message. `text` must already be normalized; signals are sorted for determinism."""
    return prompts.DETECTOR_USER.format(
        channel=channel,
        sender=(sender or "").strip() or "unknown",
        sender_status=sender_status(sender),
        rule_signals=", ".join(sorted(set(rule_signals))),
        text=text,
    )


def target_json(out: DetectorOutput) -> str:
    """§8.2 assistant output: compact JSON, keys in order verdict, category, red_flags."""
    return json.dumps(out.model_dump(), ensure_ascii=False, separators=(",", ":"))


def chat_example(
    channel: str, sender: str | None, rule_signals: list[str], text: str, out: DetectorOutput
) -> list[dict]:
    """System (A.1) + user (§8.2) + assistant target, as chat messages."""
    return [
        {"role": "system", "content": prompts.DETECTOR_SYSTEM},
        {"role": "user", "content": user_message(channel, sender, rule_signals, text)},
        {"role": "assistant", "content": target_json(out)},
    ]
