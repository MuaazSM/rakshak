"""Detector client for llama-server: prompt, JSON parse/retry, p_scam from logprobs, grounding filter (PRD FR-13, FR-14, §8.2).

`user_message` and `target_json` define the §8.2 I/O format; the dataset builders use them
too, so training and inference inputs are identical.
"""

import json
import math
import re
from dataclasses import dataclass

import httpx
from pydantic import ValidationError

from hub import prompts
from hub.normalize import normalize_text, sender_status
from hub.schemas import DetectorOutput
from hub.settings import get_settings

LABELS = ("SCAM", "SUSPICIOUS", "SAFE")
TOP_LOGPROBS = 10
MAX_TOKENS = 256
TIMEOUT_S = 30.0
_VERDICT_KEY = re.compile(r'"verdict"\s*:\s*"')


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


class DetectorUnavailable(Exception):
    """llama-server is unreachable or timed out (callers fall back to rules-only, never SAFE)."""


@dataclass
class DetectorResult:
    output: DetectorOutput  # parsed, red flags NOT yet grounding-filtered
    p_scam: float  # P(verdict == SCAM) renormalized over the three labels
    red_flags: list[dict]  # grounded flags only: {"quote", "reason", "source": "model"}
    grounding_rate: float | None  # kept / total before filtering; None if the model gave no flags
    raw: str  # raw model text (never log it)
    retries: int  # 0 or 1 (constrained retry used)


def p_scam_from_logprobs(logprob_content: list[dict] | None) -> float | None:
    """P(verdict == "SCAM") from the token logprobs of a chat completion (PRD FR-13).

    `logprob_content` is `choices[0].logprobs.content`: one entry per generated token with
    `token`, `logprob` and `top_logprobs` (alternatives at that position, incl. the sampled one).

    Method. The generated tokens are joined to find the character offset of the verdict value
    (right after `"verdict"<ws>:<ws>"`), independent of how the tokenizer split the key, the
    colon or the quote. From the token that covers that offset we walk forward along the
    *sampled* path carrying a weight w (initially 1). At each position every alternative token
    t, appended to the value prefix consumed so far, is matched against the labels:
    - consistent with exactly one label (prefix of it, or extending it, e.g. `SC`, `SCAM"`):
      w * P(t) is added to that label and the mass is resolved;
    - consistent with several labels (e.g. `S`, shared by SCAM/SAFE/SUSPICIOUS): only the
      sampled token can be followed; its mass becomes the new w for the next position, where
      the same rule splits it further. This is the shared-prefix case: the labels diverge at a
      later token and the mass is combined from every position at which they diverge;
    - consistent with no label, or an ambiguous non-sampled alternative: dropped.
    The three label masses are renormalized to sum to 1 and P(SCAM) is returned. Returns None
    when the verdict position cannot be located or no label mass was found (the caller then
    falls back to the label itself).
    """
    if not logprob_content:
        return None
    texts = [str(t.get("token", "")) for t in logprob_content]
    joined = "".join(texts)
    m = _VERDICT_KEY.search(joined)
    if not m:
        return None
    start = m.end()
    pos, idx = 0, None
    for i, tok in enumerate(texts):
        if pos + len(tok) > start:
            idx = i
            break
        pos += len(tok)
    if idx is None:
        return None
    skip = start - pos  # chars of the first token that precede the value

    mass = dict.fromkeys(LABELS, 0.0)
    weight, consumed = 1.0, ""
    for i in range(idx, len(logprob_content)):
        entry = logprob_content[i]
        sampled = texts[i]
        cut = skip if i == idx else 0
        nxt: tuple[float, str] | None = None
        alts = list(entry.get("top_logprobs") or [])
        if not any(a.get("token") == sampled for a in alts):
            alts.append({"token": sampled, "logprob": entry.get("logprob", -math.inf)})
        for alt in alts:
            tok = str(alt.get("token", ""))
            if tok[:cut] != sampled[:cut]:
                continue
            s = consumed + tok[cut:]
            if not s:
                continue
            matches = [L for L in LABELS if L.startswith(s) or s.startswith(L)]
            prob = math.exp(alt.get("logprob", -math.inf))
            if len(matches) == 1:
                mass[matches[0]] += weight * prob
            elif len(matches) > 1 and tok == sampled:
                nxt = (weight * prob, s)
        if nxt is None:
            break
        weight, consumed = nxt
    total = sum(mass.values())
    if total <= 0:
        return None
    return mass["SCAM"] / total


def ground_red_flags(output: DetectorOutput, text: str) -> tuple[list[dict], float | None]:
    """FR-14: keep flags whose quote is an exact substring of the normalized `text`."""
    flags = output.red_flags
    kept = [
        {"quote": f.quote, "reason": f.reason, "source": "model"}
        for f in flags
        if f.quote and f.quote in text
    ]
    return kept, (len(kept) / len(flags) if flags else None)


def _parse(content: str) -> DetectorOutput | None:
    try:
        return DetectorOutput.model_validate(json.loads(content))
    except (ValueError, ValidationError):
        return None


def _request_body(messages: list[dict], constrained: bool) -> dict:
    body: dict = {
        "model": "rakshak-detector",
        "messages": messages,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "logprobs": True,
        "top_logprobs": TOP_LOGPROBS,
        # llama-server runs Qwen3.5 with --jinja: disable the thinking preamble.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if constrained:
        # Verified on this llama-server build: OpenAI-style response_format json_schema works.
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "detector",
                "strict": True,
                "schema": DetectorOutput.model_json_schema(),
            },
        }
    return body


async def _call(client: httpx.AsyncClient, url: str, body: dict) -> tuple[str, list[dict] | None]:
    try:
        r = await client.post(f"{url.rstrip('/')}/chat/completions", json=body)
        r.raise_for_status()
        choice = r.json()["choices"][0]
    except (httpx.TransportError, httpx.HTTPStatusError) as e:
        raise DetectorUnavailable(type(e).__name__) from e
    except (ValueError, KeyError, IndexError) as e:
        raise DetectorUnavailable("BadResponse") from e
    content = (choice.get("message") or {}).get("content") or ""
    return content, (choice.get("logprobs") or {}).get("content")


async def detect(
    channel: str,
    sender: str | None,
    rule_signals: list[str],
    text: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> DetectorResult | None:
    """FR-13/FR-14. Returns None when the output is invalid JSON twice (rules-only mode);
    raises DetectorUnavailable when the server is down. `text` is normalized here again
    (idempotent) so the grounding filter and the prompt use the same string."""
    text = normalize_text(text)
    messages = [
        {"role": "system", "content": prompts.DETECTOR_SYSTEM},
        {"role": "user", "content": user_message(channel, sender, rule_signals, text)},
    ]
    url = get_settings().detector_url
    own = client is None
    client = client or httpx.AsyncClient(timeout=TIMEOUT_S)
    try:
        for retries, constrained in enumerate((False, True)):
            content, lp = await _call(client, url, _request_body(messages, constrained))
            out = _parse(content)
            if out is None:
                continue
            p = p_scam_from_logprobs(lp)
            if p is None:  # no logprobs: fall back to the label itself
                p = {"SCAM": 1.0, "SUSPICIOUS": 0.5, "SAFE": 0.0}[out.verdict]
            flags, rate = ground_red_flags(out, text)
            return DetectorResult(out, p, flags, rate, content, retries)
        return None
    finally:
        if own:
            await client.aclose()
