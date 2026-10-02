"""Shared helpers for the Phase 1 smoke tests (IMPLEMENTATION.md Phase 1).

Privacy (CLAUDE.md hard rules): the sample messages here are synthetic. Anything derived
from a real screenshot or voice note is written only under `data/raw/smoke/` (git-ignored)
and printed only with an explicit `--show`.
"""

import json
import re
import sys
import time
from pathlib import Path

import httpx

from hub import prompts
from hub.schemas import DetectorOutput
from hub.settings import get_settings

ROOT = Path(__file__).resolve().parents[2]
PRIVATE_OUT = ROOT / "data" / "raw" / "smoke"

# Synthetic scam SMS (reserved .example domain, no real numbers or names).
SAMPLE_SCAM = (
    "Dear customer, your SBI account will be blocked today due to incomplete KYC. "
    "Update now at http://sbi-kyc-verify.example/login to avoid suspension."
)
SAMPLE_SCAM_FLAGS = [
    {"quote": "will be blocked today", "reason": "urgency_deadline"},
    {"quote": "http://sbi-kyc-verify.example/login", "reason": "lookalike_link"},
]

URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b[\w-]+(?:\.[\w-]+)+/\S*", re.IGNORECASE)


def settings():
    return get_settings()


def banner(title: str) -> None:
    print(f"\n=== {title} ===")


def verdict_line(ok: bool, name: str, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    return ok


def finish(ok: bool) -> None:
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


def save_private(name: str, text: str) -> Path:
    """Write a transcript of real input under data/raw/smoke/ (git-ignored)."""
    PRIVATE_OUT.mkdir(parents=True, exist_ok=True)
    path = PRIVATE_OUT / name
    path.write_text(text, encoding="utf-8")
    return path


def private_note(path: Path) -> None:
    """Warn if a real input file lives outside data/raw/."""
    raw = (ROOT / "data" / "raw").resolve()
    if raw not in path.resolve().parents:
        print(f"note: {path} is outside data/raw/; keep real messages under data/raw/")


def similarity(a: str, b: str) -> float:
    from rapidfuzz import fuzz

    return fuzz.ratio(" ".join(a.split()), " ".join(b.split())) / 100


def links(text: str) -> list[str]:
    return [u.rstrip(".,)") for u in URL_RE.findall(text)]


def pct(values: list[float], q: float) -> float:
    """Nearest-rank percentile."""
    s = sorted(values)
    k = max(0, min(len(s) - 1, round(q / 100 * len(s) + 0.5) - 1))
    return s[k]


# --- Ollama (Gemma) ---


def ollama_chat(
    messages: list[dict], think: bool = False, timeout: float = 180
) -> tuple[dict, float]:
    """POST /api/chat (non-streaming). Returns (response json, wall ms).

    Gemma 4 thinks by default in Ollama (~500 extra tokens); `think=False` turns it off.
    """
    s = settings()
    body = {
        "model": s.gemma_model,
        "messages": messages,
        "stream": False,
        "think": think,
        "options": {"temperature": 0, "seed": 7},
        "keep_alive": "15m",
    }
    t0 = time.perf_counter()
    r = httpx.post(f"{s.ollama_host}/api/chat", json=body, timeout=timeout)
    ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return r.json(), ms


def ollama_timing(resp: dict) -> str:
    load = resp.get("load_duration", 0) / 1e6
    ev, evd = resp.get("eval_count", 0), resp.get("eval_duration", 0) / 1e9
    tps = f"{ev / evd:.1f} tok/s" if evd else "n/a"
    return f"load {load:.0f} ms, {ev} tokens out, {tps}"


def explainer_messages(verdict_json: dict, lang: str, name: str, age: int) -> list[dict]:
    s = settings()
    system = prompts.EXPLAINER_SYSTEM.format(
        PARENT_NAME=name,
        PARENT_AGE=age,
        LANGUAGE_NAME=prompts.LANGUAGE_NAMES[lang],
        SON_NAME=s.son_name,
    )
    user = prompts.EXPLAINER_USER.format(verdict_json=json.dumps(verdict_json, ensure_ascii=False))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# --- Detector (llama-server / Tinker) ---


def detector_messages(text: str = SAMPLE_SCAM) -> list[dict]:
    user = prompts.DETECTOR_USER.format(
        channel="sms",
        sender="VK-SBIUPD",
        sender_status="unregistered",
        rule_signals="",
        text=text,
    )
    return [
        {"role": "system", "content": prompts.DETECTOR_SYSTEM},
        {"role": "user", "content": user},
    ]


def detector_chat(text: str = SAMPLE_SCAM, timeout: float = 120) -> tuple[str, float]:
    """OpenAI-compatible chat on llama-server with thinking off (PRD §7.1)."""
    s = settings()
    body = {
        "model": s.detector_version,
        "messages": detector_messages(text),
        "temperature": 0,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    r = httpx.post(f"{s.detector_url}/chat/completions", json=body, timeout=timeout)
    ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or "", ms


def check_detector_json(raw: str, message: str = SAMPLE_SCAM) -> tuple[bool, str]:
    """Pass = the reply parses as JSON (Phase 1 bar for a base model). The detail says whether
    it also matches the strict §8.2 schema and whether quotes are exact substrings."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return False, f"not JSON: {e}"
    try:
        out = DetectorOutput.model_validate(data)
    except Exception as e:
        first = str(e).splitlines()[1:3]
        return (
            True,
            f"JSON ok; §8.2 schema mismatch (expected before fine-tuning): {str(first)[:160]}",
        )
    bad = [f.quote for f in out.red_flags if f.quote not in message]
    detail = f"JSON ok; schema ok; verdict={out.verdict} category={out.category} flags={len(out.red_flags)}"
    return True, detail + (f"; non-substring quotes: {bad}" if bad else "; quotes exact")
