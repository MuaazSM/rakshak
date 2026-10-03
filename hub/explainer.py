"""Explainer: Gemma verdict explanation in en/hi with post-check and template fallback
(PRD FR-16, §7.5, Appendix A.3; §17 "think": false, fixed SAFE template).

Only SCAM and SUSPICIOUS go to Gemma. SAFE and UNKNOWN always use the fixed template, and so
does any Gemma failure. The explainer never changes the verdict and never raises.
Templates (hub/data/templates/{lang}.json) carry the copy of the design frames B6-B9.
Nothing here logs or traces message text or explanations.
"""

import json
import logging
import re
from functools import lru_cache
from pathlib import Path

import httpx

from hub import prompts
from hub.settings import Parent, get_settings

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
GEMMA_VERDICTS = ("SCAM", "SUSPICIOUS")
MAX_SENTENCES = 3
MAX_WORDS = 60
TIMEOUT_S = 60.0
OLLAMA_KEEP_ALIVE = -1  # keep Gemma resident so it doesn't reload between checks
NEGATION_WINDOW = 3  # words checked around a verdict word

_SENTENCE_SPLIT = re.compile(r"[.!?।]+")
_URL = re.compile(r"(?:https?://|www\.)\S+|\b[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}(?:/\S*)?", re.I)
_NUMBER = re.compile(r"\d+")
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_WORD_SPLIT = re.compile(r"[\s()\"“”]+")
_CLAUSE_SPLIT = re.compile(r"[,;:—–]|\s-\s|\b(?:but|and|so)\b")
_NEGATORS = {
    "en": {"not", "no", "never", "cannot", "without"},
    "hi": {"नहीं", "नही", "न", "ना", "मत"},
}
# Verdict words that must not appear in an explanation of this verdict (A.3: never change it).
# SCAM may still say "be careful"; SUSPICIOUS may not say "scam".
_FORBIDDEN_VERDICTS = {
    "SCAM": ("SAFE", "UNKNOWN"),
    "SUSPICIOUS": ("SCAM", "SAFE", "UNKNOWN"),
}


@lru_cache
def _verdict_words() -> dict[str, dict[str, list[str]]]:
    return json.loads((DATA_DIR / "verdict_words.json").read_text("utf-8"))


@lru_cache
def _templates(lang: str) -> dict[str, dict[str, str]]:
    path = DATA_DIR / "templates" / f"{lang}.json"
    if not path.exists():
        path = DATA_DIR / "templates" / "en.json"
    return json.loads(path.read_text("utf-8"))


def template(verdict: str, category: str | None, lang: str, son: str) -> str:
    """Fixed explanation: headline, then the verdict/category sentence(s) (design B6-B9)."""
    tpl = _templates(lang)
    entry = tpl.get(verdict) or tpl["UNKNOWN"]
    body = entry.get(category or "", entry["default"])
    return f"{entry['headline']}. {body.format(son=son)}"


def _numbers(text: str) -> set[str]:
    return set(_NUMBER.findall(text.translate(_DEVANAGARI_DIGITS)))


def _tokens(text: str) -> list[str]:
    return [t for t in _WORD_SPLIT.split(text) if t]


def _is_negator(token: str, lang: str) -> bool:
    return token in _NEGATORS.get(lang, _NEGATORS["en"]) or (lang != "hi" and token.endswith("n't"))


def _negated(sentence: str, word: str, lang: str) -> bool:
    """True if `word` occurs in `sentence` with a negator just before it (en) or just before
    or after it (hi: "यह स्कैम नहीं है")."""
    start = sentence.find(word)
    while start != -1:
        # a negator in another clause ("Do not reply, be careful") doesn't count
        before = _tokens(_CLAUSE_SPLIT.split(sentence[:start])[-1])[-NEGATION_WINDOW:]
        after = _tokens(_CLAUSE_SPLIT.split(sentence[start + len(word) :])[0])[:NEGATION_WINDOW]
        near = before + after if lang == "hi" else before
        if any(_is_negator(t, lang) for t in near):
            return True
        start = sentence.find(word, start + 1)
    return False


def verdict_word_problems(output: str, verdict: str, lang: str) -> list[str]:
    """Own verdict word present and not negated; no other verdict's word (en/hi tables)."""
    table = _verdict_words().get(lang) or _verdict_words()["en"]
    low = output.lower().replace("’", "'")
    own = [w.lower().replace("’", "'") for w in table.get(verdict, [])]
    if not any(w in low for w in own):
        return ["verdict_word_missing"]
    problems = []
    sentences = [x for x in _SENTENCE_SPLIT.split(low) if x.strip()]
    if any(_negated(sen, w, lang) for sen in sentences for w in own):
        problems.append("negated_verdict_word")
    others = [
        w.lower().replace("’", "'")
        for v in _FORBIDDEN_VERDICTS.get(verdict, ())
        for w in table.get(v, [])
    ]
    if any(w in low for w in others):
        problems.append("other_verdict_word")
    return problems


def post_check(output: str, verdict: str, lang: str, input_text: str) -> list[str]:
    """PRD §7.5. Returns the list of failed checks (empty = pass)."""
    problems: list[str] = []
    problems += verdict_word_problems(output, verdict, lang)
    inp = input_text.lower()
    if any(u.lower().rstrip(".,)") not in inp for u in _URL.findall(output)):
        problems.append("foreign_url")
    if not _numbers(output) <= _numbers(input_text):
        problems.append("foreign_number")
    if len([s for s in _SENTENCE_SPLIT.split(output) if s.strip()]) > MAX_SENTENCES:
        problems.append("too_many_sentences")
    if len(output.split()) > MAX_WORDS:
        problems.append("too_many_words")
    return problems


def _messages(verdict: dict, parent: Parent, lang: str) -> list[dict]:
    flags = [
        {"quote": f.get("quote", ""), "reason": f.get("reason", "")}
        for f in verdict.get("red_flags") or []
        if isinstance(f, dict)
    ]
    verdict_json = {
        "verdict": verdict["verdict"],
        "category": verdict.get("category"),
        "red_flags": flags,
    }
    system = prompts.EXPLAINER_SYSTEM.format(
        PARENT_NAME=parent.name,
        PARENT_AGE=parent.age,
        LANGUAGE_NAME=prompts.LANGUAGE_NAMES.get(lang, prompts.LANGUAGE_NAMES["en"]),
        SON_NAME=get_settings().son_name,
    )
    user = prompts.EXPLAINER_USER.format(verdict_json=json.dumps(verdict_json, ensure_ascii=False))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def _generate(
    client: httpx.AsyncClient, messages: list[dict], *, temperature: float, seed: int
) -> str:
    s = get_settings()
    body = {
        "model": s.gemma_model,
        "messages": messages,
        "stream": False,
        "think": False,  # PRD §17: Gemma 4 thinks by default (12-14 s); off = 1.2-1.8 s
        "options": {"temperature": temperature, "seed": seed},
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    r = await client.post(f"{s.ollama_host}/api/chat", json=body, timeout=TIMEOUT_S)
    r.raise_for_status()
    return (r.json().get("message", {}).get("content") or "").strip().strip('"“”').strip()


async def explain(
    verdict: dict,
    parent: Parent,
    lang: str,
    input_text: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Explanation for the parent in `lang`. `verdict` has verdict, category, red_flags."""
    label = verdict.get("verdict", "UNKNOWN")
    category = verdict.get("category")
    son = get_settings().son_name
    try:
        fallback = template(label, category, lang, son)
    except Exception:  # corrupt template file: last resort, still never raise
        return "Couldn't check right now."
    if label not in GEMMA_VERDICTS:
        return fallback
    own = client is None
    http = client or httpx.AsyncClient()
    try:
        messages = _messages(verdict, parent, lang)
        for attempt, (temperature, seed) in enumerate(((0.0, 7), (0.4, 11))):
            try:
                out = await _generate(http, messages, temperature=temperature, seed=seed)
            except (httpx.HTTPError, ValueError, KeyError) as e:
                log.warning(
                    "explainer gemma call failed (%s), attempt %d", type(e).__name__, attempt
                )
                return fallback  # server down: a second try won't help
            problems = post_check(out, label, lang, input_text) if out else ["empty"]
            if not problems:
                return out
            log.info("explainer post-check failed: %s (attempt %d)", ",".join(problems), attempt)
        return fallback
    except Exception as e:  # never raise (module contract)
        log.warning("explainer error (%s)", type(e).__name__)
        return fallback
    finally:
        if own:
            await http.aclose()
