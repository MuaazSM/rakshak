"""Synthetic variant generation with open-weight models only (PRD FR-31, §10.4).

    uv run python -m training.synth [--batch-id batch1] [--seeds data/seeds/seeds.jsonl]
        [--out data/synthetic] [--scenarios 5] [--variants 5] [--until HH:MM] [--concurrency 4]

Seeds (a handful of hand-written messages, never training data themselves) are expanded in two
levels by Gemma (Ollama, `settings.gemma_model`), so every training/dev example is model-produced:

  (a) scenarios: `--scenarios` new situations of the same category/intent (obfuscation none,
      language rotated), key `s{i}`;
  (b) variants: each scenario rewritten `--variants` times over language x obfuscation x channel
      (P3-gen prompt, PROMPTBOOK P3.1), key `s{i}.v{j}`.

SAFE seeds use the P3-hardneg prompt instead (genuine messages with alarming wording); each
(scenario, variant) slot is one independent hardneg call.

Gemma also returns red flags as [{"quote", "reason"}]; every quote must be an exact substring of
`normalize_text(text)` and every reason must be in the seed's reason set (a subset of §10.1),
else the item is dropped. SAFE items carry no red flags.

Output is one JSON object per line in `{out}/{batch_id}.jsonl` (append-only; a rerun skips keys
already present) plus a 10% sample in `review_{batch_id}.jsonl` for hand checking. Only counts
are printed, never message text.
"""

import argparse
import asyncio
import json
import random
import re
import sys
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import get_args

import httpx

from hub.normalize import extract_urls, normalize, normalize_text, sender_status
from hub.rules import _host, _is_allowed, evaluate, load_rule_data
from hub.schemas import DetectorVerdict, Reason, SafeCategory, ScamCategory

BUILD_SEED = 20261003
VERDICTS = tuple(get_args(DetectorVerdict))
SCAM_CATEGORIES = tuple(get_args(ScamCategory))
SAFE_CATEGORIES = tuple(get_args(SafeCategory))
REASONS = tuple(get_args(Reason))
LANGUAGES = ("en", "hinglish", "hi")
OBFUSCATIONS = ("none", "misspell", "spacing", "emoji", "lookalike_chars", "short_link")
CHANNELS = ("sms", "whatsapp")
SEED_CHANNELS = (*CHANNELS, "call_description")
GENERATOR = "gemma4:e2b"
REVIEW_FRACTION = 0.10
MAX_RETRIES = 3
MAX_CONN_ERRORS = 12  # connection errors are retried separately, with backoff

LANGUAGE_DESC = {
    "en": "English",
    "hinglish": "hinglish (Hindi written in Latin script mixed with English)",
    "hi": "Hindi written in Devanagari script",
}
OBFUSCATION_DESC = {
    "none": "none",
    "misspell": "misspell several key words the way scammers do (e.g. 'acount', 'blokd', 'verifiy')",
    "spacing": "put odd spaces or dots inside a few key words (e.g. 'B l o c k e d', 'K.Y.C')",
    "emoji": "add several emoji such as warning signs and money bags around key phrases",
    "lookalike_chars": "swap some letters for look-alike characters (e.g. Cyrillic 'а', zero for 'o')",
    "short_link": "the link must be a shortened link (bit.ly, tinyurl.com, cutt.ly, rb.gy style)",
}
CATEGORY_DESC = {
    "genuine_otp": "bank or app one-time-password SMS (with a warning not to share the OTP)",
    "transaction_alert": "bank debit/credit alert, or a KYC-update reminder asking the customer to visit a branch",
    "delivery_update": "courier, e-commerce or post-office delivery update",
    "legit_promo": "ordinary promotional message from a known brand or telecom operator",
    "govt_genuine": "genuine government or utility notice (tax, provident fund, electricity bill)",
    "personal": "ordinary chat message from a family member or friend (no ask for money, links or codes)",
}

# Words that make a genuine message "alarming" (hard negatives, PRD §10.4).
_ALARM = re.compile(
    r"do not share|don't share|never asks|debited|not done by you|not you|blocked|suspend|"
    r"urgent|kyc|expire|turant|nahi kiya|share na karein|साझा न करें|तुरंत|डेबिट|केवाईसी|समाप्त",
    re.IGNORECASE,
)
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_PLACEHOLDER = re.compile(
    r"\[(?!OTP\b|ACCT\b|PHONE\b)[^\]]{0,40}\]|<(?!OTP>|ACCT>|PHONE>)[A-Za-z ]{2,30}>"
)

_NAMES = ["Sharma", "Verma", "Patil", "Iyer", "Khan", "Gupta", "Nair", "Reddy", "Joshi", "Singh"]
_CITIES = [
    "Pune",
    "Nagpur",
    "Lucknow",
    "Indore",
    "Jaipur",
    "Surat",
    "Kochi",
    "Patna",
    "Bhopal",
    "Thane",
]

Complete = Callable[[str], Awaitable[str]]


class SynthError(Exception):
    """Invalid seed or setup. Messages carry ids only, never text."""


# --- seeds ---------------------------------------------------------------------------------


def validate_item(it: dict, *, allow_empty_flags: bool = False) -> list[str]:
    """Problems with a seed-shaped item (empty list = valid). Quotes are checked against the
    normalized text, as the detector sees it."""
    errs = []
    text = normalize_text(it.get("text") or "")
    if not text:
        errs.append("empty text")
    if it.get("verdict") not in VERDICTS:
        errs.append("bad verdict")
    cats = SAFE_CATEGORIES if it.get("verdict") == "SAFE" else (*SCAM_CATEGORIES,)
    if it.get("category") not in cats:
        errs.append("category does not fit verdict")
    if it.get("channel") not in SEED_CHANNELS:
        errs.append("bad channel")
    if it.get("language") not in LANGUAGES:
        errs.append("bad language")
    flags = it.get("red_flags") or []
    if it.get("verdict") == "SAFE":
        if flags:
            errs.append("SAFE must have no red flags")
    elif not flags and not allow_empty_flags:
        errs.append("needs at least one red flag")
    for f in flags:
        if f.get("reason") not in REASONS:
            errs.append("bad reason")
        q = normalize_text(f.get("quote") or "")
        if not q or q not in text:
            errs.append("quote is not an exact substring")
    return errs


def load_seeds(path: Path, limit: int | None = 60) -> list[dict]:
    seeds = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    if limit is not None and len(seeds) > limit:
        raise SynthError(f"{path}: {len(seeds)} seeds, at most {limit} allowed")
    for s in seeds:
        errs = validate_item(s)
        if errs:
            raise SynthError(f"seed {s.get('id')!r}: {'; '.join(sorted(set(errs)))}")
    return seeds


# --- prompts -------------------------------------------------------------------------------


def hints(rng: random.Random, scam: bool = True) -> str:
    """Deterministic diversity nudges; the sender style only applies to scam-side items."""
    detail = (
        f"Details to use: name {rng.choice(_NAMES)}, city {rng.choice(_CITIES)}, "
        f"amount around Rs {rng.choice([199, 480, 1500, 2999, 7800, 14500, 36000])}."
    )
    if not scam:
        return detail
    sender = rng.choice(
        [
            "a mobile number like +91 9xxxx xxxxx",
            "a mobile number like +91 7xxxx xxxxx",
            "a short alphanumeric header that is NOT a real registered bank code",
        ]
    )
    return f"{detail} The sender should be {sender}."


def gen_prompt(
    seed_text: str, category: str, language: str, obfuscation: str, channel: str, detail: str,
    reasons: list[str], distinct: str = "",
) -> str:  # fmt: skip
    """P3-gen (PROMPTBOOK P3.1) with the red-flag extension recorded in PRD §17."""
    return (
        "You are helping build a dataset to protect elderly people in India from scams.\n"
        "Rewrite the SEED message as a new, realistic message with these properties:\n"
        f"- language/style: {LANGUAGE_DESC[language]}   (hinglish = Hindi written in Latin script mixed with English)\n"
        f"- obfuscation: {OBFUSCATION_DESC[obfuscation]}\n"
        f"- channel: {channel}\n"
        "Keep the same scam type and the same intent, but change names, amounts, links and wording.\n"
        "Use fake but realistic-looking domains and numbers; never use a real official domain for a scam link.\n"
        f"{distinct}{detail}\n"
        "Keep it short (1-4 sentences).\n"
        "Return JSON only:\n"
        '{"text": "<new message>", "sender": "<sender header or number>", '
        '"red_flags": [{"quote": "<exact substring of text>", "reason": "<one of: '
        + ", ".join(reasons)
        + '>"}]}\n'
        "Every quote must be copied character for character from your new text. Give 1-4 red flags.\n"
        f"SEED ({category}): {seed_text}"
    )


_ACTOR = {
    "genuine_otp": "bank or app",
    "transaction_alert": "bank",
    "delivery_update": "courier company, e-commerce site or India Post",
    "legit_promo": "well-known brand or telecom operator",
    "govt_genuine": "government department or electricity/water utility",
    "personal": "relative or friend",
}
_ALARM_HINT = {
    "genuine_otp": "a warning not to share the OTP",
    "transaction_alert": "a debit notice, or a KYC reminder to visit a branch",
    "delivery_update": "a failed delivery attempt, or a note that the parcel will be returned if not collected",
    "govt_genuine": "an overdue-bill reminder with a due date, or a late-fee warning",
}


def hardneg_prompt(category: str, language: str, channel: str, detail: str, example: str) -> str:
    """P3-hardneg (PROMPTBOOK P3.1), made category-specific: with the generic bank examples
    Gemma wrote bank alerts for every category."""
    hint = _ALARM_HINT.get(category)
    alarm = (
        f"It must contain at least one alarming-sounding phrase (for example {hint}) but must NOT ask the "
        "reader to click a link to log in, share an OTP, pay, or install anything.\n"
        if hint
        else "It must NOT ask the reader to click a link to log in, share an OTP or PIN, pay, or install anything.\n"
    )
    sender = (
        "Use a contact first name as the sender."
        if category == "personal"
        else "Use a registered-style sender header like AX-XXXXXX."
    )
    not_bank = (
        ""
        if category in ("genuine_otp", "transaction_alert")
        else "It must be clearly about this topic and must NOT be a bank debit/credit or transaction alert.\n"
    )
    return (
        f"Write a realistic GENUINE message of type {category} ({CATEGORY_DESC[category]}) that a "
        f"{_ACTOR[category]} would really send, in {LANGUAGE_DESC[language]}.\n"
        f"{alarm}{not_bank}"
        f"{sender} Channel: {channel}. Write any OTP digits as the literal token <OTP> and masked account or "
        "reference numbers as the literal token <ACCT>. Keep it short.\n"
        f"{detail}\n"
        f"Style example (write something different): {example}\n"
        'Return JSON only: {"text": "...", "sender": "..."}'
    )


# --- response parsing and validation --------------------------------------------------------


def parse_json(raw: str) -> dict | None:
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


_URLISH = re.compile(r"\w\.[a-z]{2,}|/", re.IGNORECASE)
_SECRET = re.compile(r"otp|pin|code|password|cvv|ओटीपी|पिन|कोड|पासवर्ड", re.IGNORECASE)


def reason_fits(quote: str, reason: str) -> bool:
    """Cheap sanity check that a quote can carry its reason (the model sometimes tags a
    misspelled word as `lookalike_link`). Only the reasons with a surface form are checked."""
    if reason in ("lookalike_link", "apk_link"):
        return bool(_URLISH.search(quote)) and (reason != "apk_link" or ".apk" in quote.lower())
    if reason == "asks_otp_or_pin":
        return bool(_SECRET.search(quote))
    return True


def clean_flags(obj: dict, text: str, allowed: set[str]) -> list[dict] | None:
    """Validated red flags, or None if any quote is not an exact substring of `text`
    (normalized) or any reason is outside `allowed`. A flag whose quote cannot carry its
    reason (`reason_fits`) is removed; the item needs at least one flag left."""
    raw = obj.get("red_flags")
    if not isinstance(raw, list):
        return None
    flags, seen = [], set()
    for f in raw[:6]:
        if not isinstance(f, dict):
            return None
        q = normalize_text(str(f.get("quote") or ""))
        r = f.get("reason")
        if len(q) < 2 or q not in text or r not in allowed:
            return None
        if (q, r) not in seen and reason_fits(q, r):
            seen.add((q, r))
            flags.append({"quote": q, "reason": r})
    return flags[:4] or None


def relabel_language(requested: str, text: str) -> str:
    """Language actually present in the text, from the Devanagari share of its letters (the
    model often drifts: Devanagari with Latin words, or Latin script when asked for Hindi)."""
    letters = [c for c in text if c.isalpha()]
    share = sum(1 for c in letters if _DEVANAGARI.match(c)) / max(1, len(letters))
    if requested == "hi":
        return "hi" if share >= 0.3 else "hinglish"
    if requested == "hinglish":
        return "hi" if share >= 0.6 else "hinglish"
    return "en" if share < 0.1 else ("hi" if share >= 0.6 else "hinglish")


def ambiguous_ok(norm: str, sender: str | None) -> bool:
    """SUSPICIOUS items must stay ambiguous: no link, no payment/OTP/PIN ask, no rule signal
    beyond an unregistered sender."""
    n = normalize(norm, sender)
    sig = evaluate(n)
    return not n.urls and not sig.hard and not (set(sig.soft) - {"unregistered_sender"})


def accept_response(obj: dict | None, seed: dict, allowed: set[str]) -> dict | None:
    """Validate Gemma's JSON for one job. Returns {text, sender, red_flags} or None."""
    if not obj:
        return None
    text = obj.get("text")
    if not isinstance(text, str):
        return None
    norm = normalize_text(text)
    if not 8 <= len(norm) <= 700 or _PLACEHOLDER.search(norm):
        return None
    sender = obj.get("sender")
    sender = normalize_text(sender) if isinstance(sender, str) else ""
    if seed["verdict"] == "SAFE":
        if seed["category"] != "personal" and sender_status(sender) != "registered":
            return None  # P3-hardneg: registered-style header required
        if evaluate(normalize(norm, sender or None)).hard:
            return None  # a "genuine" item with apk/look-alike/UPI-PIN signals is mislabeled
        return {"text": norm, "sender": sender or "unknown", "red_flags": []}
    flags = clean_flags(obj, norm, allowed)
    if flags is None:
        return None
    if seed["verdict"] == "SUSPICIOUS" and not ambiguous_ok(norm, sender or None):
        return None
    return {"text": norm, "sender": sender or "unknown", "red_flags": flags}


# --- ollama --------------------------------------------------------------------------------


def make_complete(host: str, model: str, client: httpx.AsyncClient) -> Complete:
    """`prompt -> raw JSON string` via Ollama /api/chat (thinking off, JSON mode, temp 0.9)."""

    async def complete(prompt: str) -> str:
        r = await client.post(
            f"{host.rstrip('/')}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "format": "json",
                "options": {"temperature": 0.9, "num_predict": 500},
            },
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    return complete


# --- planning ------------------------------------------------------------------------------


@dataclass
class Job:
    seed: dict
    s: int  # scenario index
    v: int  # 0 = scenario level, >0 = variant index
    language: str
    obfuscation: str
    channel: str
    key: str = field(init=False)

    def __post_init__(self) -> None:
        self.key = f"{self.seed['id']}/s{self.s}" + (f"/v{self.v}" if self.v else "")


def make_job(seed: dict, s: int, v: int) -> Job:
    """Deterministic axes for (seed, scenario, variant) from BUILD_SEED (PRD §10.4)."""
    rng = random.Random(f"{BUILD_SEED}|{seed['id']}|{s}|{v}")
    base = LANGUAGES.index(seed["language"])
    language = LANGUAGES[(base + s + v) % len(LANGUAGES)]
    if seed["verdict"] == "SAFE":
        obf = "none"
        channel = "whatsapp" if seed["category"] == "personal" else rng.choices(CHANNELS, [4, 1])[0]
    else:
        order = list(OBFUSCATIONS)
        random.Random(f"{BUILD_SEED}|{seed['id']}|{s}").shuffle(order)
        obf = "none" if v == 0 else order[(v - 1) % len(order)]
        channel = (
            "call_description" if seed["channel"] == "call_description" else rng.choice(CHANNELS)
        )
    return Job(seed, s, v, language, obf, channel)


def plan_stage(seeds: list[dict], scenarios: int, v: int) -> list[Job]:
    return [make_job(sd, s, v) for s in range(scenarios) for sd in seeds]


def build_prompt(job: Job, source: dict | None) -> str:
    rng = random.Random(f"{BUILD_SEED}|hint|{job.key}")
    seed = job.seed
    detail = hints(rng, scam=seed["verdict"] != "SAFE")
    if seed["verdict"] == "SAFE":
        return hardneg_prompt(seed["category"], job.language, job.channel, detail, seed["text"])
    reasons = sorted({f["reason"] for f in seed["red_flags"]})
    distinct = (
        f"This is scenario {job.s + 1}: use a different pretext or story than the seed.\n"
        if job.v == 0
        else ""
    )
    src = source["text"] if source else seed["text"]
    if seed["verdict"] == "SUSPICIOUS":
        detail += (
            " The message must stay genuinely ambiguous: it must NOT contain any link, threat, "
            "payment request, OTP/PIN request or deadline."
        )
    return gen_prompt(
        src, seed["category"], job.language, job.obfuscation, job.channel, detail, reasons, distinct
    )


def is_hard_negative(category: str, verdict: str, text: str) -> bool:
    return verdict == "SAFE" and (
        category in ("genuine_otp", "transaction_alert") or bool(_ALARM.search(text))
    )


def make_item(job: Job, got: dict) -> dict:
    seed = job.seed
    return {
        "id": f"syn-{seed['id']}-s{job.s}" + (f"-v{job.v}" if job.v else ""),
        "text": got["text"],
        "sender": got["sender"],
        "channel": job.channel,
        "verdict": seed["verdict"],
        "category": seed["category"],
        "red_flags": got["red_flags"],
        "language": relabel_language(job.language, got["text"]),
        "meta": {
            "requested_language": job.language,
            "source": "synthetic",
            "seed_group": seed["id"],
            "scenario": job.s,
            "variant": job.v,
            "key": job.key,
            "obfuscation": job.obfuscation,
            "obfuscated": job.obfuscation != "none",
            "hard_negative": is_hard_negative(seed["category"], seed["verdict"], got["text"]),
            "generator": GENERATOR,
        },
    }


async def run_job(job: Job, source: dict | None, complete: Complete) -> dict | None:
    """One generation with up to MAX_RETRIES retries on bad JSON / failed validation."""
    seed = job.seed
    allowed = {f["reason"] for f in seed["red_flags"]}
    prompt = build_prompt(job, source)
    attempt = conn_errors = 0
    while attempt <= MAX_RETRIES:
        try:
            raw = await complete(prompt)
        except (httpx.TransportError, httpx.HTTPStatusError):
            # server restarting or busy: back off without spending a JSON retry
            conn_errors += 1
            if conn_errors > MAX_CONN_ERRORS:
                return None
            await asyncio.sleep(min(2 * conn_errors, 15))
            continue
        except (KeyError, ValueError):
            attempt += 1
            continue
        attempt += 1
        got = accept_response(parse_json(raw), seed, allowed)
        if got:
            item = finalize_item(make_item(job, got), strict=True)
            if item:
                return item
    return None


# --- post-generation repairs ---------------------------------------------------------------

# Gemma collapses senders to a handful of strings (AX-987654, +91 7890 123456) and drifts away
# from the requested safe category (bank-alert text under `delivery_update`). Both would teach
# shortcuts, so senders are re-drawn deterministically and safe categories are re-derived.
_HEADERS = {
    "bank": ["AX-HDFCBK", "VM-SBIINB", "JD-ICICIB", "AD-AXISBK", "JM-KOTAKB", "VK-PNBSMS",
             "BZ-BOBSMS", "AX-CANBNK", "VM-UBINBK", "JD-IDFCFB", "AD-YESBNK", "VK-INDBNK"],
    "delivery": ["VK-AMAZON", "AX-BLUEDT", "VM-DELHVR", "JD-FKRTCR", "AD-INDPST", "BZ-DTDCIN",
                 "VK-ECOMEX", "AX-MEESHO", "JM-SHDWFX", "AD-XPRSBE"],
    "promo": ["VM-MYNTRA", "AX-JIOINF", "JD-AIRTEL", "VK-ZOMATO", "AD-SWIGGY", "BZ-NYKAAA",
              "VM-PAYTMM", "AX-BIGBSK", "JD-VODAFN", "AD-TATACL"],
    "govt": ["AD-ITDEPT", "VM-EPFOHO", "AX-MAHADS", "JD-UIDAIN", "VK-IRCTCI", "BZ-TNEBLT",
             "AD-DIGLKR", "JM-NPCIIN", "VK-TORRNT", "AX-BESTUN"],
}  # fmt: skip
_HEADER_KIND = {
    "genuine_otp": "bank",
    "transaction_alert": "bank",
    "delivery_update": "delivery",
    "legit_promo": "promo",
    "govt_genuine": "govt",
}
_SAFE_FIT = {
    "genuine_otp": re.compile(r"<OTP>"),
    "delivery_update": re.compile(
        r"deliver|parcel|order|shipment|courier|package|dispatch|out for|track|डिलीवरी|पार्सल|ऑर्डर|शिपमेंट|कूरियर",
        re.IGNORECASE,
    ),
    "govt_genuine": re.compile(
        r"income tax|itr|epfo|\bpf\b|uidai|aadhaar|electricity|bijli|bill|ration|pension|passport|"
        r"msedcl|govt|government|आयकर|बिजली|बिल|आधार|पेंशन|सरकार|राशन|पासपोर्ट",
        re.IGNORECASE,
    ),
    "legit_promo": re.compile(
        r"offer|sale|cashback|discount|% off|recharge|plan|coupon|reward|ऑफर|छूट|कैशबैक|रिचार्ज|सेल",
        re.IGNORECASE,
    ),
    "transaction_alert": re.compile(
        r"debit|credit|a/c|account|kyc|transaction|transfer|खाते|खाता|डेबिट|क्रेडिट|केवाईसी|लेनदेन",
        re.IGNORECASE,
    ),
}
_SAFE_ORDER = ("genuine_otp", "delivery_update", "govt_genuine", "legit_promo", "transaction_alert")


def safe_category_for(declared: str, text: str, sender: str) -> str | None:
    """The declared safe category if the text fits it, else the first fitting one, else None."""
    if declared == "personal":
        return declared if sender_status(sender) != "registered" and "<OTP>" not in text else None
    if _SAFE_FIT[declared].search(text):
        return declared
    return next((c for c in _SAFE_ORDER if _SAFE_FIT[c].search(text)), None)


def finalize_item(item: dict, *, strict: bool = False) -> dict | None:
    """Deterministic sender re-draw (all verdicts) and safe-category re-derivation; None if a
    SAFE text fits no safe category (with `strict`, also when it fits only another category)."""
    rng = random.Random(f"{BUILD_SEED}|sender|{item['meta']['key']}")
    if item["verdict"] == "SAFE":
        cat = safe_category_for(item["category"], item["text"], item["sender"])
        if cat is None or (strict and cat != item["category"]):
            return None
        if cat != item["category"]:
            item["meta"]["category_requested"] = item["category"]
            item["category"] = cat
        item["meta"]["hard_negative"] = is_hard_negative(cat, "SAFE", item["text"])
        if cat != "personal":
            item["sender"] = rng.choice(_HEADERS[_HEADER_KIND[cat]])
        elif sender_status(item["sender"]) == "registered":
            item["sender"] = rng.choice(_NAMES)
    elif sender_status(item["sender"]) != "registered":
        if rng.random() < 0.2:  # spoofed DLT-style header, as in real scam SMS
            item["sender"] = (
                "".join(rng.choices("ABCDEFGHIJKLMNOPRSTUVW", k=2))
                + "-"
                + "".join(rng.choices("ABCDEFGHIKLMNOPRSTUVY", k=6))
            )
        elif sender_status(item["sender"]) == "unregistered" or item["sender"] == "unknown":
            item["sender"] = (
                f"+91 {rng.choice('6789')}{rng.randrange(10**4):04d} {rng.randrange(10**5):05d}"
            )
    return item


def redraw_call_sender(item: dict) -> dict:
    """Gemma writes one caller number for every call (+91 98765 43210); draw varied ones:
    mostly mobile numbers, some toll-free for institutional callbacks, some withheld."""
    rng = random.Random(f"{BUILD_SEED}|callsender|{item['meta']['key']}")
    r = rng.random()
    institutional = item["category"] in ("transaction_alert", "govt_genuine", "legit_promo")
    if institutional and item["verdict"] == "SAFE" and r < 0.4:
        item["sender"] = f"1800 {rng.randrange(100, 999)} {rng.randrange(1000, 9999)}"
    elif r < 0.12:
        item["sender"] = "unknown"
    else:
        item["sender"] = (
            f"+91 {rng.choice('6789')}{rng.randrange(10**4):04d} {rng.randrange(10**5):05d}"
        )
    return item


def repair_file(path: Path) -> dict[str, int]:
    """Apply `finalize_item` to an existing batch file in place (idempotent)."""
    items = read_items(path)
    out = [
        redraw_call_sender(i)
        if i["meta"]["key"].startswith(CALL_GROUP_PREFIX)
        else finalize_item(i)
        for i in items
    ]
    out = [x for x in out if x is not None]
    path.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in out), "utf-8")
    return {"before": len(items), "after": len(out)}


# --- call descriptions without seeds (genuine / ambiguous calls) -----------------------------

# Scam seeds were the only call_description items, so "channel = call_description" would
# separate verdicts. These specs drive P3-hardneg-style prompts with no seed text (PRD §17).
# (id, verdict, category, scenario, reasons for SUSPICIOUS flags)
CALL_SPECS = [
    ("c1", "SAFE", "transaction_alert",
     "the bank's genuine customer care called back about a debit card the person had earlier asked "
     "to block; the caller only confirmed the card is blocked and a new one will be posted, and said "
     "the bank never asks for OTP, PIN or CVV on a call", ()),
    ("c2", "SAFE", "delivery_update",
     "a delivery agent called to ask for a landmark or directions to the house and said he will "
     "arrive in about ten minutes; he asked for nothing else", ()),
    ("c3", "SAFE", "govt_genuine",
     "the electricity board's recorded call announced a scheduled maintenance power cut tomorrow "
     "between 10 AM and 1 PM in the area; nothing was asked", ()),
    ("c4", "SAFE", "personal",
     "a relative called from a new number because their phone is being repaired, said they reached "
     "home safely and just wanted to chat; no money, code or favour asked", ()),
    ("c5", "SAFE", "legit_promo",
     "a telecom operator's customer care called to mention a new recharge plan and asked whether the "
     "person would like to hear details later; nothing else asked", ()),
    ("c6", "SAFE", "govt_genuine",
     "the gas agency called to say the cylinder delivery is scheduled for tomorrow morning and to "
     "keep the old empty cylinder ready; no payment or code asked on the call", ()),
    ("c7", "SAFE", "transaction_alert",
     "bank branch staff called to remind that the account KYC is due and asked the person to visit "
     "the branch with ID proof this month; they said not to share any details on the phone", ()),
    ("c8", "SUSPICIOUS", "other_scam",
     "an unknown caller asked whether this is the account holder's number, said he will call back "
     "later and hung up; he asked for nothing", ("impersonates_authority", "urgency_deadline")),
    ("c9", "SUSPICIOUS", "other_scam",
     "a caller said he is from an insurance company and mentioned a special scheme, asking the person "
     "to call back on a number if interested; no payment or code asked", ("impersonates_authority", "too_good_to_be_true")),
    ("c10", "SUSPICIOUS", "other_scam",
     "a caller said he is doing a survey for the bank and wanted a good time to call again; no "
     "payment, code or link asked", ("impersonates_authority", "urgency_deadline")),
]  # fmt: skip
CALL_GROUP_PREFIX = "calls-"


def call_prompt(spec: tuple, language: str, detail: str) -> str:
    _, verdict, _, scenario, reasons = spec
    flags = (
        ', "red_flags": [{"quote": "<exact substring of text>", "reason": "<one of: '
        + ", ".join(reasons)
        + '>"}]'
        if verdict == "SUSPICIOUS"
        else ""
    )
    return (
        "You are helping build a dataset to protect elderly people in India from scams.\n"
        "Write a realistic CALL DESCRIPTION: 1-3 short sentences by the person who took a phone call, "
        "summarising what the caller said, in the past tense (for example 'Caller said ...').\n"
        f"Language/style: {LANGUAGE_DESC[language]}\n"
        f"What happened: {scenario}.\n"
        "The call must NOT ask the person to share an OTP or PIN, pay money, click a link or install an app.\n"
        f"{detail}\n"
        'Return JSON only: {"text": "<description>", "sender": "<caller number such as +91 98xxx xxxxx '
        f'or a toll-free number like 1800 xxx xxxx>"{flags}'
        "}"
    )


def accept_call_response(obj: dict | None, spec: tuple) -> dict | None:
    if not obj or not isinstance(obj.get("text"), str):
        return None
    _, verdict, _, _, reasons = spec
    norm = normalize_text(obj["text"])
    if not 8 <= len(norm) <= 500 or _PLACEHOLDER.search(norm):
        return None
    sender = obj.get("sender")
    sender = normalize_text(sender) if isinstance(sender, str) else ""
    if not sender or sender_status(sender) == "registered":
        sender = "unknown"
    n = normalize(norm, None if sender == "unknown" else sender)
    sig = evaluate(n)
    if sig.hard or n.urls or (set(sig.soft) - {"unregistered_sender"}):
        return None  # a genuine or ambiguous call must stay free of ask/link signals
    flags: list[dict] = []
    if verdict == "SUSPICIOUS":
        flags = clean_flags(obj, norm, set(reasons)) or []
        if not flags:
            return None
    return {"text": norm, "sender": sender, "red_flags": flags}


async def generate_calls(
    complete: Complete,
    out_path: Path,
    *,
    per_spec: int = 8,
    concurrency: int = 4,
    log: Callable[[str], None] = print,
) -> dict[str, int]:
    """One batch of seedless call descriptions; resumable by key, like `generate`."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {it["meta"]["key"] for it in read_items(out_path)}
    sem, stats = asyncio.Semaphore(concurrency), Counter()

    async def one(spec: tuple, n: int, fh) -> None:
        sid, verdict, category = spec[0], spec[1], spec[2]
        key = f"{CALL_GROUP_PREFIX}{sid}/{n}"
        if key in done:
            stats["resumed"] += 1
            return
        rng = random.Random(f"{BUILD_SEED}|{key}")
        language = LANGUAGES[(n + int(sid[1:])) % 3]
        prompt = call_prompt(spec, language, hints(rng, scam=False))
        async with sem:
            got = None
            for _ in range(MAX_RETRIES + 1):
                try:
                    raw = await complete(prompt)
                except (httpx.TransportError, httpx.HTTPStatusError, KeyError, ValueError):
                    await asyncio.sleep(2)
                    continue
                got = accept_call_response(parse_json(raw), spec)
                if got:
                    break
        if not got:
            stats["dropped"] += 1
            return
        item = {
            "id": f"syn-{CALL_GROUP_PREFIX}{sid}-{n}",
            "text": got["text"],
            "sender": got["sender"],
            "channel": "call_description",
            "verdict": verdict,
            "category": category,
            "red_flags": got["red_flags"],
            "language": relabel_language(language, got["text"]),
            "meta": {
                "requested_language": language,
                "source": "synthetic",
                "seed_group": f"{CALL_GROUP_PREFIX}{sid}",
                "scenario": n,
                "variant": 0,
                "key": key,
                "obfuscation": "none",
                "obfuscated": False,
                "hard_negative": is_hard_negative(category, verdict, got["text"]),
                "generator": GENERATOR,
            },
        }
        fh.write(json.dumps(redraw_call_sender(item), ensure_ascii=False) + "\n")
        fh.flush()
        stats["generated"] += 1

    with out_path.open("a", encoding="utf-8") as fh:
        await asyncio.gather(*(one(sp, n, fh) for n in range(per_spec) for sp in CALL_SPECS))
    log(f"calls: {len(done) + stats['generated']} items total")
    return dict(stats)


# --- genuine messages with real official links (no seed text) ---------------------------------

# Without these, "contains a link" separates SCAM from SAFE perfectly. Each group fixes a few
# domains from hub/data/official_domains.txt; every URL in an accepted item must be official.
LINK_DOMAINS = {
    "transaction_alert": ["hdfcbank.com", "icicibank.com", "axisbank.com", "onlinesbi.sbi", "kotak.com", "sbi.co.in"],
    "genuine_otp": ["hdfcbank.com", "icicibank.com", "axisbank.com", "onlinesbi.sbi", "paytm.com", "phonepe.com"],
    "delivery_update": ["amazon.in", "indiapost.gov.in", "amazon.in", "indiapost.gov.in"],
    "legit_promo": ["amazon.in", "phonepe.com", "paytm.com", "google.com", "cred.club", "mobikwik.com"],
    "govt_genuine": ["incometax.gov.in", "uidai.gov.in", "epfindia.gov.in", "mahadiscom.in", "tatapower.com",
                     "adanielectricity.com", "bestundertaking.com", "torrentpower.com", "passportindia.gov.in",
                     "digilocker.gov.in"],
}  # fmt: skip
LINK_PATHS = {
    "transaction_alert": ["support", "security", "branch-locator", "kyc-info"],
    "genuine_otp": ["security-tips", "fraud-alert", "safe-banking"],
    "delivery_update": ["track", "your-orders", "tracking"],
    "legit_promo": ["offers", "sale", "deals"],
    "govt_genuine": ["services", "status", "help", "notices"],
}
LINK_GROUPS_PER_CATEGORY = 4
LINK_PREFIX = "links-"


def link_for(category: str, group: int, n: int) -> str:
    rng = random.Random(f"{BUILD_SEED}|link|{category}|{group}|{n}")
    domains = LINK_DOMAINS[category]
    d = domains[(group * 3 + n) % len(domains)]
    path = rng.choice(LINK_PATHS[category])
    host = d if d.endswith((".gov.in", ".sbi")) and rng.random() < 0.5 else f"www.{d}"
    return rng.choice([f"https://{host}/{path}", f"https://{host}/{path}", f"{host}/{path}"])


def links_prompt(category: str, language: str, channel: str, detail: str, link: str) -> str:
    base = hardneg_prompt(category, language, channel, detail, "(none)")
    base = base.replace("Style example (write something different): (none)\n", "")
    base = base.replace('Return JSON only: {"text": "...", "sender": "..."}', "").rstrip()
    return (
        f"{base}\nThe message must also contain exactly this official link, written exactly like this: {link}\n"
        "The link is only for information (for example support, tracking, security tips or details); the "
        "message must not ask the reader to log in, enter details, pay or install anything through it, and "
        "must not contain any other link.\n"
        'Return JSON only: {"text": "...", "sender": "..."}'
    )


def accept_link_response(obj: dict | None, category: str, link: str) -> dict | None:
    """Genuine item with at least one URL, every URL official, no hard rule signal and no
    OTP-share / urgency-plus-payment signal."""
    got = accept_response(obj, {"verdict": "SAFE", "category": category}, set())
    if got is None:
        return None
    n = normalize(got["text"], got["sender"])
    data = load_rule_data()
    hosts = [_host(u) for u in n.urls]
    if not hosts or any(h is None or not _is_allowed(h, data) for h in hosts):
        return None
    if link.split("://")[-1].rstrip("/") not in got["text"]:
        return None
    if set(evaluate(n).soft) & {"asks_otp_or_pin", "urgency_plus_payment", "shortener_link"}:
        return None
    return got


async def generate_links(
    complete: Complete,
    out_path: Path,
    *,
    per_group: int = 8,
    concurrency: int = 4,
    deadline: float | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, int]:
    """Batch of genuine items with official links, `LINK_GROUPS_PER_CATEGORY` seed groups of
    `per_group` items per safe category. Resumable by key."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {it["meta"]["key"] for it in read_items(out_path)}
    sem, stats = asyncio.Semaphore(concurrency), Counter()

    async def one(category: str, k: int, n: int, fh) -> None:
        group = f"{LINK_PREFIX}{category}-{k}"
        key = f"{group}/{n}"
        if key in done:
            stats["resumed"] += 1
            return
        if deadline is not None and time.time() > deadline:
            stats["skipped_deadline"] += 1
            return
        rng = random.Random(f"{BUILD_SEED}|{key}")
        language = LANGUAGES[(n + k) % 3]
        channel = rng.choices(CHANNELS, [3, 1])[0]
        link = link_for(category, k, n)
        prompt = links_prompt(category, language, channel, hints(rng, scam=False), link)
        got = None
        async with sem:
            for _ in range(MAX_RETRIES + 1):
                try:
                    raw = await complete(prompt)
                except (httpx.TransportError, httpx.HTTPStatusError, KeyError, ValueError):
                    await asyncio.sleep(2)
                    continue
                got = accept_link_response(parse_json(raw), category, link)
                if got:
                    break
        item = (
            finalize_item(
                {
                    "id": f"syn-{group}-{n}",
                    "text": got["text"],
                    "sender": got["sender"],
                    "channel": channel,
                    "verdict": "SAFE",
                    "category": category,
                    "red_flags": [],
                    "language": relabel_language(language, got["text"]),
                    "meta": {
                        "requested_language": language,
                        "source": "synthetic",
                        "seed_group": group,
                        "scenario": n,
                        "variant": 0,
                        "key": key,
                        "obfuscation": "none",
                        "obfuscated": False,
                        "hard_negative": True,
                        "generator": GENERATOR,
                    },
                },
                strict=True,
            )
            if got
            else None
        )
        if item is None:
            stats["dropped"] += 1
            return
        fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        fh.flush()
        stats["generated"] += 1

    with out_path.open("a", encoding="utf-8") as fh:
        await asyncio.gather(
            *(
                one(c, k, n, fh)
                for n in range(per_group)
                for k in range(LINK_GROUPS_PER_CATEGORY)
                for c in LINK_DOMAINS
            )
        )
    log(f"links: {len(done) + stats['generated']} items total")
    return dict(stats)


# --- sanity pass over the review sample -------------------------------------------------------


def suspect_reasons(item: dict) -> list[str]:
    """Automated label-noise heuristics (no model): a hit means "look at this one by hand"."""
    text, out = item["text"], []
    n = normalize(text, item["sender"] if item["sender"] != "unknown" else None)
    sig = evaluate(n)
    if item["verdict"] == "SAFE":
        data = load_rule_data()
        if "asks_otp_or_pin" in sig.soft:
            out.append("safe_otp_ask")
        if "urgency_plus_payment" in sig.soft:
            out.append("safe_urgency_payment")
        for u in extract_urls(n.text):
            h = _host(u)
            if h and not _is_allowed(h, data):
                out.append("safe_nonofficial_link")
                break
    if _PLACEHOLDER.search(text):
        out.append("placeholder")
    has_dev = bool(_DEVANAGARI.search(text))
    if item["language"] == "hi" and not has_dev:
        out.append("lang_hi_without_devanagari")
    if item["language"] == "en" and has_dev:
        out.append("lang_en_with_devanagari")
    return out


def review_sample(items: list[dict]) -> list[dict]:
    k = max(1, round(len(items) * REVIEW_FRACTION)) if items else 0
    return random.Random(BUILD_SEED).sample(items, k)


# --- orchestration ---------------------------------------------------------------------------


def read_items(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


async def generate(
    seeds: list[dict],
    complete: Complete,
    out_path: Path,
    *,
    scenarios: int = 5,
    variants: int = 5,
    concurrency: int = 4,
    deadline: float | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, int]:
    """Run all stages (scenario level, then variant j = 1..variants), appending each accepted
    item to `out_path`. Keys already present are skipped, so reruns resume."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_key = {it["meta"]["key"]: it for it in read_items(out_path)}
    sem = asyncio.Semaphore(concurrency)
    stats = Counter()

    async def one(job: Job, fh) -> None:
        if deadline is not None and time.time() > deadline:
            stats["skipped_deadline"] += 1
            return
        source = None
        if job.v and job.seed["verdict"] != "SAFE":
            source = by_key.get(f"{job.seed['id']}/s{job.s}")
            if source is None:
                stats["no_scenario"] += 1
                return
        async with sem:
            if deadline is not None and time.time() > deadline:
                stats["skipped_deadline"] += 1
                return
            item = await run_job(job, source, complete)
        if item is None:
            stats["dropped"] += 1
            return
        by_key[job.key] = item
        fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        fh.flush()
        stats["generated"] += 1

    with out_path.open("a", encoding="utf-8") as fh:
        for v in range(variants + 1):
            jobs = [j for j in plan_stage(seeds, scenarios, v) if j.key not in by_key]
            stats["resumed"] += len(plan_stage(seeds, scenarios, v)) - len(jobs)
            await asyncio.gather(*(one(j, fh) for j in jobs))
            log(f"stage {v}/{variants}: {len(by_key)} items total")
    return dict(stats)


def counts(items: list[dict]) -> dict[str, dict[str, int]]:
    def c(f):
        return dict(sorted(Counter(str(f(i)) for i in items).items()))

    return {
        "verdict": c(lambda i: i["verdict"]),
        "category": c(lambda i: i["category"]),
        "language": c(lambda i: i["language"]),
        "obfuscated": c(lambda i: i["meta"]["obfuscated"]),
        "hard_negative": c(lambda i: i["meta"]["hard_negative"]),
    }


def write_review(items: list[dict], path: Path) -> list[dict]:
    sample = review_sample(items)
    path.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in sample), "utf-8")
    return sample


def parse_until(hhmm: str | None) -> float | None:
    if not hhmm:
        return None
    h, m = (int(x) for x in hhmm.split(":"))
    now = datetime.now()
    return now.replace(hour=h, minute=m, second=0, microsecond=0).timestamp()


async def amain(args: argparse.Namespace) -> int:
    from hub.settings import get_settings

    cfg = get_settings()
    if args.links:
        out_path = args.out / f"{args.batch_id}.jsonl"
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
            complete = make_complete(cfg.ollama_host, cfg.gemma_model, client)
            stats = await generate_links(
                complete,
                out_path,
                per_group=args.per_spec,
                concurrency=args.concurrency,
                deadline=parse_until(args.until),
            )
        items = read_items(out_path)
        write_review(items, args.out / f"review_{args.batch_id}.jsonl")
        print(f"links items={len(items)} run={stats}")
        for key, c in counts(items).items():
            print(f"  {key:<14} " + "  ".join(f"{k}={v}" for k, v in c.items()))
        return 0
    if args.calls:
        out_path = args.out / f"{args.batch_id}.jsonl"
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
            complete = make_complete(cfg.ollama_host, cfg.gemma_model, client)
            stats = await generate_calls(
                complete, out_path, per_spec=args.per_spec, concurrency=args.concurrency
            )
        items = read_items(out_path)
        write_review(items, args.out / f"review_{args.batch_id}.jsonl")
        print(f"calls items={len(items)} run={stats}")
        for key, c in counts(items).items():
            print(f"  {key:<14} " + "  ".join(f"{k}={v}" for k, v in c.items()))
        return 0
    seeds = load_seeds(args.seeds)
    if args.only_categories:
        keep = set(args.only_categories.split(","))
        seeds = [x for x in seeds if x["category"] in keep]
    out_path = args.out / f"{args.batch_id}.jsonl"
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        complete = make_complete(cfg.ollama_host, cfg.gemma_model, client)
        stats = await generate(
            seeds,
            complete,
            out_path,
            scenarios=args.scenarios,
            variants=args.variants,
            concurrency=args.concurrency,
            deadline=parse_until(args.until),
        )
    items = read_items(out_path)
    sample = write_review(items, args.out / f"review_{args.batch_id}.jsonl")
    sus = Counter(r for it in sample for r in suspect_reasons(it))
    flagged = sum(1 for it in sample if suspect_reasons(it))
    print(f"seeds={len(seeds)} items={len(items)} run={stats}")
    for key, c in counts(items).items():
        print(f"  {key:<14} " + "  ".join(f"{k}={v}" for k, v in c.items()))
    print(
        f"review sample {len(sample)}: suspect {flagged} ({flagged / max(1, len(sample)):.1%}) {dict(sus)}"
    )
    print(f"wrote {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batch-id", default="batch1")
    ap.add_argument("--seeds", type=Path, default=Path("data/seeds/seeds.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--scenarios", type=int, default=5)
    ap.add_argument("--variants", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument(
        "--calls", action="store_true", help="generate the seedless call-description batch"
    )
    ap.add_argument(
        "--links", action="store_true", help="generate the genuine-with-official-links batch"
    )
    ap.add_argument("--per-spec", type=int, default=8)
    ap.add_argument("--only-categories", help="comma-separated seed categories to (re)generate")
    ap.add_argument("--until", help="stop starting new generations at local HH:MM")
    ap.add_argument(
        "--repair",
        action="store_true",
        help="re-apply sender/category repairs to the batch file and exit",
    )
    args = ap.parse_args(argv)
    if args.repair:
        print(repair_file(args.out / f"{args.batch_id}.jsonl"))
        return 0
    try:
        return asyncio.run(amain(args))
    except SynthError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
