"""Normalize node: NFKC, zero-width strip, whitespace, URL/phone/UPI/sender extraction (PRD FR-10).

Pure functions, no I/O. `normalize_text` and `sender_status` are shared with the dataset
builders so that training, eval and inference see identical detector inputs (PRD §8.2).
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

import tldextract

SenderStatus = Literal["registered", "unregistered", "unknown"]

# Zero-width, soft-hyphen and bidi control characters used to break up scam keywords.
_ZW_CODEPOINTS = [
    0x200B,
    0x200C,
    0x200D,
    0x2060,
    0xFEFF,
    0x00AD,
    *range(0x202A, 0x202F),
    *range(0x2066, 0x206A),
]
_ZERO_WIDTH = re.compile("[" + "".join(chr(c) for c in _ZW_CODEPOINTS) + "]")
_HSPACE = re.compile(r"[^\S\n]+")

# DLT-registered alphanumeric headers: AX-HDFCBK, VK-SBIUPD, JD-AIRTEL-S.
_REGISTERED = re.compile(r"^[A-Z]{2}-[A-Z0-9]{3,9}(?:-[A-Z])?$")
_NUMBER = re.compile(r"^\+?[\d\s\-()]*(?:<PHONE>)?[\d\s\-()]*$")

# Perception output for screenshots (PRD Appendix A.2): "SENDER: x\nMESSAGE:\n<text>".
_PERCEPTION_HEADER = re.compile(
    r"\ASENDER:[ \t]*(?P<sender>[^\n]*)\nMESSAGE:[ \t]*\n?", re.IGNORECASE
)

_TRAIL = ".,;:!?)]}'\"’”>"
_URL_SCHEME = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.IGNORECASE)
# Bare domains (Unicode labels allowed, for IDN look-alikes); validated with tldextract.
_URL_BARE = re.compile(r"(?<![@\w.\-/])(?:[\w-]+\.)+[\w-]{2,}(?![\w.\-]*@)(?:/[^\s<>\"']*)?")
# Phones: "+cc ..." (also redacted "+92 301 2<PHONE>"), or Indian 10-digit mobiles.
_PHONE_PLUS = re.compile(
    r"(?<![\w+])(?:\+|00)\d{1,3}(?:[\s-]?\d){1,12}(?:<PHONE>)?|(?<![\w+])(?:\+|00)\d{1,3}[\s-]?<PHONE>"
)
_PHONE_IN = re.compile(r"(?<![\w+])0?[6-9](?:[\s-]?\d){9}(?!\d)|(?<![\w+])[6-9]\d{3}<PHONE>")
_UPI = re.compile(r"[\w.\-]+@[a-zA-Z]{2,}")  # FR-10

# Offline: use the bundled public-suffix snapshot, never fetch at runtime.
_TLD = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)


@dataclass(frozen=True)
class NormalizedInput:
    """Output of the normalize node (PRD FR-10; fields map onto CheckState §8.3)."""

    text: str
    sender: str | None
    sender_status: SenderStatus
    urls: list[str] = field(default_factory=list)
    phones: list[str] = field(default_factory=list)
    upi_handles: list[str] = field(default_factory=list)


def normalize_text(text: str) -> str:
    """NFKC, strip zero-width chars, collapse whitespace (runs of spaces → one space; trim
    lines; drop blank lines). Line breaks between non-empty lines are kept."""
    text = _ZERO_WIDTH.sub("", unicodedata.normalize("NFKC", text))
    lines = (_HSPACE.sub(" ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def sender_status(sender: str | None) -> SenderStatus:
    """`registered` for a DLT header, `unregistered` for a phone number, else `unknown`."""
    s = (sender or "").strip()
    if not s:
        return "unknown"
    if _REGISTERED.match(s.upper()):
        return "registered"
    if any(c.isdigit() for c in s) and _NUMBER.match(s):
        return "unregistered"
    return "unknown"


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def extract_urls(text: str) -> list[str]:
    """Links with a scheme or `www.`, plus bare domains whose suffix is a real TLD.
    Each result is an exact substring of `text`."""
    urls, taken = [], []
    for m in _URL_SCHEME.finditer(text):
        urls.append(m.group(0).rstrip(_TRAIL))
        taken.append(m.span())
    for m in _URL_BARE.finditer(text):
        if any(s <= m.start() < e for s, e in taken):
            continue
        cand = m.group(0).rstrip(_TRAIL)
        ext = _TLD(cand.split("/", 1)[0])
        if ext.domain and ext.suffix:
            urls.append(cand)
    return _dedupe(urls)


def extract_phones(text: str) -> list[str]:
    phones, taken = [], []
    for m in _PHONE_PLUS.finditer(text):
        phones.append(m.group(0).strip(" -"))
        taken.append(m.span())
    for m in _PHONE_IN.finditer(text):
        if not any(s <= m.start() < e for s, e in taken):
            phones.append(m.group(0))
    return _dedupe(phones)


def extract_upi_handles(text: str) -> list[str]:
    return _dedupe(m.group(0) for m in _UPI.finditer(text))


def normalize(text: str, sender: str | None = None) -> NormalizedInput:
    """FR-10. If `sender` is missing and the text starts with the A.2 perception header
    (`SENDER: …` / `MESSAGE:`), the sender is taken from it and the header is removed."""
    clean = normalize_text(text)
    m = _PERCEPTION_HEADER.match(clean)
    if m:
        header_sender = m.group("sender").strip()
        if not sender and header_sender.lower() != "unknown":
            sender = header_sender
        clean = clean[m.end() :]
    sender = normalize_text(sender) or None if sender else None
    return NormalizedInput(
        text=clean,
        sender=sender,
        sender_status=sender_status(sender),
        urls=extract_urls(clean),
        phones=extract_phones(clean),
        upi_handles=extract_upi_handles(clean),
    )
