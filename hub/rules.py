"""Rules engine: hard and soft signals as pure functions, no I/O (PRD FR-12, §7.3).

    evaluate(normalize(text, sender)) -> RuleSignals(hard, soft, evidence)

Word lists and domain lists live in `hub/data/` and are loaded once by `load_rule_data`
(cached); `evaluate` itself only reads its arguments. Every evidence string is an exact
substring of the normalized text, so fusion can show it as a red flag (PRD FR-14).

`rule_signals(text, sender, channel)` is the flat-list contract used by the dataset
builders (training/freeze_test.py) for the §8.2 RULE_SIGNALS line.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from rapidfuzz.distance import Levenshtein

from hub.normalize import _TLD, NormalizedInput, normalize, normalize_text

DATA_DIR = Path(__file__).parent / "data"

HARD = ("apk_link", "lookalike_domain", "upi_pin_to_receive")
SOFT = (
    "unregistered_sender",
    "shortener_link",
    "threat_lexicon",
    "urgency_plus_payment",
    "foreign_code_authority",
    "asks_otp_or_pin",
)
SECTIONS = (
    "threat", "urgency", "payment", "bank", "govt", "otp",
    "share", "negation", "install", "pin", "approve", "receive",
)  # fmt: skip

# Letters including Devanagari combining marks (matras are not \w in Python).
_W = r"[\wऀ-ॿ]"
_SENTENCE_BREAK = re.compile(r"[.!?।]+(?=\s|$)|\n")
_CLAUSE_BREAK = re.compile(
    r"[.!?।]+(?=\s|$)|\n|[,;]|\b(?:but|however|lekin|magar)\b|लेकिन|मगर|परंतु", re.IGNORECASE
)
_APK_TOKEN = re.compile(r"[\w./\-]+\.apk\b", re.IGNORECASE)
# "na" / "न" at the end of a clause is a softener ("bhej do na"), not a negation.
_SOFT_NEGATIONS = {"na", "न"}


@dataclass(frozen=True)
class RuleSignals:
    hard: list[str] = field(default_factory=list)
    soft: list[str] = field(default_factory=list)
    evidence: dict[str, list[str]] = field(default_factory=dict)

    def as_state(self) -> dict:
        """CheckState.rule_signals shape (PRD §8.3)."""
        return {"hard": list(self.hard), "soft": list(self.soft)}


@dataclass(frozen=True)
class RuleData:
    official: tuple[str, ...]  # registrable domains; host matches itself or subdomains
    restricted: tuple[str, ...]  # "*.x" zones: allowed, never fuzzy-compared
    brands: tuple[str, ...]
    shorteners: frozenset[str]
    lexicon: dict[str, re.Pattern]


# --- data loading (the only I/O, done once) ---


def _lines(path: Path) -> list[str]:
    out = []
    for raw in path.read_text("utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def _phrase_regex(entry: str) -> str:
    entry = normalize_text(entry)
    prefix = entry.endswith("*")
    words = entry.rstrip("*").split()
    body = r"\s+".join(re.escape(w) for w in words)
    return body + (f"{_W}*" if prefix else "")


def compile_lexicon(sections: dict[str, list[str]]) -> dict[str, re.Pattern]:
    out = {}
    for name in SECTIONS:
        entries = sorted(set(sections.get(name, [])), key=len, reverse=True)
        if not entries:
            out[name] = re.compile(r"(?!x)x")  # matches nothing
            continue
        alt = "|".join(_phrase_regex(e) for e in entries)
        out[name] = re.compile(rf"(?<!{_W})(?:{alt})(?!{_W})", re.IGNORECASE)
    return out


def parse_lexicon(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            sections.setdefault(current, [])
        elif current:
            sections[current].append(line)
    return sections


@lru_cache
def load_rule_data(data_dir: Path = DATA_DIR) -> RuleData:
    domains = [d.lower() for d in _lines(data_dir / "official_domains.txt")]
    merged: dict[str, list[str]] = {}
    for path in sorted((data_dir / "lexicon").glob("*.txt")):
        for name, entries in parse_lexicon(path.read_text("utf-8")).items():
            merged.setdefault(name, []).extend(entries)
    return RuleData(
        official=tuple(d for d in domains if not d.startswith("*.")),
        restricted=tuple(d[2:] for d in domains if d.startswith("*.")),
        brands=tuple(b.lower() for b in _lines(data_dir / "brand_tokens.txt")),
        shorteners=frozenset(s.lower() for s in _lines(data_dir / "shorteners.txt")),
        lexicon=compile_lexicon(merged),
    )


# --- helpers ---


def _spans(text: str, breaker: re.Pattern) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in breaker.finditer(text):
        if m.start() > start:
            spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _find(
    pat: re.Pattern, text: str, start: int = 0, end: int | None = None
) -> list[tuple[int, int]]:
    end = len(text) if end is None else end
    return [(start + m.start(), start + m.end()) for m in pat.finditer(text[start:end])]


def _negated(text: str, span: tuple[int, int], data: RuleData) -> bool:
    """Is there a negation in the clause containing `span`?"""
    for cs, ce in _spans(text, _CLAUSE_BREAK):
        if cs <= span[0] < ce:
            for ns, ne in _find(data.lexicon["negation"], text, cs, ce):
                word = text[ns:ne].lower()
                if word in _SOFT_NEGATIONS and not re.search(_W, text[ne:ce]):
                    continue  # clause-final softener
                return True
            return False
    return False


def _host(url: str) -> str | None:
    try:
        host = urlsplit(url if "://" in url else f"http://{url}").hostname
    except ValueError:
        return None
    return host.rstrip(".").lower() if host else None


def _registrable(host: str) -> str:
    ext = _TLD(host)
    return f"{ext.domain}.{ext.suffix}" if ext.domain and ext.suffix else host


def _is_allowed(host: str, data: RuleData) -> bool:
    return any(host == d or host.endswith("." + d) for d in data.official + data.restricted)


def _is_idn(host: str) -> bool:
    return not host.isascii() or any(label.startswith("xn--") for label in host.split("."))


def _near_official(registrable: str, data: RuleData) -> bool:
    for off in data.official:
        max_d = 1 if len(off.split(".")[0]) <= 4 else 2
        if 0 < Levenshtein.distance(registrable, off, score_cutoff=max_d) <= max_d:
            return True
    return False


def _has_brand(host: str, data: RuleData) -> bool:
    labels = [p for p in re.split(r"[.\-]", host) if p]
    return any(p == t or p.startswith(t) or p.endswith(t) for p in labels for t in data.brands)


def _is_foreign(phone: str) -> bool:
    p = phone.replace(" ", "").replace("-", "")
    return (p.startswith("+") and not p.startswith("+91")) or (
        p.startswith("00") and not p.startswith("0091")
    )


# --- signals (§7.3) ---


def apk_link(n: NormalizedInput, data: RuleData) -> list[str]:
    ev = []
    for url in n.urls:
        path = urlsplit(url if "://" in url else f"http://{url}").path
        if path.lower().endswith(".apk"):
            ev.append(url)
    ev += [m.group(0) for m in _APK_TOKEN.finditer(n.text)]
    install = _find(data.lexicon["install"], n.text)
    foreign_links = [u for u in n.urls if (h := _host(u)) and not _is_allowed(h, data)]
    if install and foreign_links:
        ev += [n.text[s:e] for s, e in install] + foreign_links
    return ev


def lookalike_domain(n: NormalizedInput, data: RuleData) -> list[str]:
    ev = []
    for url in n.urls:
        host = _host(url)
        if not host or _is_allowed(host, data):
            continue
        if _is_idn(host) or _near_official(_registrable(host), data) or _has_brand(host, data):
            ev.append(url)
    return ev


def upi_pin_to_receive(n: NormalizedInput, data: RuleData) -> list[str]:
    ev, t = [], n.text
    for ss, se in _spans(t, _SENTENCE_BREAK):
        asks = _find(data.lexicon["pin"], t, ss, se) + _find(data.lexicon["approve"], t, ss, se)
        receive = _find(data.lexicon["receive"], t, ss, se)
        live = [a for a in asks if not _negated(t, a, data)]
        if live and receive:
            ev += [t[s:e] for s, e in live + receive]
    return ev


def unregistered_sender(n: NormalizedInput, data: RuleData) -> list[str]:
    if n.sender_status != "unregistered":
        return []
    claims = _find(data.lexicon["bank"], n.text) + _find(data.lexicon["govt"], n.text)
    return [n.text[s:e] for s, e in sorted(claims)]


def shortener_link(n: NormalizedInput, data: RuleData) -> list[str]:
    ev = []
    for url in n.urls:
        host = _host(url)
        if host and (host in data.shorteners or _registrable(host) in data.shorteners):
            ev.append(url)
    return ev


def threat_lexicon(n: NormalizedInput, data: RuleData) -> list[str]:
    return [n.text[s:e] for s, e in _find(data.lexicon["threat"], n.text)]


def urgency_plus_payment(n: NormalizedInput, data: RuleData) -> list[str]:
    urgency = _find(data.lexicon["urgency"], n.text)
    payment = _find(data.lexicon["payment"], n.text)
    if not (urgency and payment):
        return []
    return [n.text[s:e] for s, e in sorted(urgency + payment)]


def foreign_code_authority(n: NormalizedInput, data: RuleData) -> list[str]:
    foreign_in_text = [p for p in n.phones if _is_foreign(p)]
    foreign_sender = bool(n.sender) and _is_foreign(n.sender)
    if not (foreign_in_text or foreign_sender):
        return []
    claims = _find(data.lexicon["govt"], n.text)
    if not claims:
        return []
    return foreign_in_text + [n.text[s:e] for s, e in claims]


def asks_otp_or_pin(n: NormalizedInput, data: RuleData) -> list[str]:
    """An ask to share an OTP/PIN in the same sentence, unless the ask is negated
    ("do not share", "share mat karo", "किसी के साथ साझा न करें")."""
    ev, t = [], n.text
    for ss, se in _spans(t, _SENTENCE_BREAK):
        secrets = _find(data.lexicon["otp"], t, ss, se)
        if not secrets:
            continue
        live = [a for a in _find(data.lexicon["share"], t, ss, se) if not _negated(t, a, data)]
        if live:
            ev += [t[s:e] for s, e in sorted(secrets + live)]
    return ev


_CHECKS = {name: globals()[name] for name in HARD + SOFT}


def evaluate(n: NormalizedInput, data: RuleData | None = None) -> RuleSignals:
    """Run every §7.3 signal. Signal order follows §7.3; evidence is deduplicated."""
    data = data or load_rule_data()
    hard, soft, evidence = [], [], {}
    for name, check in _CHECKS.items():
        ev = list(dict.fromkeys(check(n, data)))
        if ev:
            (hard if name in HARD else soft).append(name)
            evidence[name] = ev
    return RuleSignals(hard=hard, soft=soft, evidence=evidence)


def rule_signals(text: str, sender: str | None, channel: str) -> list[str]:
    """Flat list of signal names for the §8.2 RULE_SIGNALS line (dataset builders)."""
    del channel  # no signal depends on channel yet
    r = evaluate(normalize(text, sender))
    return r.hard + r.soft
