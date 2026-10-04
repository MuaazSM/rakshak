"""Redaction of real examples before they leave data/raw/ (PRD FR-30, §10.3).

    uv run python -m training.redact [--raw data/raw] [--out data/redacted]
        [--names data/redaction_names.txt] [--force]

Reads `data/raw/*.jsonl` (fields: id, text, sender?, channel, source, source_url?) and writes the same
records with `text` and `sender` masked to `data/redacted/<same name>.jsonl`.

Masks (§10.3):
  family names / addresses (from the names file)  → <NAME> / <ADDR>
  personal phone numbers (`phone:` lines)         → <PHONE>
  any other phone number (treated as a scammer's) → country code + first 4 digits + <PHONE>
  OTP / PIN digits near an OTP/PIN/code keyword   → <OTP>
  account / card / Aadhaar fragments              → <ACCT>
Kept: amounts, links and domains, UPI handles, sender headers (e.g. VK-SBIUPD), toll-free
1800/1860 numbers.

Names file (git-ignored; format in `data/redaction_names.example.txt`): one entry per line,
`# comments`; plain lines are names, `addr: ...` lines are addresses, `phone: ...` lines are
personal numbers. Matching is case-insensitive and whole-word; list Devanagari and Latin
spellings separately.

The report on stdout gives mask counts per item, never text. Every item must still be
reviewed by hand before it is used (§10.3).
"""

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

KINDS = ("name", "addr", "phone_personal", "phone_partial", "otp", "acct")

# Letters incl. Devanagari combining marks, so whole-word checks work for Hindi names too.
_WORD = r"[\w\u0900-\u097F]"

# Spans that are never touched by the digit masks.
_PROTECT = re.compile(
    r"<[A-Z]+>"  # earlier masks
    r"|(?:https?://|www\.)\S+"  # links
    r"|\b[\w-]+(?:\.[\w-]+)*\.(?:[a-z]{2,24})(?:/\S*)?(?=[\s,;:!?)\]]|$)"  # bare domains
    r"|[\w.\-]+@[\w.\-]+"  # UPI handles / email-like
    r"|(?<!\d)1[89]\d0[\s-]?\d{3}[\s-]?\d{3,4}(?!\d)",  # toll-free 1800/1860
    re.IGNORECASE,
)

_CURRENCY_BEFORE = re.compile(r"(?:rs\.?|inr|₹|rupees?|रु\.?|रुपये)\s*$", re.IGNORECASE)
_CURRENCY_AFTER = re.compile(r"^\s*(?:/-|rs\b|rupees?|रुपये|रुपए|lakh|लाख|crore|करोड़)", re.IGNORECASE)

# Card (4-4-4-4) and Aadhaar (4-4-4) groups.
_CARD = re.compile(r"(?<![\d])\d{4}(?:[ -]\d{4}){2,3}(?![\d])")
# "A/c XX1234", "account no. 123456789", "card ending 4321", "खाता 1234".
_ACCT_KEYWORD = re.compile(
    r"(?P<kw>\b(?:a/c|acct|account|ac|card|khata)\b|खाता|कार्ड)"
    r"(?P<mid>\s*(?:no\.?|number|num|ending(?:\s+(?:with|in))?|ends\s+with)?\s*[:#.\-]?\s*)"
    r"(?P<num>[Xx*]*\d{3,18})(?![\d])",
    re.IGNORECASE,
)
_ACCT_MASKED = re.compile(r"(?<![\w])[Xx*]{2,}\d{2,6}(?![\d])")
_LONG_DIGITS = re.compile(r"(?<![\d])\d{9,18}(?![\d])")

# Phones. Group "cc" = country code or trunk prefix, kept for partial masks.
_PHONE_INTL = re.compile(
    r"(?<![\w+])(?P<cc>\+(?!91)\d{1,3}[\s-]?)(?P<num>\d(?:[\s-]?\d){6,12})(?![\d])"
)
_PHONE_IN = re.compile(
    r"(?<![\w+])(?P<cc>(?:\+91|0091|91)[\s-]?|0)?(?P<num>[6-9](?:[\s-]?\d){9})(?![\d])"
)
_PHONE_LANDLINE = re.compile(r"(?<![\w+])(?P<cc>0)(?P<num>\d{2,4}[\s-]\d{6,8})(?![\d])")

_OTP_KEYWORD = re.compile(
    r"\b(?:otp|one[\s-]?time[\s-]?password|m?pin|passcode|password|pass\s?code|"
    r"verification|verify|code|cvv)\b|ओटीपी|पिन|कोड|पासवर्ड",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"[.!?।\n]\s")
_OTP_CANDIDATE = re.compile(r"(?<![\d.,:/])\d{4,8}(?![\d:/]|\.\d)")


@dataclass
class Names:
    names: list[str] = field(default_factory=list)
    addrs: list[str] = field(default_factory=list)
    phones: list[str] = field(default_factory=list)  # last 10 digits


def load_names(path: Path | None) -> Names:
    out = Names()
    if path is None or not path.exists():
        return out
    for raw in path.read_text("utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(":")
        if key.strip().lower() == "addr" and value.strip():
            out.addrs.append(value.strip())
        elif key.strip().lower() == "phone" and value.strip():
            digits = re.sub(r"\D", "", value)
            if len(digits) >= 6:
                out.phones.append(digits[-10:])
        else:
            out.names.append(line)
    return out


def _sub_list(text: str, entries: list[str], tag: str) -> tuple[str, int]:
    total = 0
    for entry in sorted(set(entries), key=len, reverse=True):
        pattern = r"\s+".join(re.escape(w) for w in entry.split())
        text, n = re.subn(rf"(?<!{_WORD}){pattern}(?!{_WORD})", tag, text, flags=re.IGNORECASE)
        total += n
    return text, total


def _sub_personal_phones(text: str, phones: list[str]) -> tuple[str, int]:
    total = 0
    for digits in phones:
        body = r"[\s-]?".join(digits)
        text, n = re.subn(
            rf"(?<![\d+])(?:(?:\+\d{{1,3}}|0091|91)[\s-]?|0)?{body}(?!\d)", "<PHONE>", text
        )
        total += n
    return text, total


def _partial(m: re.Match) -> str:
    """Keep country code (or trunk 0) + first 4 national digits, mask the rest."""
    cc, num = m.group("cc") or "", m.group("num")
    kept, seen = [], 0
    for ch in num:
        kept.append(ch)
        seen += ch.isdigit()
        if seen == 4:
            break
    return f"{cc}{''.join(kept)}<PHONE>"


def _is_amount(seg: str, start: int, end: int) -> bool:
    return bool(
        _CURRENCY_BEFORE.search(seg[max(0, start - 8) : start]) or _CURRENCY_AFTER.match(seg[end:])
    )


def _mask_otp(seg: str) -> tuple[str, int]:
    out, last, n = [], 0, 0
    for m in _OTP_CANDIDATE.finditer(seg):
        s, e = m.span()
        # Keyword must be in the same sentence, close to the digits.
        before = _SENTENCE_END.split(seg[max(0, s - 30) : s])[-1]
        after = _SENTENCE_END.split(seg[e : e + 20])[0]
        context = before + " " + after
        if _OTP_KEYWORD.search(context) and not _is_amount(seg, s, e):
            out += [seg[last:s], "<OTP>"]
            last, n = e, n + 1
    out.append(seg[last:])
    return "".join(out), n


def _mask_long(seg: str) -> tuple[str, int]:
    def repl(m: re.Match) -> str:
        return m.group(0) if _is_amount(seg, *m.span()) else "<ACCT>"

    new = _LONG_DIGITS.sub(repl, seg)
    return new, new.count("<ACCT>") - seg.count("<ACCT>")


def _mask_digits(seg: str, counts: Counter) -> str:
    """Digit masks on a segment that contains no protected span."""
    seg, n = _CARD.subn("<ACCT>", seg)
    counts["acct"] += n
    seg, n = _ACCT_KEYWORD.subn(lambda m: f"{m.group('kw')}{m.group('mid')}<ACCT>", seg)
    counts["acct"] += n
    seg, n = _ACCT_MASKED.subn("<ACCT>", seg)
    counts["acct"] += n
    for pat in (_PHONE_INTL, _PHONE_IN, _PHONE_LANDLINE):
        seg, n = pat.subn(_partial, seg)
        counts["phone_partial"] += n
    seg, n = _mask_otp(seg)
    counts["otp"] += n
    seg, n = _mask_long(seg)
    counts["acct"] += n
    return seg


def redact_text(text: str, names: Names, counts: Counter | None = None) -> str:
    """Apply all §10.3 masks to one string. Mask counts are added to `counts`."""
    counts = Counter() if counts is None else counts
    text, n = _sub_list(text, names.addrs, "<ADDR>")
    counts["addr"] += n
    text, n = _sub_list(text, names.names, "<NAME>")
    counts["name"] += n
    text, n = _sub_personal_phones(text, names.phones)
    counts["phone_personal"] += n

    out, last = [], 0
    for m in _PROTECT.finditer(text):
        out += [_mask_digits(text[last : m.start()], counts), m.group(0)]
        last = m.end()
    out.append(_mask_digits(text[last:], counts))
    return "".join(out)


def redact_record(rec: dict, names: Names) -> tuple[dict, Counter]:
    for key in ("id", "text", "channel", "source"):
        if key not in rec:
            raise ValueError(f"record {rec.get('id', '?')!r} is missing {key!r}")
    if rec["source"] == "public_report" and not rec.get("source_url"):
        raise ValueError(f"record {rec['id']!r}: public_report needs source_url (PRD §10.2)")
    counts: Counter = Counter()
    out = dict(rec)
    out["text"] = redact_text(rec["text"], names, counts)
    if rec.get("sender"):
        # Sender headers have no digits worth masking; numbers and saved contact names do.
        out["sender"] = redact_text(rec["sender"], names, counts)
    return out, counts


def digits_left(rec: dict) -> int:
    """Unmasked runs of ≥ 4 digits outside links and amounts: a hint for manual review."""
    text = re.sub(r"\d[\d\s-]*<PHONE>", "<PHONE>", rec["text"])  # kept prefix of a scammer number
    text = _PROTECT.sub(" ", text)
    return sum(1 for m in re.finditer(r"\d{4,}", text) if not _is_amount(text, *m.span()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/redacted"))
    ap.add_argument("--names", type=Path, default=Path("data/redaction_names.txt"))
    ap.add_argument("--force", action="store_true", help="overwrite existing redacted files")
    args = ap.parse_args(argv)

    if not args.names.exists():
        print(f"warn: {args.names} not found; names, addresses and personal phones won't be masked")
    names = load_names(args.names)
    files = sorted(args.raw.glob("*.jsonl"))
    if not files:
        print(f"no *.jsonl in {args.raw}")
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    total: Counter = Counter()
    for src in files:
        dst = args.out / src.name
        if dst.exists() and not args.force:
            print(f"skip {src.name}: {dst} exists (may hold manual review edits); use --force")
            continue
        print(f"\n{src.name} → {dst}")
        print(f"  {'id':<20} " + " ".join(f"{k:>14}" for k in KINDS) + f" {'digits_left':>12}")
        lines = []
        for i, raw in enumerate(src.read_text("utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            try:
                rec, counts = redact_record(json.loads(raw), names)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"error: {src.name} line {i}: {type(e).__name__}: {e}", file=sys.stderr)
                return 2
            total.update(counts)
            left = digits_left(rec)
            print(
                f"  {rec['id']!s:<20} "
                + " ".join(f"{counts[k]:>14}" for k in KINDS)
                + f" {left:>12}"
            )
            lines.append(json.dumps(rec, ensure_ascii=False))
        dst.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\ntotal  " + "  ".join(f"{k}={total[k]}" for k in KINDS))
    print("Review every redacted item by hand before labeling (PRD §10.3).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
