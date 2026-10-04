"""Minimal terminal labeler for redacted real items (PRD §10.1; IMPLEMENTATION.md Phase 2).

    uv run python -m training.label [--redacted data/redacted] [--labels data/redacted/_labels.jsonl]

Walks every item in `data/redacted/*.jsonl` (files starting with `_` are skipped) that has
no label yet, shows its text and asks for: verdict, category, red-flag quotes (each must be
an exact substring of the text) with a reason, language, source_phone, obfuscated, and
whether the item is TEST (test items are never used as synthesis seeds; only real
`family_real` / `own_inbox` items can be test, PRD §10.5). `public_report` (advisory) items are
labelable but never TEST — they form a separate eval slice via `training.build_advisory` (§10.2).

Each label is appended to the labels file as soon as the item is done, so quitting loses at
most the current item. The last label for an id wins. At any prompt: `s` skips the item,
`q` quits.
"""

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import get_args

from hub.schemas import DetectorVerdict, Reason, SafeCategory, ScamCategory

VERDICTS = list(get_args(DetectorVerdict))
SCAM_CATEGORIES = list(get_args(ScamCategory))
SAFE_CATEGORIES = list(get_args(SafeCategory))
REASONS = list(get_args(Reason))
LANGUAGES = ["en", "hinglish", "hi"]
SOURCE_PHONES = ["mom", "dad", "own", "relative"]
# Only the owner's own messages can become the frozen TEST set or train real items (PRD §10.5).
REAL_SOURCES = {"family_real", "own_inbox"}
PHONE_SOURCES = {"family_real", "own_inbox"}  # public reports come from no phone of ours
# Public advisory sources: labelable, but never TEST and never train — a separate eval slice
# built by `training.build_advisory` (PRD §10.2).
ADVISORY_SOURCES = {"public_report"}


class Skip(Exception):
    pass


class Quit(Exception):
    pass


def categories_for(verdict: str) -> list[str]:
    if verdict == "SCAM":
        return SCAM_CATEGORIES
    if verdict == "SAFE":
        return SAFE_CATEGORIES
    return SCAM_CATEGORIES + SAFE_CATEGORIES  # SUSPICIOUS: genuinely ambiguous


def load_items(redacted: Path) -> list[dict]:
    items = []
    for path in sorted(redacted.glob("*.jsonl")):
        if path.name.startswith("_"):
            continue
        for line in path.read_text("utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                rec["_file"] = path.name
                items.append(rec)
    return items


def load_labels(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    labels = {}
    for line in path.read_text("utf-8").splitlines():
        if line.strip():
            lab = json.loads(line)
            labels[lab["id"]] = lab
    return labels


def append_label(path: Path, label: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(label, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


class Labeler:
    def __init__(self, ask: Callable[[str], str] = input, say: Callable[[str], None] = print):
        self.ask, self.say = ask, say

    def _read(self, prompt: str) -> str:
        answer = self.ask(prompt).strip()
        if answer == "q":
            raise Quit
        if answer == "s":
            raise Skip
        return answer

    def choose(self, name: str, options: list[str], allow_blank: bool = False) -> str | None:
        """Pick by number, exact value, or unique prefix."""
        menu = "  ".join(f"{i}={o}" for i, o in enumerate(options, 1))
        while True:
            answer = self._read(f"{name} [{menu}]{' (blank = none)' if allow_blank else ''}: ")
            if not answer and allow_blank:
                return None
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                return options[int(answer) - 1]
            matches = [o for o in options if o.lower() == answer.lower()] or [
                o for o in options if o.lower().startswith(answer.lower())
            ]
            if answer and len(matches) == 1:
                return matches[0]
            self.say(f"  ? choose one of: {', '.join(options)}")

    def yes_no(self, name: str) -> bool:
        while True:
            answer = self._read(f"{name} [y/n]: ").lower()
            if answer in ("y", "yes", "n", "no"):
                return answer.startswith("y")
            self.say("  ? y or n")

    def red_flags(self, text: str) -> list[dict]:
        flags: list[dict] = []
        while True:
            quote = self.ask(f"red-flag quote #{len(flags) + 1} (exact text, blank = done): ")
            if quote.strip() in ("q", "s"):
                self._read(quote)  # raises
            if not quote.strip():
                return flags
            if quote not in text:
                stripped = quote.strip()
                if stripped and stripped in text:
                    quote = stripped
                else:
                    self.say("  ? not an exact substring of the text; copy it exactly")
                    continue
            flags.append({"quote": quote, "reason": self.choose("  reason", REASONS)})

    def label(self, item: dict) -> dict:
        text = item["text"]
        verdict = self.choose("verdict", VERDICTS)
        category = self.choose("category", categories_for(verdict))
        flags = [] if verdict == "SAFE" else self.red_flags(text)
        language = self.choose("language", LANGUAGES)
        is_real = item.get("source") in REAL_SOURCES
        source_phone = self.choose(
            "source_phone", SOURCE_PHONES, allow_blank=item.get("source") not in PHONE_SOURCES
        )
        obfuscated = self.yes_no("obfuscated")
        if is_real:
            is_test = self.yes_no("TEST item (never a seed)")
        elif item.get("source") in ADVISORY_SOURCES:
            is_test = False
            self.say(f"  (source={item.get('source')}: advisory — separate eval slice, never test)")
        else:
            is_test = False
            self.say(f"  (source={item.get('source')}: not real, so not test)")
        return {
            "id": item["id"],
            "file": item["_file"],
            "verdict": verdict,
            "category": category,
            "red_flags": flags,
            "language": language,
            "source": item.get("source"),
            "source_phone": source_phone,
            "obfuscated": obfuscated,
            "is_test": is_test,
            "seed_group": item["id"],
        }

    def show(self, item: dict, n: int, total: int) -> None:
        self.say("\n" + "─" * 72)
        self.say(
            f"[{n}/{total}] {item['_file']}  id={item['id']}  channel={item.get('channel')}  "
            f"source={item.get('source')}  sender={item.get('sender') or 'unknown'}"
        )
        self.say("─" * 72)
        self.say(item["text"])
        self.say("─" * 72)


def run(redacted: Path, labels_path: Path, labeler: Labeler) -> int:
    """Label every unlabeled item. Returns the number labeled this session."""
    done = load_labels(labels_path)
    todo = [it for it in load_items(redacted) if it["id"] not in done]
    labeler.say(f"{len(done)} labeled, {len(todo)} to go. s = skip, q = quit.")
    count = 0
    for n, item in enumerate(todo, 1):
        labeler.show(item, n, len(todo))
        try:
            label = labeler.label(item)
        except Skip:
            labeler.say("  skipped")
            continue
        except Quit:
            break
        append_label(labels_path, label)
        count += 1
        labeler.say(f"  saved ({label['verdict']}/{label['category']}, test={label['is_test']})")
    labeler.say(f"\n{count} labeled this session; labels in {labels_path}")
    return count


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--redacted", type=Path, default=Path("data/redacted"))
    ap.add_argument("--labels", type=Path, default=Path("data/redacted/_labels.jsonl"))
    args = ap.parse_args(argv)
    try:
        run(args.redacted, args.labels, Labeler())
    except (KeyboardInterrupt, EOFError):
        print("\nstopped; earlier items are saved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
