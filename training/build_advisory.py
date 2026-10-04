"""Build the advisory eval slice from public fraud reports (PRD §10.2).

    uv run python -m training.build_advisory [--raw data/raw/public_reports.jsonl]
        [--review data/raw/public_reports.review.jsonl] [--names data/redaction_names.txt]
        [--redacted-out data/redacted/public_reports.jsonl] [--out data/splits/advisory_eval.jsonl]

Public advisories (RBI/CERT-In/news/fact-checkers) are a **separate eval slice**, never the
frozen test set (PRD §10.2, §10.5). This renders them to the §8.2 chat format used by the
detector, so `eval.run_eval --split advisory` can report independent (non-synthetic) numbers.

Labels are **auto-derived** from the reviewed `category_guess`: verdict = SCAM for a §10.1 scam
category, SAFE for a safe one (SUSPICIOUS is not auto-assigned). There are no gold red-flag
quotes, so span-F1 is not computable on this slice. Every item is redacted (§10.3) first. The
slice and its lock entry are marked `auto_labeled: true` and `provisional: true`; nothing here
touches `train.jsonl` / `dev.jsonl` / `test.jsonl`. Prints counts only, never message text.
"""

import argparse
import collections
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import get_args

from hub.detector import chat_example
from hub.normalize import normalize_text
from hub.rules import rule_signals
from hub.schemas import DetectorOutput, SafeCategory, ScamCategory
from training.redact import load_names, redact_record

ROOT = Path(__file__).resolve().parent.parent
SCAM_CATEGORIES = set(get_args(ScamCategory))
SAFE_CATEGORIES = set(get_args(SafeCategory))
LANGUAGES = {"en", "hinglish", "hi"}


def verdict_for(category: str) -> str:
    if category in SCAM_CATEGORIES:
        return "SCAM"
    if category in SAFE_CATEGORIES:
        return "SAFE"
    raise ValueError(f"category {category!r} is not in the PRD §10.1 vocabulary")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def build_examples(raw: list[dict], review: dict[str, dict], names) -> list[dict]:
    """Redact, derive labels and render each advisory item to the §8.2 chat format."""
    examples = []
    for rec in sorted(raw, key=lambda r: str(r["id"])):
        rid = rec["id"]
        red, _ = redact_record(rec, names)
        text = normalize_text(red["text"])
        category = rec["category_guess"]
        verdict = verdict_for(category)
        rev = review.get(rid, {})
        language = rev.get("language_guess", "en")
        if language not in LANGUAGES:
            language = "en"
        # No gold red-flag quotes for advisory items; verdict/category only.
        out = DetectorOutput.model_validate(
            {"verdict": verdict, "category": category, "red_flags": []}
        )
        sender = red.get("sender")
        signals = rule_signals(text, sender, red["channel"])
        examples.append(
            {
                "messages": chat_example(red["channel"], sender, signals, text, out),
                "meta": {
                    "id": rid,
                    "split": "advisory",
                    "verdict": verdict,
                    "category": category,
                    "language": language,
                    "source": "public_report",
                    "source_phone": None,
                    "obfuscated": False,
                    "seed_group": rid,
                    "hard_negative": category in ("genuine_otp", "transaction_alert"),
                    "auto_labeled": True,
                    "source_url": rec.get("source_url"),
                    "source_name": rec.get("source_name"),
                    "published_date": rec.get("published_date"),
                },
            }
        )
    return examples


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
            or None
        )
    except Exception:
        return None


def _counts(examples: list[dict]) -> dict:
    by = lambda k: dict(collections.Counter(e["meta"][k] for e in examples))  # noqa: E731
    return {
        "n": len(examples),
        "verdict": by("verdict"),
        "category": by("category"),
        "language": by("language"),
        "channel": dict(
            collections.Counter(e["messages"][1]["content"].split("\n")[0] for e in examples)
        ),
    }


def write_slice(examples: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples), encoding="utf-8"
    )


def update_lock(out: Path, examples: list[dict], lock_path: Path) -> None:
    lock = json.loads(lock_path.read_text("utf-8")) if lock_path.exists() else {}
    lock["advisory_eval"] = {
        "path": out.name,
        "sha256": _sha256(out),
        "n": len(examples),
        "source": "public_report",
        "auto_labeled": True,
        "provisional": True,
        "note": "advisory eval slice (PRD §10.2); not the test set; labels derived from category_guess",
        "counts": _counts(examples),
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
    }
    lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=ROOT / "data/raw/public_reports.jsonl")
    ap.add_argument("--review", type=Path, default=ROOT / "data/raw/public_reports.review.jsonl")
    ap.add_argument("--names", type=Path, default=ROOT / "data/redaction_names.txt")
    ap.add_argument(
        "--redacted-out", type=Path, default=ROOT / "data/redacted/public_reports.jsonl"
    )
    ap.add_argument("--out", type=Path, default=ROOT / "data/splits/advisory_eval.jsonl")
    ap.add_argument("--lock", type=Path, default=ROOT / "data/splits/splits.lock.json")
    args = ap.parse_args(argv)

    raw = _read_jsonl(args.raw)
    review = {r["id"]: r for r in _read_jsonl(args.review)} if args.review.exists() else {}
    names = load_names(args.names)

    # Redacted provenance copy (git-ignored), for human re-review.
    args.redacted_out.parent.mkdir(parents=True, exist_ok=True)
    redacted = [redact_record(r, names)[0] for r in raw]
    args.redacted_out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in redacted), encoding="utf-8"
    )

    examples = build_examples(raw, review, names)
    write_slice(examples, args.out)
    update_lock(args.out, examples, args.lock)

    c = _counts(examples)
    print(f"advisory eval slice: {c['n']} items → {args.out.name} (PROVISIONAL, auto-labeled)")
    print(f"  verdict:  {c['verdict']}")
    print(f"  category: {c['category']}")
    print(f"  language: {c['language']}")
    print("  (separate eval slice, PRD §10.2; not train/dev/test)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
