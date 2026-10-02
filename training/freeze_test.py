"""Freeze the real test set (PRD FR-32, §10.5): labeled TEST items → data/splits/test.jsonl.

    uv run python -m training.freeze_test [--redacted data/redacted]
        [--labels data/redacted/_labels.jsonl] [--out data/splits] [--force]

Each line is one §8.2 chat example plus metadata for eval slices (§12.4):
    {"messages": [system A.1, user §8.2, assistant target], "meta": {...}}
Text is normalized with `hub.normalize.normalize_text` (FR-10). RULE_SIGNALS come from
`hub.rules.rule_signals` once it exists, else `[]`; the lock file records which.

Writes `splits.lock.json` with the SHA-256 of test.jsonl and counts by verdict, category,
language, source, source_phone and obfuscated. Refuses to overwrite an existing
test.jsonl without --force (re-freezing also drops stale train/dev entries from the lock).
Prints counts only, never message text.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from hub.detector import chat_example
from hub.normalize import normalize_text
from hub.schemas import DetectorOutput
from training.label import REAL_SOURCES, load_items, load_labels

COUNT_KEYS = ("verdict", "category", "language", "source", "source_phone", "obfuscated")


class FreezeError(Exception):
    """Validation failure. Messages carry ids only, never text."""


def rule_signals_fn():
    """`hub.rules.rule_signals` if implemented, else None."""
    try:
        from hub import rules
    except ImportError:
        return None
    return getattr(rules, "rule_signals", None)


def build_examples(items: list[dict], labels: dict[str, dict], signals_fn) -> list[dict]:
    by_id = {it["id"]: it for it in items}
    examples = []
    for lab in sorted(
        (lab for lab in labels.values() if lab.get("is_test")), key=lambda x: str(x["id"])
    ):
        item_id = lab["id"]
        item = by_id.get(item_id)
        if item is None:
            raise FreezeError(f"label {item_id!r} has no redacted item")
        if lab.get("source") not in REAL_SOURCES or item.get("source") not in REAL_SOURCES:
            raise FreezeError(f"{item_id!r}: test items must be real ({sorted(REAL_SOURCES)})")
        text = normalize_text(item["text"])
        flags = [
            {"quote": normalize_text(f["quote"]), "reason": f["reason"]} for f in lab["red_flags"]
        ]
        try:
            out = DetectorOutput.model_validate(
                {"verdict": lab["verdict"], "category": lab["category"], "red_flags": flags}
            )
        except Exception as e:
            raise FreezeError(
                f"{item_id!r}: label fails §8.2 schema ({type(e).__name__})"
            ) from None
        if any(f.quote not in text for f in out.red_flags):
            raise FreezeError(
                f"{item_id!r}: a red-flag quote is no longer an exact substring; relabel it"
            )
        signals = signals_fn(text, item.get("sender"), item["channel"]) if signals_fn else []
        examples.append(
            {
                "messages": chat_example(item["channel"], item.get("sender"), signals, text, out),
                "meta": {
                    "id": item_id,
                    "split": "test",
                    "verdict": out.verdict,
                    "category": out.category,
                    "language": lab["language"],
                    "source": lab["source"],
                    "source_phone": lab.get("source_phone"),
                    "obfuscated": lab["obfuscated"],
                    "seed_group": lab.get("seed_group", item_id),
                },
            }
        )
    return examples


def counts(examples: list[dict]) -> dict[str, dict[str, int]]:
    def name(v) -> str:
        return str(v).lower() if isinstance(v, bool) or v is None else str(v)

    return {
        key: dict(sorted(Counter(name(ex["meta"][key]) for ex in examples).items()))
        for key in COUNT_KEYS
    }


def git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--redacted", type=Path, default=Path("data/redacted"))
    ap.add_argument("--labels", type=Path, default=Path("data/redacted/_labels.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/splits"))
    ap.add_argument("--force", action="store_true", help="overwrite an existing frozen test set")
    args = ap.parse_args(argv)

    test_path, lock_path = args.out / "test.jsonl", args.out / "splits.lock.json"
    if test_path.exists() and not args.force:
        print(
            f"refusing: {test_path} is frozen; use --force only before any training run",
            file=sys.stderr,
        )
        return 1

    signals_fn = rule_signals_fn()
    try:
        examples = build_examples(load_items(args.redacted), load_labels(args.labels), signals_fn)
    except FreezeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not examples:
        print("error: no labeled items marked TEST", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(ex, ensure_ascii=False) + "\n" for ex in examples).encode("utf-8")
    test_path.write_bytes(data)

    lock = json.loads(lock_path.read_text("utf-8")) if lock_path.exists() else {}
    stale = [k for k in ("train", "dev") if k in lock]
    for k in stale:
        del lock[k]  # built against the old test set (leakage check); rebuild them
    lock["test"] = {
        "path": test_path.name,  # relative to the lock file; no local paths
        "sha256": hashlib.sha256(data).hexdigest(),
        "n": len(examples),
        "counts": counts(examples),
        "rule_signals": "hub.rules.rule_signals"
        if signals_fn
        else "none (hub.rules not implemented)",
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
    }
    lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    t = lock["test"]
    print(f"froze {t['n']} test items → {test_path} (sha256 {t['sha256'][:12]}…)")
    for key, c in t["counts"].items():
        print(f"  {key:<13} " + "  ".join(f"{k}={v}" for k, v in c.items()))
    print(f"  rule_signals  {t['rule_signals']}")
    if stale:
        print(f"dropped stale {', '.join(stale)} entries from {lock_path}; rebuild them")
    print(f"wrote {lock_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
