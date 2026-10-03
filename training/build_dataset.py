"""Dataset builder: dedup, splits by seed_group, splits.lock.json with SHA-256 (PRD FR-32, §10.5).

    uv run python -m training.build_dataset [--synthetic data/synthetic] [--redacted data/redacted]
        [--labels data/redacted/_labels.jsonl] [--seeds data/seeds/seeds.jsonl] [--out data/splits]
        [--dev-per-stratum 8] [--dropout 0.2] [--seed 20261003]

Inputs: accepted synthetic batches (`data/synthetic/*.jsonl`, not `review_*`) and, when they
exist, redacted real items that are labeled and NOT marked TEST (PRD §17: labels live in
`data/redacted/_labels.jsonl`).

Steps: validate (§8.2 schema, quotes exact substrings of the normalized text) → drop anything
>= 0.8 MinHash-Jaccard (char 5-grams) similar to the frozen test set, if there is one → split by
`seed_group` (dev = one or more whole groups per verdict/category stratum, capped per group;
the rest of a dev group is discarded, never sent to train) → MinHash dedupe within dev and
within train → drop train items similar to any dev item → render the §8.2 chat format with
`hub.detector.chat_example` and `hub.rules.rule_signals` (20% rule-signal dropout on train only).

Never creates `test.jsonl`. If it is absent the lock gets `"test": null, "provisional": true`
and a loud warning is printed; the test file is only read for hashing, never printed.
Prints counts only, never message text.
"""

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from datasketch import MinHash, MinHashLSH

from hub.detector import chat_example
from hub.normalize import normalize_text, sender_status
from hub.rules import rule_signals
from hub.schemas import DetectorOutput
from training.label import REAL_SOURCES, load_items, load_labels
from training.synth import BUILD_SEED, is_hard_negative

NUM_PERM = 128
JACCARD = 0.8
SHINGLE = 5
COUNT_KEYS = ("verdict", "category", "language", "source", "obfuscated", "hard_negative")
FORBIDDEN_OUTPUTS = {"test.jsonl"}


class BuildError(Exception):
    """Validation failure. Messages carry ids and counts only, never text."""


# --- loading -------------------------------------------------------------------------------


def valid_item(it: dict) -> dict | None:
    """Normalize and validate one pool item; None if it fails the §8.2 schema or a quote is
    not an exact substring of the normalized text."""
    text = normalize_text(it.get("text") or "")
    if not text:
        return None
    flags = [
        {"quote": normalize_text(f["quote"]), "reason": f["reason"]}
        for f in it.get("red_flags") or []
    ]
    try:
        out = DetectorOutput.model_validate(
            {"verdict": it["verdict"], "category": it["category"], "red_flags": flags}
        )
    except Exception:
        return None
    if any(f.quote not in text for f in out.red_flags):
        return None
    if out.verdict != "SAFE" and not out.red_flags:
        return None
    sender = it.get("sender")
    return {**it, "text": text, "sender": None if sender in (None, "", "unknown") else sender}


def load_synthetic(directory: Path) -> tuple[list[dict], int]:
    """(items, n_invalid) from accepted batches; `review_*` files are samples, not inputs."""
    items, bad = [], 0
    for path in sorted(directory.glob("*.jsonl")):
        if path.name.startswith("review_"):
            continue
        for line in path.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            it = valid_item(raw)
            if it is None:
                bad += 1
                continue
            meta = raw.get("meta", {})
            items.append(
                {
                    **it,
                    "source": meta.get("source", "synthetic"),
                    "source_phone": None,
                    "obfuscated": bool(meta.get("obfuscated", False)),
                    "seed_group": meta.get("seed_group", raw["id"]),
                    "hard_negative": bool(
                        meta.get(
                            "hard_negative",
                            is_hard_negative(it["category"], it["verdict"], it["text"]),
                        )
                    ),
                }
            )
    return items, bad


def load_real(redacted: Path, labels_path: Path) -> tuple[list[dict], int]:
    """Labeled, non-TEST real items (family_real / own_inbox). Empty when nothing is labeled."""
    labels = load_labels(labels_path)
    by_id = {it["id"]: it for it in load_items(redacted)} if redacted.exists() else {}
    items, bad = [], 0
    for lab in labels.values():
        raw = by_id.get(lab["id"])
        if lab.get("is_test") or raw is None:
            continue
        if lab.get("source") not in REAL_SOURCES or raw.get("source") not in REAL_SOURCES:
            continue  # advisory items are paraphrased seeds elsewhere, not examples
        it = valid_item({**raw, **{k: lab[k] for k in ("verdict", "category", "red_flags")}})
        if it is None:
            bad += 1
            continue
        items.append(
            {
                **it,
                "language": lab["language"],
                "source": lab["source"],
                "source_phone": lab.get("source_phone"),
                "obfuscated": bool(lab.get("obfuscated", False)),
                "seed_group": lab.get("seed_group", lab["id"]),
                "hard_negative": is_hard_negative(it["category"], it["verdict"], it["text"]),
            }
        )
    return items, bad


# --- MinHash ---------------------------------------------------------------------------------


def shingles(text: str) -> set[str]:
    t = normalize_text(text).lower()
    return {t[i : i + SHINGLE] for i in range(max(1, len(t) - SHINGLE + 1))}


def minhash(text: str) -> MinHash:
    m = MinHash(num_perm=NUM_PERM, seed=1)
    for s in shingles(text):
        m.update(s.encode("utf-8"))
    return m


class Index:
    """MinHash LSH over texts with an exact Jaccard check on every candidate hit."""

    def __init__(self) -> None:
        self.lsh = MinHashLSH(threshold=JACCARD, num_perm=NUM_PERM)
        self.sets: dict[str, set[str]] = {}

    def add(self, key: str, text: str) -> None:
        self.sets[key] = shingles(text)
        self.lsh.insert(key, minhash(text))

    def similar(self, text: str) -> bool:
        cand = self.lsh.query(minhash(text))
        mine = shingles(text)
        return any(len(mine & self.sets[k]) / len(mine | self.sets[k]) >= JACCARD for k in cand)


def dedupe(items: list[dict], ref: Index | None = None) -> list[dict]:
    """Keep the first of every near-duplicate cluster (items are visited in id order). With
    `ref`, also drop anything similar to it (leakage / train-vs-dev)."""
    idx, kept = Index(), []
    for it in sorted(items, key=lambda x: x["id"]):
        if idx.similar(it["text"]) or (ref is not None and ref.similar(it["text"])):
            continue
        idx.add(it["id"], it["text"])
        kept.append(it)
    return kept


def index_of(items: list[dict]) -> Index:
    idx = Index()
    for it in items:
        idx.add(it["id"], it["text"])
    return idx


def load_test_index(test_path: Path) -> Index:
    """MinHash index of the frozen test set's MESSAGE texts. Read for hashing only."""
    idx = Index()
    for i, line in enumerate(test_path.read_text("utf-8").splitlines()):
        if line.strip():
            user = json.loads(line)["messages"][1]["content"]
            idx.add(f"test-{i}", user.split("MESSAGE:\n", 1)[1])
    return idx


# --- split -----------------------------------------------------------------------------------


def stratum(members: list[dict]) -> tuple[str, str]:
    """Dev stratum of a seed group: its verdict and (majority) category; one stratum per scam
    category, per safe category, and SUSPICIOUS."""
    verdict = members[0]["verdict"]
    if verdict == "SUSPICIOUS":
        return (verdict, "-")
    top = Counter(m["category"] for m in members).most_common()
    return (verdict, sorted(c for c, n in top if n == top[0][1])[0])


def split_groups(
    items: list[dict], *, dev_per_stratum: int, dev_max_per_group: int, seed: int
) -> tuple[list[dict], list[dict], list[str]]:
    """(train, dev, dev_group_ids). Per stratum, whole seed groups go to dev (chosen with a
    seeded shuffle) until the stratum has `dev_per_stratum` dev items; a stratum with a single
    group stays in train. Only `dev_max_per_group` items of a dev group are kept; the rest of
    the group is discarded."""
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        groups[it["seed_group"]].append(it)
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for gid in sorted(groups):
        strata[stratum(groups[gid])].append(gid)
    dev_groups: list[str] = []
    dev: list[dict] = []
    for key in sorted(strata):
        gids = strata[key][:]
        rng.shuffle(gids)
        have = 0
        for gid in gids[: max(0, len(gids) - 1)]:  # always leave one group for train
            if have >= dev_per_stratum:
                break
            members = sorted(dedupe(groups[gid]), key=lambda x: x["id"])
            picked = rng.sample(members, min(dev_max_per_group, len(members)))
            dev += picked
            dev_groups.append(gid)
            have += len(picked)
    dev_set = set(dev_groups)
    train = [it for it in items if it["seed_group"] not in dev_set]
    return train, sorted(dev, key=lambda x: x["id"]), sorted(dev_groups)


# --- build-time diversification (breaks format shortcuts) ---------------------------------------

SCREENSHOT_FRACTION = 0.15
_PLACEHOLDERS = re.compile(r"<OTP>|<ACCT>")
_REF_CONTEXT = re.compile(r"(ref|id|no\.?|number|#|order|संदर्भ|संख्या)\W{0,4}$", re.IGNORECASE)
_LAST4_CONTEXT = re.compile(r"(xx|\*+|ending( in| with)?|अंतिम|ending)\s*$", re.IGNORECASE)


def _digits(rng: random.Random, lo: int, hi: int) -> str:
    n = rng.randint(lo, hi)
    return str(rng.randint(1, 9)) + "".join(rng.choice("0123456789") for _ in range(n - 1))


def _replacement(token: str, before: str, rng: random.Random) -> str:
    """A realistic value for one redaction placeholder, chosen from its left context."""
    if token == "<OTP>":
        return _digits(rng, 4, 6)
    if _LAST4_CONTEXT.search(before):
        return _digits(rng, 4, 4)
    if _REF_CONTEXT.search(before):
        return _digits(rng, 6, 12)
    return rng.choice([f"XX{_digits(rng, 4, 4)}", f"**{_digits(rng, 4, 4)}", _digits(rng, 6, 12)])


def fill_placeholders(item: dict, rng: random.Random) -> dict | None:
    """Replace <OTP> / <ACCT> with realistic digits, in the text and (consistently) in every
    red-flag quote. A flag whose quote only partly overlaps a placeholder is dropped; None if a
    SCAM/SUSPICIOUS item has no flag left. The hub does not redact at runtime, so training on
    placeholders would teach "placeholder -> SAFE"."""
    text = item["text"]
    spans = []
    for m in _PLACEHOLDERS.finditer(text):
        spans.append((m.start(), m.end(), _replacement(m.group(0), text[: m.start()][-14:], rng)))
    if not spans:
        return item

    def apply(lo: int, hi: int) -> str | None:
        out, pos = [], lo
        for a, b, rep in spans:
            if b <= lo or a >= hi:
                continue
            if a < lo or b > hi:
                return None  # quote cuts through a placeholder
            out += [text[pos:a], rep]
            pos = b
        return "".join(out) + text[pos:hi]

    flags = []
    for f in item["red_flags"]:
        start = text.find(f["quote"])
        new_q = apply(start, start + len(f["quote"])) if start >= 0 else None
        if new_q:
            flags.append({"quote": new_q, "reason": f["reason"]})
    if item["verdict"] != "SAFE" and not flags:
        return None
    return {**item, "text": apply(0, len(text)), "red_flags": flags}


def add_reference(item: dict, rng: random.Random) -> dict:
    """Append a masked account / reference number to a scam-side text (quotes stay valid), so
    digit-run formats are not tied to SAFE."""
    hi = item["language"] == "hi"
    tail = rng.choice(
        [
            f"Ref No: {_digits(rng, 8, 12)}",
            f"A/c XX{_digits(rng, 4, 4)}",
            f"Case ID {_digits(rng, 6, 9)}",
            f"Txn ID {_digits(rng, 10, 12)}",
        ]
    )
    if hi and tail.startswith("Ref"):
        tail = f"संदर्भ संख्या: {_digits(rng, 8, 12)}"
    sep = "\n" if item["channel"] == "whatsapp" and rng.random() < 0.3 else " "
    return {**item, "text": f"{item['text']}{sep}{tail}"}


def diversify(item: dict, seed: int) -> dict | None:
    """Deterministic per-item (seeded by id) format changes applied at build time; the source
    batch files are never modified:
      * <OTP> / <ACCT> placeholders -> realistic digits (consistent in text and quotes);
      * ~35% of SCAM and ~35% of SUSPICIOUS texts get a masked account / reference number;
      * `personal` SAFE senders: 45% phone numbers, 15% unknown, rest contact names; 15% of SAFE
        delivery updates come from a delivery agent's +91 number; a further 20% of
        non-call SCAM/SUSPICIOUS senders become spoofed DLT-style headers;
      * ~15% of sms / whatsapp items become channel "screenshot" (all verdicts alike)."""
    rng = random.Random(f"{seed}|diversify|{item['id']}")
    out = fill_placeholders(item, rng)
    if out is None:
        return None
    if out["verdict"] != "SAFE" and rng.random() < 0.35:
        out = add_reference(out, rng)
    if out["verdict"] == "SAFE":
        phone = f"+91 {rng.choice('6789')}{_digits(rng, 4, 4)} {rng.randrange(10**5):05d}"
        r = rng.random()
        if out["category"] == "personal" and out["channel"] != "call_description":
            out = {**out, "sender": phone if r < 0.45 else None if r < 0.60 else out["sender"]}
        elif (
            out["category"] == "delivery_update"
            and out["channel"] != "call_description"
            and r < 0.15
        ):
            out = {**out, "sender": phone}
    elif (
        out["channel"] != "call_description"
        and sender_status(out["sender"]) != "registered"
        and rng.random() < 0.20
    ):  # scam side: a further 20% of senders spoof a DLT-style header
        head = "".join(rng.choices("ABCDEFGHIJKLMNOPRSTUVW", k=2))
        tail = "".join(rng.choices("ABCDEFGHIKLMNOPRSTUVY", k=6))
        out = {**out, "sender": f"{head}-{tail}"}
    if out["channel"] in ("sms", "whatsapp") and rng.random() < SCREENSHOT_FRACTION:
        out = {**out, "channel": "screenshot"}
    return out


def shortcut_report(examples: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    """Per verdict: share of items by channel and sender status, and share containing a 4+
    digit run. Counts only (no text), for the "no trivial separator" check."""
    rep: dict = {}
    for v in sorted({e["meta"]["verdict"] for e in examples}):
        xs = [e for e in examples if e["meta"]["verdict"] == v]
        users = [e["messages"][1]["content"] for e in xs]
        status = Counter(re.search(r"SENDER: .*\((\w+)\)\n", u).group(1) for u in users)
        chan = Counter(re.match(r"CHANNEL: (\w+)", u).group(1) for u in users)
        digit = sum(bool(re.search(r"\d{4,}", u.split("MESSAGE:\n", 1)[1])) for u in users)
        n = len(xs)
        rep[v] = {
            "n": n,
            "channel": {k: round(c / n, 2) for k, c in sorted(chan.items())},
            "sender_status": {k: round(c / n, 2) for k, c in sorted(status.items())},
            "digit_run": round(digit / n, 2),
        }
    return rep


# --- render ----------------------------------------------------------------------------------


def render(it: dict, split: str, *, dropout: float, seed: int) -> dict:
    """§8.2 chat example. RULE_SIGNALS come from hub.rules on the normalized text; on TRAIN
    only, a deterministic `dropout` fraction of examples gets RULE_SIGNALS: [] (PRD §11.1)."""
    signals = rule_signals(it["text"], it["sender"], it["channel"])
    if split == "train" and random.Random(f"{seed}|dropout|{it['id']}").random() < dropout:
        signals = []
    out = DetectorOutput.model_validate(
        {"verdict": it["verdict"], "category": it["category"], "red_flags": it["red_flags"]}
    )
    return {
        "messages": chat_example(it["channel"], it["sender"], signals, it["text"], out),
        "meta": {
            "id": it["id"],
            "split": split,
            "verdict": it["verdict"],
            "category": it["category"],
            "language": it["language"],
            "source": it["source"],
            "source_phone": it.get("source_phone"),
            "obfuscated": it["obfuscated"],
            "seed_group": it["seed_group"],
            "hard_negative": it["hard_negative"],
        },
    }


def counts(examples: list[dict]) -> dict[str, dict[str, int]]:
    def name(v) -> str:
        return str(v).lower() if isinstance(v, bool) or v is None else str(v)

    return {
        key: dict(sorted(Counter(name(ex["meta"][key]) for ex in examples).items()))
        for key in COUNT_KEYS
    }


def serialize(examples: list[dict]) -> bytes:
    return "".join(json.dumps(ex, ensure_ascii=False) + "\n" for ex in examples).encode("utf-8")


def write_split(path: Path, data: bytes) -> str:
    """Write a split file and return its SHA-256. Refuses to touch the frozen test set."""
    if path.name in FORBIDDEN_OUTPUTS:
        raise BuildError(f"refusing to write {path.name}: the test set is created by freeze_test")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def update_lock(
    lock_path: Path, entries: dict[str, dict], *, test_present: bool, extra: dict
) -> dict:
    """Write train/dev entries; keep an existing "test" entry untouched, else `"test": null`
    and `"provisional": true`."""
    lock = json.loads(lock_path.read_text("utf-8")) if lock_path.exists() else {}
    lock.update(entries)
    lock.setdefault("test", None)
    lock["provisional"] = lock["test"] is None or not test_present
    lock.update(extra)
    lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return lock


# --- main ------------------------------------------------------------------------------------


def build(
    pool: list[dict],
    *,
    test_index: Index | None,
    dev_per_stratum: int = 8,
    dev_max_per_group: int = 8,
    dropout: float = 0.2,
    seed: int = BUILD_SEED,
) -> tuple[list[dict], list[dict], dict[str, int]]:
    """Pool → (train examples, dev examples, stats). Pure apart from rule evaluation."""
    stats = {"pool": len(pool)}
    if test_index is not None:
        pool = [it for it in pool if not test_index.similar(it["text"])]
    stats["after_leakage"] = len(pool)
    train, dev, dev_groups = split_groups(
        pool, dev_per_stratum=dev_per_stratum, dev_max_per_group=dev_max_per_group, seed=seed
    )
    dev = dedupe(dev)
    train_d = dedupe(train)
    train_d = dedupe(train_d, ref=index_of(dev))
    stats |= {
        "train_dup_removed": len(train) - len(train_d),
        "dev_groups": len(dev_groups),
        "discarded_dev_group_items": sum(1 for it in pool if it["seed_group"] in set(dev_groups))
        - len(dev),
    }
    if {it["seed_group"] for it in train_d} & {it["seed_group"] for it in dev}:
        raise BuildError("seed_group present in both train and dev")
    if test_index is not None and any(test_index.similar(it["text"]) for it in [*train_d, *dev]):
        raise BuildError("leakage: a train/dev item is >= 0.8 similar to a test item")
    rng = random.Random(seed)
    train_d = [x for x in (diversify(it, seed) for it in train_d) if x is not None]
    dev = [x for x in (diversify(it, seed) for it in dev) if x is not None]
    train_ex = [render(it, "train", dropout=dropout, seed=seed) for it in train_d]
    rng.shuffle(train_ex)
    dev_ex = [render(it, "dev", dropout=dropout, seed=seed) for it in dev]
    return train_ex, dev_ex, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--synthetic", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--redacted", type=Path, default=Path("data/redacted"))
    ap.add_argument("--labels", type=Path, default=Path("data/redacted/_labels.jsonl"))
    ap.add_argument("--seeds", type=Path, default=Path("data/seeds/seeds.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/splits"))
    ap.add_argument("--dev-per-stratum", type=int, default=8)
    ap.add_argument("--dev-max-per-group", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=BUILD_SEED)
    args = ap.parse_args(argv)

    syn, bad_syn = load_synthetic(args.synthetic)
    real, bad_real = load_real(args.redacted, args.labels)
    pool = real + syn
    if not pool:
        print("error: no synthetic or redacted items found", file=sys.stderr)
        return 2

    test_path = args.out / "test.jsonl"
    test_index = load_test_index(test_path) if test_path.exists() else None
    if test_index is None:
        print(
            "WARNING: data/splits/test.jsonl does not exist. Skipping the test leakage check; "
            'writing a PROVISIONAL lock ("test": null). Re-run after freeze_test.',
            file=sys.stderr,
        )

    try:
        train, dev, stats = build(
            pool,
            test_index=test_index,
            dev_per_stratum=args.dev_per_stratum,
            dev_max_per_group=args.dev_max_per_group,
            dropout=args.dropout,
            seed=args.seed,
        )
        built_at = datetime.now(UTC).isoformat(timespec="seconds")
        sha, entries = git_sha(), {}
        for name, ex in (("train", train), ("dev", dev)):
            data = serialize(ex)
            digest = write_split(args.out / f"{name}.jsonl", data)
            entries[name] = {
                "path": f"{name}.jsonl",
                "sha256": digest,
                "n": len(ex),
                "counts": counts(ex),
                "built_at": built_at,
                "git_sha": sha,
            }
    except BuildError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    seeds_n = len(args.seeds.read_text("utf-8").splitlines()) if args.seeds.exists() else 0
    extra = {
        "build": {
            "build_seed": args.seed,
            "generator": "gemma4:e2b via Ollama (open-weight only, PRD FR-31)" if syn else None,
            "seeds": seeds_n,
            "rule_signals": "hub.rules.rule_signals",
            "rule_signal_dropout_train": args.dropout,
            "minhash": f"char {SHINGLE}-grams, Jaccard >= {JACCARD}",
            "invalid_dropped": bad_syn + bad_real,
            "diversify": "placeholders->digits, scam refs 35%, safe phone senders, screenshot 15%",
            **stats,
        }
    }
    lock = update_lock(
        args.out / "splits.lock.json", entries, test_present=test_index is not None, extra=extra
    )

    print(f"train {len(train)}  dev {len(dev)}  provisional={lock['provisional']}  {stats}")
    for name in ("train", "dev"):
        print(f"[{name}] sha256 {entries[name]['sha256'][:12]}…")
        for key, c in entries[name]["counts"].items():
            print(f"  {key:<13} " + "  ".join(f"{k}={v}" for k, v in c.items()))
    for name, ex in (("train", train), ("dev", dev)):
        for v, r in shortcut_report(ex).items():
            print(
                f"[{name}] {v:<10} n={r['n']} channel={r['channel']} sender={r['sender_status']} digits={r['digit_run']}"
            )
    print(f"wrote {args.out / 'splits.lock.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
