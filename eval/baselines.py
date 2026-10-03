"""Baselines: rules only, Gemma zero-shot, Qwen base few-shot (PRD §12.3, Appendix A.4).

    uv run python -m eval.baselines --build-fewshot     # writes eval/fewshot.jsonl from TRAIN
    uv run python -m eval.baselines                     # runs the three baselines on dev

Few-shot set (A.4): 6 fixed examples drawn from the train split only — 2 scam, 2 genuine hard
negatives, 1 suspicious, 1 safe personal. Choice is deterministic (fixed seed over id-sorted
candidates); clean (non-obfuscated), short items are preferred, the two scams differ in
category (and language when possible), and the hard negatives are one `genuine_otp` and one
`transaction_alert` when both exist. Items are interleaved so no class sits last. Each line is
`{"id", "seed_group", "kind", "messages": [user, assistant]}`; prints ids and kinds only.
"""

import argparse
import json
import random
import sys
from pathlib import Path

from eval.metrics import is_hard_negative
from eval.systems import FEWSHOT_PATH, Item, load_items

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "splits" / "train.jsonl"
SEED = 20261003
MAX_CHARS = 400
ORDER = ("scam", "hard_negative", "suspicious", "scam", "safe_personal", "hard_negative")
BASELINES = "rules_only,gemma_zeroshot,qwen_base_fewshot"


def _pool(items: list[Item], pred) -> list[Item]:
    matching = sorted((it for it in items if pred(it)), key=lambda it: it.id)
    clean = [it for it in matching if not it.meta.get("obfuscated") and len(it.text) <= MAX_CHARS]
    return clean or matching


def _pick(rng: random.Random, pool: list[Item], taken: set[str]) -> Item | None:
    free = [it for it in pool if it.id not in taken]
    return rng.choice(free) if free else None


def select_fewshot(items: list[Item], seed: int = SEED) -> list[tuple[str, Item]]:
    """Return [(kind, item)] in prompt order. Raises ValueError if a kind has no candidate."""
    rng = random.Random(seed)
    taken: set[str] = set()
    hard = lambda it: is_hard_negative(it.meta, it.gold.verdict, it.gold.category)  # noqa: E731

    scam1 = _pick(rng, _pool(items, lambda it: it.gold.verdict == "SCAM"), taken)
    if scam1 is None:
        raise ValueError("no SCAM item in train")
    taken.add(scam1.id)
    lang1 = scam1.meta.get("language")
    scam2 = _pick(
        rng,
        _pool(
            items,
            lambda it: (
                it.gold.verdict == "SCAM"
                and it.gold.category != scam1.gold.category
                and it.meta.get("language") != lang1
            ),
        ),
        taken,
    ) or _pick(
        rng,
        _pool(
            items,
            lambda it: it.gold.verdict == "SCAM" and it.gold.category != scam1.gold.category,
        ),
        taken,
    )
    chosen: dict[str, list[Item]] = {"scam": [scam1] + ([scam2] if scam2 else [])}
    if scam2:
        taken.add(scam2.id)

    hn = []
    for cat in ("genuine_otp", "transaction_alert"):
        it = _pick(rng, _pool(items, lambda x, c=cat: hard(x) and x.gold.category == c), taken)
        if it:
            hn.append(it)
            taken.add(it.id)
    while len(hn) < 2:
        it = _pick(rng, _pool(items, hard), taken)
        if it is None:
            break
        hn.append(it)
        taken.add(it.id)
    chosen["hard_negative"] = hn

    sus = _pick(rng, _pool(items, lambda it: it.gold.verdict == "SUSPICIOUS"), taken)
    chosen["suspicious"] = [sus] if sus else []
    if sus:
        taken.add(sus.id)
    per = _pick(
        rng,
        _pool(items, lambda it: it.gold.verdict == "SAFE" and it.gold.category == "personal"),
        taken,
    )
    chosen["safe_personal"] = [per] if per else []

    need = {k: ORDER.count(k) for k in set(ORDER)}
    missing = {k: n - len(chosen[k]) for k, n in need.items() if len(chosen[k]) < n}
    if missing:
        raise ValueError(f"train has too few candidates for few-shot kinds: {missing}")
    cursor = dict.fromkeys(need, 0)
    out = []
    for kind in ORDER:
        out.append((kind, chosen[kind][cursor[kind]]))
        cursor[kind] += 1
    return out


def fewshot_lines(selected: list[tuple[str, Item]]) -> list[dict]:
    return [
        {
            "id": it.id,
            "seed_group": it.meta.get("seed_group"),
            "kind": kind,
            "messages": [m for m in it.messages if m["role"] in ("user", "assistant")],
        }
        for kind, it in selected
    ]


def build_fewshot(train: Path = TRAIN, out: Path = FEWSHOT_PATH) -> list[dict]:
    if train.name == "test.jsonl" or "test" in train.stem:
        raise ValueError("few-shot examples must come from train only")
    lines = fewshot_lines(select_fewshot(load_items(train)))
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines), "utf-8")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build-fewshot", action="store_true")
    ap.add_argument("--train", type=Path, default=TRAIN)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    if args.build_fewshot:
        lines = build_fewshot(args.train)
        for x in lines:
            print(f"{x['kind']:<14} {x['id']}")
        print(f"wrote {FEWSHOT_PATH.relative_to(ROOT)} ({len(lines)} examples)")
        return 0
    from eval.run_eval import main as run_eval

    extra = ["--limit", str(args.limit)] if args.limit else []
    return run_eval(["--systems", BASELINES, "--split", "dev", *extra])


if __name__ == "__main__":
    sys.exit(main())
