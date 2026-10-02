"""Phase 1: Gemma explainer (PRD Appendix A.3) on fake verdicts, via Ollama.

    uv run python -m scripts.smoke.gemma_text [--name Mom --age 60]

Runs a SCAM and a SAFE verdict in English (default) and one SCAM sample in Hindi (the
setting). All inputs are synthetic, so the explanations are printed in full.
Format checks follow A.3: ≤ 3 sentences, < 60 words, no links or numbers.
"""

import argparse
import re

from scripts.smoke._common import (
    SAMPLE_SCAM_FLAGS,
    banner,
    explainer_messages,
    finish,
    ollama_chat,
    ollama_timing,
    settings,
    verdict_line,
)

SCAM = {"verdict": "SCAM", "category": "kyc_account_block", "red_flags": SAMPLE_SCAM_FLAGS}
SAFE = {"verdict": "SAFE", "category": "delivery_update", "red_flags": []}
CASES = [("SCAM", "en", SCAM), ("SAFE", "en", SAFE), ("SCAM", "hi", SCAM)]


def format_problems(text: str, verdict: str, lang: str) -> list[str]:
    problems = []
    sentences = [s for s in re.split(r"[.!?।]+", text) if s.strip()]
    max_sentences = 1 if verdict == "SAFE" else 3
    if len(sentences) > max_sentences:
        problems.append(f"{len(sentences)} sentences (max {max_sentences})")
    if len(text.split()) >= 60:
        problems.append(f"{len(text.split())} words (max 59)")
    if re.search(r"https?://|www\.|\.example", text):
        problems.append("contains a link")
    if re.search(r"[0-9०-९]", text):
        problems.append("contains a number")
    if lang == "hi" and not re.search(r"[ऀ-ॿ]", text):
        problems.append("no Devanagari")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="Mom", help="placeholder parent name (not real data)")
    ap.add_argument("--age", type=int, default=60)
    ap.add_argument("--think", action="store_true", help="leave Gemma thinking on")
    args = ap.parse_args()
    s = settings()
    print(f"Ollama {s.ollama_host}  model {s.gemma_model}  son={s.son_name}")

    ok = True
    for verdict, lang, vj in CASES:
        banner(f"{verdict} / {lang}")
        try:
            resp, ms = ollama_chat(
                explainer_messages(vj, lang, args.name, args.age), think=args.think
            )
        except Exception as e:
            ok = verdict_line(False, f"{verdict}/{lang}", f"{type(e).__name__}: {e}") and ok
            continue
        text = resp["message"]["content"].strip()
        if resp["message"].get("thinking"):
            print("warn: model produced a thinking block (adds latency)")
        print(text)
        print(f"latency {ms:.0f} ms ({ollama_timing(resp)})")
        problems = format_problems(text, verdict, lang)
        ok = verdict_line(not problems, f"{verdict}/{lang} format", "; ".join(problems)) and ok

    print("\nJudge by eye too: would a parent say this back to you, calmly?")
    finish(ok)


if __name__ == "__main__":
    main()
