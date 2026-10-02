"""Phase 1: latency baseline — N requests each to Ollama (explainer) and llama-server (detector).

    uv run python -m scripts.smoke.latency [-n 20]

One warm-up request per server is excluded. Synthetic inputs only. Run with both servers up
(Ollama via `brew services start ollama`, llama-server via qwen_gguf.sh).
"""

import argparse

from scripts.smoke._common import (
    SAMPLE_SCAM_FLAGS,
    banner,
    detector_chat,
    explainer_messages,
    finish,
    ollama_chat,
    pct,
    settings,
    verdict_line,
)

VERDICT = {"verdict": "SCAM", "category": "kyc_account_block", "red_flags": SAMPLE_SCAM_FLAGS}


def run(name: str, call, n: int) -> bool:
    banner(name)
    try:
        call()  # warm-up (model load), not counted
    except Exception as e:
        return verdict_line(False, name, f"{type(e).__name__}: {e}")
    times, errors = [], 0
    for _ in range(n):
        try:
            times.append(call())
        except Exception:
            errors += 1
    if times:
        print(
            f"n={len(times)}  p50 {pct(times, 50):.0f} ms  p95 {pct(times, 95):.0f} ms  "
            f"min {min(times):.0f}  max {max(times):.0f}"
        )
    return verdict_line(errors == 0, name, f"{errors} errors" if errors else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-n", type=int, default=20)
    args = ap.parse_args()
    s = settings()
    msgs = explainer_messages(VERDICT, "en", "Mom", 60)

    ok = run(f"Ollama {s.gemma_model} (explainer, en)", lambda: ollama_chat(msgs)[1], args.n)
    ok = (
        run(f"llama-server {s.detector_url} (detector A.1)", lambda: detector_chat()[1], args.n)
        and ok
    )
    finish(ok)


if __name__ == "__main__":
    main()
