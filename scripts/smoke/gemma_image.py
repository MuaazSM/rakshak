"""Phase 1: Gemma image → text (PRD Appendix A.2 image prompt), via Ollama.

    uv run python -m scripts.smoke.gemma_image data/raw/<screenshot>.png
        [--expect data/raw/<truth>.txt] [--tesseract] [--show]

The transcript of a real screenshot is saved to data/raw/smoke/ (git-ignored) and is
printed only with --show. Without --show the output is metadata only, so it is safe to
paste back. --expect takes the hand-typed true text (also kept in data/raw/) and scores
similarity and exact links. --tesseract runs the OCR fallback on the same image to compare.
"""

import argparse
import base64
import re
import time
from pathlib import Path

from hub import prompts
from scripts.smoke._common import (
    ROOT,
    banner,
    finish,
    links,
    ollama_chat,
    ollama_timing,
    private_note,
    save_private,
    settings,
    similarity,
    verdict_line,
)


def score(name: str, text: str, truth: str | None) -> bool:
    """Print metadata (never the text). Returns True if links are exact (when truth given)."""
    found = links(text)
    print(f"{name}: {len(text)} chars, {len(text.split())} words, {len(found)} links")
    if truth is None:
        return True
    want = links(truth)
    exact = sorted(found) == sorted(want)
    print(
        f"{name}: similarity to truth {similarity(text, truth):.3f}; links exact: {exact} "
        f"({len(set(found) & set(want))}/{len(want)} matched)"
    )
    return exact


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", type=Path)
    ap.add_argument("--expect", type=Path, help="true text of the screenshot (keep in data/raw/)")
    ap.add_argument("--tesseract", action="store_true", help="also run the Tesseract fallback")
    ap.add_argument("--show", action="store_true", help="print transcripts (private!)")
    args = ap.parse_args()
    private_note(args.path)
    truth = args.expect.read_text("utf-8") if args.expect else None
    s = settings()
    print(
        f"Ollama {s.ollama_host}  model {s.gemma_model}  image {args.path.suffix} "
        f"{args.path.stat().st_size // 1024} KB"
    )

    banner("Gemma")
    img = base64.b64encode(args.path.read_bytes()).decode()
    try:
        resp, ms = ollama_chat(
            [{"role": "user", "content": prompts.PERCEIVE_IMAGE, "images": [img]}]
        )
    except Exception as e:
        verdict_line(False, "gemma image", f"{type(e).__name__}: {e}")
        finish(False)
    text = resp["message"]["content"].strip()
    out = save_private(f"image-{args.path.stem}-gemma.txt", text)
    print(f"latency {ms:.0f} ms ({ollama_timing(resp)})  saved {out.relative_to(ROOT)}")
    has_format = bool(re.search(r"^SENDER:", text, re.M)) and bool(
        re.search(r"^MESSAGE:\s*\n\s*\S", text, re.M)
    )
    links_ok = score("gemma", text, truth)
    if args.show:
        print("--- transcript ---\n" + text + "\n------------------")
    ok = verdict_line(
        has_format, "gemma A.2 output format", "" if has_format else "missing SENDER:/MESSAGE:"
    )
    if truth is not None:
        ok = verdict_line(links_ok, "gemma links exact") and ok

    if args.tesseract:
        banner("Tesseract (OCR_FALLBACK)")
        try:
            import pytesseract
            from PIL import Image

            t0 = time.perf_counter()
            ocr = pytesseract.image_to_string(Image.open(args.path), lang="hin+eng").strip()
            tms = (time.perf_counter() - t0) * 1000
            tout = save_private(f"image-{args.path.stem}-tesseract.txt", ocr)
            print(f"latency {tms:.0f} ms  saved {tout.relative_to(ROOT)}")
            score("tesseract", ocr, truth)
            if args.show:
                print("--- transcript ---\n" + ocr + "\n------------------")
        except Exception as e:
            print(f"tesseract failed: {type(e).__name__}: {e}")

    if truth is None:
        print("\nNo --expect given: open the saved file and check text and links by eye.")
    finish(ok)


if __name__ == "__main__":
    main()
