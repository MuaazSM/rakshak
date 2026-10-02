"""Phase 1: Gemma audio → text (PRD Appendix A.2 audio prompt) via llama-server, with
mlx-whisper as the fallback (ASR_FALLBACK).

    uv run --extra mlx python -m scripts.smoke.gemma_audio data/raw/<voice-note>.ogg
        [--expect data/raw/<truth>.txt] [--whisper] [--whisper-model REPO] [--lang hi] [--show]

GEMMA_AUDIO_URL must point at a llama-server (OpenAI-compatible, e.g.
http://127.0.0.1:8082/v1) started with a Gemma 4 E2B GGUF and an mmproj that includes the
audio encoder, e.g.:
    llama-server -m <gemma-4-e2b>.gguf --mmproj <mmproj-with-audio>.gguf --port 8082 -ngl 99

The file is converted to 16 kHz mono WAV with ffmpeg and sent as an `input_audio` part.
If that fails, the error is printed and mlx-whisper runs on the same file. --whisper runs
mlx-whisper even when Gemma works, for comparison. Transcripts go to data/raw/smoke/
(git-ignored) and are printed only with --show.
"""

import argparse
import base64
import subprocess
import tempfile
import time
from pathlib import Path

import httpx

from hub import prompts
from scripts.smoke._common import (
    ROOT,
    banner,
    finish,
    private_note,
    save_private,
    settings,
    similarity,
    verdict_line,
)


def to_wav(src: Path, dst: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(dst),
        ],
        check=True,
    )


def report(name: str, text: str, ms: float, stem: str, truth: str | None, show: bool) -> None:
    out = save_private(f"audio-{stem}-{name}.txt", text)
    print(
        f"{name}: latency {ms:.0f} ms, {len(text)} chars, {len(text.split())} words, "
        f"saved {out.relative_to(ROOT)}"
    )
    if truth is not None:
        print(f"{name}: similarity to truth {similarity(text, truth):.3f}")
    if show:
        print("--- transcript ---\n" + text + "\n------------------")


def gemma(wav: Path) -> tuple[str, float]:
    url = settings().gemma_audio_url
    if not url:
        raise RuntimeError("GEMMA_AUDIO_URL is not set in .env")
    body = {
        "model": "gemma-audio",
        "temperature": 0,
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompts.PERCEIVE_AUDIO},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(wav.read_bytes()).decode(),
                            "format": "wav",
                        },
                    },
                ],
            }
        ],
    }
    t0 = time.perf_counter()
    r = httpx.post(f"{url.rstrip('/')}/chat/completions", json=body, timeout=300)
    ms = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    text = (r.json()["choices"][0]["message"]["content"] or "").strip()
    if not text:
        raise RuntimeError("empty transcript")
    return text, ms


def whisper(wav: Path, repo: str, lang: str | None) -> tuple[str, float]:
    import mlx_whisper

    t0 = time.perf_counter()
    out = mlx_whisper.transcribe(str(wav), path_or_hf_repo=repo, language=lang)
    return out["text"].strip(), (time.perf_counter() - t0) * 1000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", type=Path)
    ap.add_argument("--expect", type=Path, help="true transcript (keep in data/raw/)")
    ap.add_argument("--whisper", action="store_true", help="always run mlx-whisper too")
    ap.add_argument("--whisper-model", default="mlx-community/whisper-small-mlx")
    ap.add_argument(
        "--lang", default=None, help="whisper language hint, e.g. hi or en (default: auto)"
    )
    ap.add_argument("--show", action="store_true", help="print transcripts (private!)")
    args = ap.parse_args()
    private_note(args.path)
    truth = args.expect.read_text("utf-8") if args.expect else None

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "in.wav"
        to_wav(args.path, wav)
        print(f"audio {args.path.suffix} → 16 kHz mono WAV, {wav.stat().st_size // 1024} KB")

        banner(f"Gemma via llama-server ({settings().gemma_audio_url or 'unset'})")
        gemma_ok = False
        try:
            text, ms = gemma(wav)
            report("gemma", text, ms, args.path.stem, truth, args.show)
            gemma_ok = verdict_line(True, "gemma audio", "check the saved transcript is usable")
        except Exception as e:
            verdict_line(False, "gemma audio", f"{type(e).__name__}: {e}")

        whisper_ok = None
        if not gemma_ok or args.whisper:
            banner(f"mlx-whisper ({args.whisper_model})")
            try:
                text, ms = whisper(wav, args.whisper_model, args.lang)
                report("whisper", text, ms, args.path.stem, truth, args.show)
                whisper_ok = verdict_line(
                    bool(text), "mlx-whisper", "first run includes model download"
                )
            except Exception as e:
                whisper_ok = verdict_line(False, "mlx-whisper", f"{type(e).__name__}: {e}")

    if gemma_ok:
        print("\nSuggested: ASR_FALLBACK=gemma (if the transcript is usable)")
    elif whisper_ok:
        print("\nSuggested: ASR_FALLBACK=mlx-whisper, GEMMA_AUDIO_URL blank")
    finish(gemma_ok)


if __name__ == "__main__":
    main()
