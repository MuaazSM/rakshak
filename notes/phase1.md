# Phase 1 — local model checks

See IMPLEMENTATION.md Phase 1. Answers PRD §16 Q4 and Q5 (base model). Run on 3 Oct 2026 on
the M3 Pro (18 GB) with `scripts/smoke/`. Decisions are logged in PRD §17.

**All inputs were synthetic.** `data/raw/` had no real forwards yet, so the screenshots are
rendered SMS bubbles (English KYC scam, Hindi electricity-disconnect scam) with known text,
and the voice notes are macOS `say` (Lekha hi_IN, Rishi en_IN). TTS audio is cleaner than
a real voice note; re-check image and audio on one real sample each in Phase 6.

## Results

| Test | Result | Numbers | Fallback chosen |
|---|---|---|---|
| Gemma explanations | **PASS** for SCAM (en, hi); **FAIL** for SAFE format | 1.2–1.8 s with thinking off (12–14 s with it on). SAFE: 3 sentences incl. "Do not click anything" | Send `"think": false`; **fixed template for SAFE**, Gemma for SCAM / SUSPICIOUS |
| Gemma image → text | **PASS** | similarity 0.993 en / 0.996 hi, links exact, ~2.7 s. Tesseract: 0.910 / 0.415, links exact, 0.2–0.5 s | `OCR_FALLBACK=gemma` |
| Gemma audio | **PASS** (experimental in llama.cpp) | warm: 6.1 s hi (12 s note), 3.9 s en; similarity 0.981 hi / 0.995 en. mlx-whisper small: 1.9 s / 1.0 s, 0.945 / 1.000 | `ASR_FALLBACK=gemma`, `GEMMA_AUDIO_URL=http://127.0.0.1:8082/v1`; mlx-whisper ready if voice p95 > 12 s |
| Tinker hello | **PASS** | `Qwen/Qwen3.5-4B` listed by the server; sampling client + `qwen3_5_disable_thinking` renderer returned valid JSON (verdict SCAM, 50 tokens, 7.9 s round trip incl. client setup) | None needed; Phase 4 can start |
| GGUF conversion of base Qwen3.5-4B | **PASS** | convert 51 s (8.1 GB f16), Q4_K_M 54 s (2.6 GB), server healthy in 3 s, valid JSON, verdict SCAM | Plan A (GGUF); MLX-LM Plan B not needed. Merged LoRA still to test in Phase 5 |
| Qwen on Hindi / Hinglish | **PASS** (9/10) | Hinglish 5/5, Devanagari 4/5 (missed electricity disconnect), JSON 10/10 | More synthetic Devanagari scams (esp. `electricity_disconnect`) in Phase 3; report slices |
| Latency baseline | **PASS** | explainer p50 1242 / p95 1299 ms; detector p50 2964 / p95 3180 ms (n=20, warm) | Keep Q4_K_M; no smaller quant needed |

## Notes

- **Ollama version.** The installed Ollama.app was 0.12.10 and cannot pull `gemma4`. Now
  using brew Ollama 0.35.1 (`/opt/homebrew/opt/ollama/bin/ollama serve`). Remove or update
  the old app so `/usr/local/bin/ollama` doesn't shadow it.
- **Thinking.** Gemma 4 in Ollama thinks by default (~500 tokens per explanation). The
  explainer and perception calls must send `"think": false`. The detector already runs with
  `chat_template_kwargs.enable_thinking=false`.
- **Base detector output.** JSON is valid, but categories and reasons are free text
  ("Phishing", sentences) rather than the §10.1 vocabularies. That's expected before
  fine-tuning. Its long reasons are why the detector takes about 3 s; the tuned model's short
  output should be faster.
- **Latency vs NFR-2** (sum of warm p95s, without the hub's own overhead): text ≈ 4.5 s
  (≤ 6 s ✓); screenshot ≈ 7.2 s (≤ 12 s ✓); voice with Gemma ≈ 10.6 s for a 12 s note
  (≤ 12 s, tight). With mlx-whisper, voice is ≈ 6.4 s. Longer real notes may push Gemma over,
  so re-measure in Phase 6 and switch with `ASR_FALLBACK=mlx-whisper` if needed.
- **Memory (NFR-10).** Resident: Ollama gemma4:e2b 4.3 GB + Gemma audio llama-server 2.3 GB
  + detector Q4_K_M ≈ 3 GB ≈ 9.6 GB. That leaves headroom on 18 GB; free memory was 23% with
  all three up. Stop all three while merging the adapter.
- **Gemma audio setup.** `models/gemma-audio/` holds ggml-org `gemma-4-E2B-it-Q4_0.gguf` and
  `mmproj-gemma-4-E2B-it-Q8_0.gguf`. Start it with
  `llama-server -m models/gemma-audio/gemma-4-E2B-it-Q4_0.gguf --mmproj models/gemma-audio/mmproj-gemma-4-E2B-it-Q8_0.gguf --host 127.0.0.1 --port 8082 -ngl 99 --jinja -c 4096`.
- **Training deps.** `tinker-cookbook` 0.1.0 (the old lock) doesn't know Qwen3.5, so Phase 4
  needs ≥ 0.5.7. That caps `transformers` at 5.5.4, which supports Qwen3.5, so `mlx-lm` is
  pinned to 0.31.x (0.32 needs transformers ≥ 5.7). Resolved: tinker 0.32.0, tinker-cookbook
  0.5.7, transformers 5.5.4, peft 0.21.2, mlx-lm 0.31.3, mlx-whisper 0.4.3.
- **Raw outputs** are in `var/smoke/*.out` and transcripts in `data/raw/smoke/` (both
  git-ignored).

## Reproduce

```bash
uv run python -m scripts.smoke.gemma_text
uv run python -m scripts.smoke.gemma_image data/raw/smoke/inputs/sms-hi.png --expect data/raw/smoke/inputs/truth-hi.txt --tesseract
GEMMA_AUDIO_URL=http://127.0.0.1:8082/v1 uv run --extra mlx python -m scripts.smoke.gemma_audio data/raw/smoke/inputs/voice-hi.ogg --expect data/raw/smoke/inputs/voice-hi.txt --whisper --lang hi
uv run --extra train python -m scripts.smoke.tinker_hello
bash scripts/smoke/qwen_gguf.sh
uv run python -m scripts.smoke.latency
```
