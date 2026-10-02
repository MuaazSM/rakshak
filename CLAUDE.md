# CLAUDE.md — working rules for agents in this repo

## What this is
Rakshak: an on-device scam guardian for the owner's parents — **Mom on Android** (PWA + Share Target) and **Dad on iPhone** (iOS Shortcuts), with explanations in **English** (Hindi exists only as a setting). A phone a suspicious message, screenshot or voice note to a FastAPI + LangGraph hub on a MacBook M3 Pro; a fine-tuned Qwen3.5-4B (llama.cpp) decides SCAM / SUSPICIOUS / SAFE, a rules engine backs it up, and Gemma 4 E2B (Ollama) reads images/audio and explains the verdict in the parent's language. Built for the Hacktoberfest 2026 DEV Weekend Challenge ("Build for a Friend").

## Source of truth
1. **`docs/PRD.md`** — requirements, schemas, prompts, thresholds, targets. **Always wins.** (`docs/PRD.pdf` is an export of it.)
2. `docs/IMPLEMENTATION.md` — phases, env, setup, exit criteria.
3. `docs/PROMPTBOOK.md` — prompts for building each phase.

`docs/` is **git-ignored**: planning docs stay local and are never committed or pushed. Bare references like "PRD §7.3" or "IMPLEMENTATION.md Phase 1" anywhere in the repo mean the files in `docs/`. Never paste PRD text into committed files beyond what code needs (e.g. prompts in `hub/prompts.py`).

Before implementing anything, read the PRD sections the task cites. If a task conflicts with the PRD, stop and say so; don't improvise. If the PRD is silent, choose the simplest option and record it in the PRD decision log (§17) in the same change.

## Hard rules (privacy and safety)
- **Never** commit or print the contents of `data/raw/`, `var/`, `media/`, `*.db`, `data/redaction_names.txt`, `config/parents.json`, `.env`, or `docs/`.
- **Never** write anything in the post, README or docs that presents a scam incident in the family as real. The story is prevention (PRD §1).
- **Never** log, trace, or send message text, quotes, explanations, images or audio to any external service (Sentry, ntfy, Tinker at runtime, any HTTP API). Metadata only (PRD §9.3).
- Only redacted (`data/redacted/`) or synthetic data may be uploaded to Tinker.
- Runtime inference is local only: Ollama (`OLLAMA_HOST`) and llama-server (`DETECTOR_URL`). Do not add cloud LLM calls.
- The hub binds to `127.0.0.1`; external access is only via `tailscale serve`.
- The explainer must never change the verdict. Red flags shown to users must be exact substrings of the input.
- If the detector is unavailable, return `UNKNOWN` ("couldn't check"), never `SAFE`.
- **Never** evaluate on, tune on, or look at the test set except in the single final eval step. Thresholds come from dev only.

## Stack
- Python 3.11+, managed with **uv** (`uv add`, `uv run`). No pip/conda.
- FastAPI, LangGraph, httpx, pydantic v2, sentry-sdk, datasketch, rapidfuzz, tldextract, idna, pytesseract, numpy/scipy.
- Training: `tinker`, `tinker-cookbook`, `transformers`, `peft`, `huggingface_hub`. Fallback: `mlx-lm`, `mlx-whisper`.
- PWA: React + Vite + TypeScript in `pwa/`, plain CSS. Built output served by FastAPI from `pwa/dist`.
- Models: Ollama `gemma4:e2b`; llama-server serving `models/rakshak-detector-v*-q4km.gguf` on `:8081`.

## Repository layout
```
rakshak/
├── CLAUDE.md  README.md
├── docs/               # git-ignored: PRD.md PRD.pdf IMPLEMENTATION.md PROMPTBOOK.md
│   └── design/         # Claude Design export (HTML frames, read-only reference)
├── .env.example  .gitignore  pyproject.toml
├── config/thresholds.json  parents.example.json  (parents.json git-ignored)
├── data/
│   ├── raw/            # git-ignored
│   ├── redacted/
│   ├── synthetic/
│   └── splits/         # train.jsonl dev.jsonl test.jsonl splits.lock.json
├── training/           # redact.py synth.py build_dataset.py train_tinker.py export.py runs/
├── eval/               # run_eval.py baselines.py calibrate.py bootstrap.py fewshot.jsonl results/
├── hub/
│   ├── app.py graph.py normalize.py perception.py rules.py detector.py fusion.py
│   ├── explainer.py alerts.py tracing.py db.py settings.py schemas.py
│   └── data/           # official_domains.txt shorteners.txt lexicon/ verdict_words.json templates/
├── pwa/                # React + Vite PWA
├── brand/              # logo SVGs/PNGs, cover, hand-over cards (outside layer)
├── models/             # git-ignored GGUFs and base weights
├── notes/              # phase1.md failures.md
├── tests/
└── var/                # git-ignored runtime DB and media
```

## Commands
```bash
uv sync
uv run pytest -q
uv run ruff check . && uv run ruff format .
uv run uvicorn hub.app:app --host 127.0.0.1 --port 8000 --reload
uv run python -m eval.run_eval --all          # writes eval/results/*.json
uv run python -m eval.calibrate --split dev   # writes config/thresholds.json
cd pwa && npm run build
```

## Conventions
- All config via `hub/settings.py` (pydantic-settings reading `.env`) and `config/parents.json`. No hard-coded keys, paths, names or languages.
- Every request carries `parent_id` (PRD FR-7). Explanation language comes from the parent profile, default `en` (FR-8); Hindi is a setting only — never add Hindi copy, toggles or badges to screens or docs.
- Schemas in `hub/schemas.py` must match PRD §8 exactly (field names and enums).
- Prompts live in `hub/prompts.py` copied verbatim from PRD Appendix A, with a comment `# canonical: PRD Appendix A.x`. Change the PRD first, then the copy.
- Every pipeline node records its latency into `state["timings_ms"]` and gets a Sentry span.
- Rules engine is pure functions with unit tests; no I/O, no model calls.
- Eval scripts are deterministic (fixed seeds), write JSON, never print raw message text from `test.jsonl`.
- Small, focused commits (see "Git and GitHub"). Note any commit after Mon 5 Oct 06:59 UTC in README.

## Design rules (PRD §18)
- Use tokens from `pwa/src/styles/tokens.css` only; no new colors. Never use crimson/maroon in the app; red means SCAM.
- Never put text in `--haldi` on light backgrounds.
- Fonts: Mukta (UI), Yatra One (wordmark only), JetBrains Mono (status numbers). Load via Google Fonts with `font-display: swap`.
- Every verdict shows color + icon + word. Tap targets ≥ 56px; body ≥ 18px; Hindi line-height 1.6.
- UI strings come from `pwa/src/i18n/en.json` (default) / `hi.json`; never hard-code copy in components.
- Visual source of truth: the Claude Design HTML in `docs/design/` (one file per frame group: A design system, B Mom/Android, C Dad/iPhone, D son/status, E outside layer). Open the matching frame before building or changing any screen and match it exactly; if a screen isn't designed, follow the nearest designed pattern and say so.
- Treat `docs/design/` as read-only reference. Don't import its HTML/CSS into `pwa/` directly; rebuild as React components using `tokens.css`.
- Brand files live in `brand/`; don't redraw or recolor the logo.

## Git and GitHub
- Remote: `https://github.com/MuaazSM/rakshak` (public), branch `main`. Remember it's public when deciding what to commit.
- **Push only as `MuaazSM`.** Before any push, PR or other `gh` command, run `gh auth status`; if the active account is `MuaazHivePro` (or anything else), run `gh auth switch -u MuaazSM` first.
- Commit identity is set repo-locally: `user.name "Muaaz Shaikh"`, `user.email "170370120+MuaazSM@users.noreply.github.com"`. Never commit with the work email or global identity; check `git config user.email` if in doubt.
- **Conventional Commits only**: `type(scope): summary` with type ∈ `feat fix docs test refactor perf build ci chore style revert`, optional scope (e.g. `hub`, `rules`, `eval`, `training`, `pwa`), imperative, lower-case, no trailing period, ≤ 72 chars. Body optional; `!` / `BREAKING CHANGE:` for breaking changes.
- **No AI attribution**: no `Co-Authored-By: Claude …` trailer, no "Generated with Claude Code" line, in commits or PR descriptions.

## Definition of done (any task)
- Matches the PRD section it implements; tests added/updated and passing; ruff clean.
- No secrets or private data in the diff (`git diff --cached | grep -i -E "otp|<PHONE>|$RAKSHAK_CANARY"` sanity check).
- If behavior or a decision changed, PRD §17 updated.

## Time box
Deadline Mon 5 Oct 12:29 IST; feature freeze Sun 15:00 IST. Prefer working and measured over complete. When a task balloons, stop, say what's blocking, and propose the fallback from `docs/IMPLEMENTATION.md`.
