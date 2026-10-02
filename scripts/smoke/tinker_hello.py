"""Phase 1: Tinker hello — sample one detector reply (PRD Appendix A.1) from DETECTOR_BASE.

    uv run --extra train python -m scripts.smoke.tinker_hello

Synthetic message only (CLAUDE.md: nothing private goes to Tinker). Verified against
tinker 0.32.0 / tinker-cookbook:
  ServiceClient(api_key=...)                       # falls back to $TINKER_API_KEY
  .get_server_capabilities().supported_models      # list of SupportedModel(model_name=...)
  .create_sampling_client(base_model=...)          # -> SamplingClient
  SamplingClient.get_tokenizer()
  renderers.get_renderer("qwen3_5_disable_thinking", tokenizer)   # thinking off, PRD §7.1
  renderer.build_generation_prompt(messages)       # -> tinker.ModelInput
  SamplingClient.sample(prompt=, num_samples=, sampling_params=).result()
  renderer.parse_response(seq.tokens)              # -> (Message, termination)
"""

import time

import tinker
from tinker_cookbook import renderers

from scripts.smoke._common import (
    SAMPLE_SCAM,
    banner,
    check_detector_json,
    detector_messages,
    finish,
    settings,
    verdict_line,
)

RENDERER = "qwen3_5_disable_thinking"


def main() -> None:
    s = settings()
    key = s.tinker_api_key.get_secret_value() if s.tinker_api_key else None
    print(
        f"base model {s.detector_base}  renderer {RENDERER}  key from {'.env' if key else '$TINKER_API_KEY'}"
    )

    try:
        service = tinker.ServiceClient(api_key=key)
        caps = service.get_server_capabilities()
        names = sorted(m.model_name for m in caps.supported_models if m.model_name)
        listed = s.detector_base in names
        verdict_line(listed, "base model listed by server", "" if listed else f"available: {names}")

        t0 = time.perf_counter()
        sampler = service.create_sampling_client(base_model=s.detector_base)
        renderer = renderers.get_renderer(RENDERER, sampler.get_tokenizer())
        prompt = renderer.build_generation_prompt(detector_messages())
        params = tinker.SamplingParams(
            max_tokens=256, temperature=0.0, seed=7, stop=renderer.get_stop_sequences()
        )
        resp = sampler.sample(prompt=prompt, num_samples=1, sampling_params=params).result()
        ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        verdict_line(False, "tinker sample", f"{type(e).__name__}: {e}")
        finish(False)

    msg, _ = renderer.parse_response(resp.sequences[0].tokens)
    text = renderers.get_text_content(msg).strip()
    banner("response (synthetic input)")
    print(text)
    print(f"\nround trip {ms:.0f} ms, {len(resp.sequences[0].tokens)} tokens")
    ok = verdict_line(bool(text), "tinker sample returned")
    json_ok, detail = check_detector_json(text, SAMPLE_SCAM)
    verdict_line(json_ok, "base model JSON (informational)", detail)
    finish(ok)


if __name__ == "__main__":
    main()
