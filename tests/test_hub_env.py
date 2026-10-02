"""Shared fixtures for the hub tests: isolated settings, parents, DB; fake services."""

import json
import math
from pathlib import Path

import pytest

from hub import detector, fusion, graph, settings
from hub.rules import RuleSignals
from hub.schemas import DetectorOutput

PARENTS = [
    {"id": "mom", "name": "Mom", "age": 60, "language": "en", "device": "android"},
    {"id": "dad", "name": "Dad", "age": 63, "language": "en", "device": "ios"},
]


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def hub_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Temp parents.json + DB; no .env, no ntfy, no Sentry; caches reset around the test."""
    parents = tmp_path / "parents.json"
    parents.write_text(json.dumps(PARENTS), "utf-8")
    monkeypatch.setitem(settings.Settings.model_config, "env_file", None)
    for var in ("NTFY_TOPIC", "SENTRY_DSN", "T_HIGH", "T_LOW", "GEMMA_AUDIO_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PARENTS_FILE", str(parents))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "var" / "rakshak.db"))
    monkeypatch.setenv("KEEP_MEDIA", "false")
    monkeypatch.setattr(fusion, "THRESHOLDS_FILE", tmp_path / "no-thresholds.json")
    for cache in (settings.get_settings, settings.get_parents, fusion.get_thresholds):
        cache.cache_clear()
    graph._GRAPH = None
    yield tmp_path
    for cache in (settings.get_settings, settings.get_parents, fusion.get_thresholds):
        cache.cache_clear()


def make_result(
    verdict: str = "SCAM",
    p_scam: float = 0.97,
    category: str = "kyc_account_block",
    flags: list[tuple[str, str]] | None = None,
    grounding: float | None = 1.0,
) -> detector.DetectorResult:
    flags = flags if flags is not None else [("account will be blocked", "urgency_deadline")]
    out = DetectorOutput.model_validate(
        {
            "verdict": verdict,
            "category": category,
            "red_flags": [{"quote": q, "reason": r} for q, r in flags],
        }
    )
    kept = [{"quote": q, "reason": r, "source": "model"} for q, r in flags]
    return detector.DetectorResult(out, p_scam, kept, grounding, "{}", 0)


def lp_entry(token: str, alts: dict[str, float], sampled_p: float | None = None) -> dict:
    """One llama-server logprob entry; `alts` maps token -> probability."""
    p = sampled_p if sampled_p is not None else alts.get(token, 1.0)
    return {
        "token": token,
        "logprob": math.log(p),
        "top_logprobs": [{"token": t, "logprob": math.log(q)} for t, q in alts.items()],
    }


class FakeRules:
    """Helper to build RuleSignals in fusion tests."""

    @staticmethod
    def make(hard=(), soft=(), evidence=None) -> RuleSignals:
        return RuleSignals(list(hard), list(soft), evidence or {})
