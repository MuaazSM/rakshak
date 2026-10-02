"""Hub configuration (CLAUDE.md "Conventions"; PRD §2, FR-7, FR-8, §7.4, §9).

Every variable in `.env.example` is a field on `Settings`. Parent profiles come from
`config/parents.json` (PRD §2) and are validated by `load_parents`.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, TypeAdapter, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Lang = Literal["hi", "en"]
Device = Literal["android", "ios"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # blank values in .env mean "use the default"
        extra="ignore",
    )

    # Phase 0: accounts (training only; never used at runtime on parent data)
    tinker_api_key: SecretStr | None = None
    hf_token: SecretStr | None = None

    # Phase 1: local models
    ollama_host: str = "http://127.0.0.1:11434"
    gemma_model: str = "gemma4:e2b"
    gemma_audio_url: str | None = None
    asr_fallback: Literal["gemma", "mlx-whisper"] = "mlx-whisper"
    ocr_fallback: Literal["gemma", "tesseract"] = "tesseract"

    # Phase 4: training
    detector_base: str = "Qwen/Qwen3.5-4B"

    # Phase 5: detector + hub
    detector_url: str = "http://127.0.0.1:8081/v1"
    detector_gguf: Path = Path("models/rakshak-detector-v1-q4km.gguf")
    detector_version: str = "rakshak-detector-v1-q4km"
    t_high: float | None = Field(default=None, ge=0.0, le=1.0)
    t_low: float | None = Field(default=None, ge=0.0, le=1.0)
    hub_host: str = "127.0.0.1"
    hub_port: int = 8000
    db_path: Path = Path("var/rakshak.db")
    keep_media: bool = False

    # Phase 6: people, phone, alerts, tracing
    parents_file: Path = Path("config/parents.json")
    default_lang: Lang = "hi"
    son_name: str = "Muaaz"
    public_base_url: str | None = None
    ntfy_server: str = "https://ntfy.sh"
    ntfy_topic: str | None = None
    sentry_dsn: str | None = None
    sentry_env: str = "weekend"

    # Phase 7: privacy test
    rakshak_canary: str = "CANARY-change-me"

    # Phase 8 (optional): demo narration only
    elevenlabs_api_key: SecretStr | None = None

    @field_validator("hub_host")
    @classmethod
    def _localhost_only(cls, v: str) -> str:
        # CLAUDE.md: the hub binds to 127.0.0.1; external access only via `tailscale serve`.
        if v == "0.0.0.0":
            raise ValueError("HUB_HOST must never be 0.0.0.0")
        return v


class Parent(BaseModel):
    """One parent profile (PRD §2)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    name: str = Field(min_length=1)
    age: int = Field(ge=0)
    language: Lang
    device: Device


_parents_adapter = TypeAdapter(list[Parent])


def load_parents(path: Path | str) -> dict[str, Parent]:
    """Load and validate parent profiles, keyed by id. Raises ValueError on bad input."""
    parents = _parents_adapter.validate_python(json.loads(Path(path).read_text("utf-8")))
    if not parents:
        raise ValueError(f"{path}: no parents defined")
    ids = [p.id for p in parents]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"{path}: duplicate parent ids {dupes}")
    return {p.id: p for p in parents}


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_parents() -> dict[str, Parent]:
    return load_parents(get_settings().parents_file)
