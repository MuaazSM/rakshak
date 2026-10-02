import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from hub.settings import Settings, load_parents

ROOT = Path(__file__).resolve().parent.parent


def test_example_parents_load():
    parents = load_parents(ROOT / "config/parents.example.json")
    assert set(parents) == {"mom", "dad"}
    assert parents["mom"].device == "android"
    assert parents["dad"].device == "ios"


@pytest.mark.parametrize(
    "patch",
    [
        {"language": "mr"},
        {"device": "windows"},
        {"id": "Mom!"},
        {"age": -1},
        {"unexpected": True},
    ],
)
def test_invalid_parent_rejected(tmp_path, patch):
    p = tmp_path / "parents.json"
    entry = {"id": "mom", "name": "Mom", "age": 0, "language": "hi", "device": "android"}
    p.write_text(json.dumps([entry | patch]))
    with pytest.raises(ValidationError):
        load_parents(p)


def test_duplicate_and_empty_parents_rejected(tmp_path):
    entry = {"id": "mom", "name": "Mom", "age": 0, "language": "hi", "device": "android"}
    p = tmp_path / "parents.json"
    p.write_text(json.dumps([entry, entry]))
    with pytest.raises(ValueError, match="duplicate"):
        load_parents(p)
    p.write_text("[]")
    with pytest.raises(ValueError, match="no parents"):
        load_parents(p)


def test_settings_cover_env_example(monkeypatch):
    env_example = ROOT / ".env.example"
    keys = [
        line.split("=", 1)[0].strip()
        for line in env_example.read_text().splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    ]
    fields = {name.upper() for name in Settings.model_fields}
    assert set(keys) <= fields, f"missing settings: {set(keys) - fields}"

    for k in keys:
        monkeypatch.delenv(k, raising=False)
    s = Settings(_env_file=env_example)
    assert s.hub_host == "127.0.0.1"
    assert s.t_high is None and s.t_low is None
    assert s.default_lang == "en"
    assert s.keep_media is False


def test_hub_host_never_all_interfaces():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, hub_host="0.0.0.0")
