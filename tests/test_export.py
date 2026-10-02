"""Export plan and steps (PRD FR-34, §11.4). No Tinker, no merge of real weights, no llama.cpp."""

import json
from pathlib import Path

import pytest

from training import export as E


def args(tmp_path: Path, *extra: str):
    return E.parse_args(
        [
            "--checkpoint",
            "tinker://run/sampler_weights/e3",
            "--version",
            "2",
            "--merged-root",
            str(tmp_path / "merged"),
            "--out",
            str(tmp_path / "rakshak-detector-v2-q4km.gguf"),
            "--llama-cpp",
            str(tmp_path / "llama.cpp"),
            "--base",
            str(tmp_path / "base"),
            *extra,
        ]
    )


def test_plan_paths_and_commands(tmp_path):
    plan = E.build_plan(args(tmp_path))
    assert plan.merged_dir == tmp_path / "merged" / "v2"
    assert plan.merge_adapter_dir == tmp_path / "merged" / "v2-adapter" / "no_unembed"
    assert E.convert_cmd(plan) == [
        str(tmp_path / "llama.cpp" / ".venv" / "bin" / "python"),
        str(tmp_path / "llama.cpp" / "convert_hf_to_gguf.py"),
        str(tmp_path / "merged" / "v2"),
        "--outfile",
        str(tmp_path / "merged" / "rakshak-detector-v2-f16.gguf"),
        "--outtype",
        "f16",
    ]
    assert E.quantize_cmd(plan) == [
        "llama-quantize",
        str(plan.f16),
        str(tmp_path / "rakshak-detector-v2-q4km.gguf"),
        "Q4_K_M",
    ]
    tied = E.build_plan(args(tmp_path, "--unembed", "tied"))
    assert tied.merge_adapter_dir == tied.adapter_dir


def test_default_output_name():
    a = E.parse_args(["--checkpoint", "tinker://r/sampler_weights/x", "--version", "3"])
    assert E.build_plan(a).out.name == "rakshak-detector-v3-q4km.gguf"


def test_checkpoint_from_run_prefers_best(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "best": {"epoch": 2, "checkpoint": "tinker://r/sampler_weights/e2"},
                "checkpoints": [{"sampler_path": "tinker://r/sampler_weights/e3"}],
            }
        )
    )
    assert E.checkpoint_from_run(tmp_path).endswith("e2")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"best": None, "checkpoints": [{"sampler_path": "tinker://r/sampler_weights/e1"}]}
        )
    )
    assert E.checkpoint_from_run(tmp_path).endswith("e1")
    (tmp_path / "config.json").write_text(json.dumps({"checkpoints": []}))
    with pytest.raises(E.ExportError):
        E.checkpoint_from_run(tmp_path)


def test_plan_rejects_bad_inputs(tmp_path):
    with pytest.raises(E.ExportError):
        E.build_plan(E.parse_args(["--version", "1"]))
    with pytest.raises(E.ExportError):
        E.build_plan(E.parse_args(["--checkpoint", "s3://x", "--version", "1"]))


def _fake_tools(tmp_path: Path, monkeypatch) -> None:
    llama = tmp_path / "llama.cpp"
    (llama / ".venv" / "bin").mkdir(parents=True)
    (llama / "convert_hf_to_gguf.py").write_text("")
    (llama / ".venv" / "bin" / "python").write_text("")
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    (base / "model.safetensors").write_text("")
    monkeypatch.setattr(E.shutil, "which", lambda name: f"/opt/homebrew/bin/{name}")


def test_dry_run_writes_nothing_and_runs_nothing(tmp_path, monkeypatch, capsys):
    _fake_tools(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(E.subprocess, "run", lambda *a, **k: calls.append(a))
    before = sorted(p for p in tmp_path.rglob("*"))
    assert E.main([*_argv(tmp_path), "--dry-run", "--min-free-gb", "0"]) == 0
    out = capsys.readouterr().out
    assert "PASS  preflight" in out and "llama-quantize" in out and "Q4_K_M" in out
    assert calls == [] and sorted(p for p in tmp_path.rglob("*")) == before


def test_preflight_fails_on_disk_and_existing_output(tmp_path, monkeypatch, capsys):
    _fake_tools(tmp_path, monkeypatch)
    assert E.main([*_argv(tmp_path), "--dry-run", "--min-free-gb", "1e9"]) == 1
    assert "need ≥" in capsys.readouterr().out
    (tmp_path / "rakshak-detector-v2-q4km.gguf").write_text("x")
    assert E.main([*_argv(tmp_path), "--dry-run", "--min-free-gb", "0"]) == 1
    assert "already exists" in capsys.readouterr().out


def _argv(tmp_path):
    return [
        "--checkpoint",
        "tinker://run/sampler_weights/e3",
        "--version",
        "2",
        "--merged-root",
        str(tmp_path / "merged"),
        "--out",
        str(tmp_path / "rakshak-detector-v2-q4km.gguf"),
        "--llama-cpp",
        str(tmp_path / "llama.cpp"),
        "--base",
        str(tmp_path / "base"),
    ]


def test_convert_and_quantize_call_tools_and_fail_fast(tmp_path, monkeypatch):
    plan = E.build_plan(args(tmp_path))
    seen = []

    class R:
        returncode = 0

    def fake_run(cmd, check=False):
        seen.append(cmd)
        Path(cmd[4] if "--outfile" in cmd else cmd[2]).parent.mkdir(parents=True, exist_ok=True)
        Path(cmd[4] if "--outfile" in cmd else cmd[2]).write_bytes(b"gguf")
        return R()

    monkeypatch.setattr(E.subprocess, "run", fake_run)
    E.step(plan, "convert f16 GGUF", lambda: E.convert(plan))
    E.step(plan, "quantize Q4_K_M", lambda: E.quantize(plan))
    assert seen == [E.convert_cmd(plan), E.quantize_cmd(plan)]
    assert [s["ok"] for s in plan.steps] == [True, True]

    R.returncode = 1
    with pytest.raises(E.ExportError):
        E.step(plan, "quantize Q4_K_M", lambda: E.quantize(plan))
    assert plan.steps[-1] == {
        "step": "quantize Q4_K_M",
        "ok": False,
        "sec": 0.0,
        "error": "ExportError",
    }


def test_cleanup_respects_keep_intermediate(tmp_path):
    plan = E.build_plan(args(tmp_path))
    plan.merged_dir.mkdir(parents=True)
    plan.adapter_dir.mkdir(parents=True)
    plan.f16.write_bytes(b"x")
    plan.keep_intermediate = True
    E.cleanup(plan)
    assert plan.f16.exists() and plan.merged_dir.exists()
    plan.keep_intermediate = False
    E.cleanup(plan)
    assert not plan.f16.exists() and not plan.merged_dir.exists() and not plan.adapter_dir.exists()


def test_copy_tokenizer_files(tmp_path):
    base, merged = tmp_path / "b", tmp_path / "m"
    base.mkdir(), merged.mkdir()
    for n in ("tokenizer.json", "chat_template.jinja", "vocab.json"):
        (base / n).write_text(n)
    (merged / "tokenizer.json").write_text("already")
    assert E.copy_tokenizer_files(base, merged) == ["vocab.json", "chat_template.jinja"]
    assert (merged / "tokenizer.json").read_text() == "already"


# --- with real torch/safetensors on tiny tensors (skipped if the train extra is absent) ---

HIDDEN, VOCAB, R = 8, 32, 4


def _tiny_qwen35(tmp_path: Path):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "tie_word_embeddings": True,
                "text_config": {"tie_word_embeddings": True},
            }
        )
    )
    keys = {
        "model.language_model.embed_tokens.weight": (VOCAB, HIDDEN),
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": (3 * HIDDEN, HIDDEN),
        "model.language_model.layers.1.self_attn.q_proj.weight": (HIDDEN, HIDDEN),
        "model.language_model.layers.1.mlp.down_proj.weight": (HIDDEN, HIDDEN),
    }
    torch.manual_seed(0)
    tensors = {k: torch.randn(*s).to(torch.bfloat16) for k, s in keys.items()}
    st.save_file(tensors, str(base / "model-00001-of-00001.safetensors"))
    (base / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {}, "weight_map": dict.fromkeys(keys, "model-00001-of-00001.safetensors")}
        )
    )
    adapter = {}
    pre = "base_model.model.model."
    for mod, out_dim in (
        ("layers.0.linear_attn.in_proj_q", HIDDEN),
        ("layers.0.linear_attn.in_proj_k", HIDDEN),
        ("layers.0.linear_attn.in_proj_v", HIDDEN),
        ("layers.1.self_attn.q_proj", HIDDEN),
        ("layers.1.mlp.down_proj", HIDDEN),
        ("unembed_tokens", VOCAB),
    ):
        adapter[f"{pre}{mod}.lora_A.weight"] = torch.randn(R, HIDDEN)
        adapter[f"{pre}{mod}.lora_B.weight"] = torch.randn(out_dim, R)
    return base, tensors, adapter


def test_inspect_and_untie_on_tiny_model(tmp_path):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    pytest.importorskip("tinker_cookbook.weights._merge_qwen3_5")
    base, tensors, adapter = _tiny_qwen35(tmp_path)
    plan = E.build_plan(args(tmp_path))
    plan.adapter_dir.mkdir(parents=True)
    st.save_file(adapter, str(plan.adapter_dir / "adapter_model.safetensors"))
    (plan.adapter_dir / "adapter_config.json").write_text(json.dumps({"r": R, "lora_alpha": 8}))

    msg = E.inspect_adapter(plan)
    assert "unembed LoRA yes" in msg and "in_proj_q" in msg

    E.strip_unembed(plan)
    kept = st.load_file(str(plan.merge_adapter_dir / "adapter_model.safetensors"))
    assert not any("unembed" in k for k in kept) and len(kept) == len(adapter) - 2

    # stand-in for the merge: the merged dir is a copy of the base
    import shutil

    shutil.copytree(base, plan.merged_dir)
    E.untie_lm_head(plan)
    cfg = json.loads((plan.merged_dir / "config.json").read_text())
    assert (
        cfg["tie_word_embeddings"] is False and cfg["text_config"]["tie_word_embeddings"] is False
    )
    idx = json.loads((plan.merged_dir / "model.safetensors.index.json").read_text())
    assert idx["weight_map"]["lm_head.weight"] == E.LM_HEAD_SHARD
    head = st.load_file(str(plan.merged_dir / E.LM_HEAD_SHARD))["lm_head.weight"]
    embed = tensors["model.language_model.embed_tokens.weight"]
    a = adapter["base_model.model.model.unembed_tokens.lora_A.weight"]
    b = adapter["base_model.model.model.unembed_tokens.lora_B.weight"]
    expected = (embed.float() + (8 / R) * b @ a).to(torch.bfloat16)
    assert head.dtype == torch.bfloat16 and torch.equal(head, expected)
    # input embeddings untouched
    merged_embed = st.load_file(str(plan.merged_dir / "model-00001-of-00001.safetensors"))[
        "model.language_model.embed_tokens.weight"
    ]
    assert torch.equal(merged_embed, embed)
