"""Adapter download, merge, GGUF conversion and Q4_K_M quantization (PRD FR-34, §11.4, NFR-10).

    uv run --extra train python -m training.export --run-id 20261003-0612-full --version 1 --dry-run
    uv run --extra train python -m training.export --run-id 20261003-0612-full --version 1
    uv run --extra train python -m training.export --checkpoint tinker://.../sampler_weights/x --version 1

Steps (each prints PASS/FAIL and stops at the first FAIL):
  0. preflight   tools on PATH, base model present, free disk ≥ --min-free-gb, outputs absent
  1. download    tinker_cookbook.weights.download(tinker_path=...) → RestClient
                 .get_checkpoint_archive_url_from_tinker_path → signed tar → models/merged/v{N}-adapter
  2. inspect     adapter_config (r, lora_alpha) + every LoRA key planned against the base's
                 safetensors index with the cookbook's Qwen3.5 merge profile (split in_proj_q/k/v →
                 fused in_proj_qkv, `model.` → `model.language_model.`); no weights loaded
  3. merge       tinker_cookbook.weights.build_hf_model(merge_strategy="shard"): one base shard at
                 a time, bf16 kept, tokenizer + chat_template.jinja copied → models/merged/v{N}
  4. lm_head     --unembed untied (default): Tinker trains a LoRA on the *unembedding* only, but
                 Qwen3.5-4B ties embeddings. The cookbook merges that delta into embed_tokens, which
                 also shifts the input embeddings. Instead we strip the unembed LoRA before step 3
                 and write lm_head.weight = embed + (alpha/r)·B·A as its own tensor
                 (tie_word_embeddings=false). llama.cpp's qwen35 loader uses output.weight when
                 present. --unembed tied keeps the cookbook behaviour.
  5. convert     ~/src/llama.cpp/convert_hf_to_gguf.py --outtype f16
  6. quantize    llama-quantize … Q4_K_M → models/rakshak-detector-v{N}-q4km.gguf
  7. cleanup     delete the f16 GGUF, merged dir and adapter unless --keep-intermediate

Writes training/runs/{run_id}/export.json (paths, sizes, sha256, durations). Stop Ollama and the
llama-servers first (NFR-10): the shard merge peaks at about one base shard (5.3 GB) plus fp32
LoRA deltas. Never run this while other models are loaded on an 18 GB Mac.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "training" / "runs"
MODELS = ROOT / "models"
LLAMA_CPP = Path(os.environ.get("LLAMA_CPP", Path.home() / "src" / "llama.cpp"))
MIN_FREE_GB = 20.0
UNEMBED_KEY = "unembed_tokens"
TOKENIZER_FILES = (
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "special_tokens_map.json",
    "added_tokens.json",
    "generation_config.json",
)
LM_HEAD_SHARD = "model-lm_head.safetensors"
ROW_CHUNK = 16384  # lm_head rows per chunk (bounded fp32 temporaries)


class ExportError(Exception):
    pass


@dataclass
class Plan:
    run_id: str | None
    checkpoint: str
    version: int
    base: Path
    adapter_dir: Path
    merge_adapter_dir: Path  # adapter actually passed to the merge (unembed stripped if untied)
    merged_dir: Path
    f16: Path
    out: Path
    unembed: str
    keep_intermediate: bool
    llama_cpp: Path
    convert_python: Path
    export_json: Path | None
    min_free_gb: float = MIN_FREE_GB
    steps: list[dict] = field(default_factory=list)


# --- plan (pure) ---


def checkpoint_from_run(run_dir: Path) -> str:
    """Best sampler checkpoint from a train_tinker run dir (§11.3 config.json)."""
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise ExportError(f"{cfg_path} not found")
    cfg = json.loads(cfg_path.read_text("utf-8"))
    best = cfg.get("best") or {}
    if best.get("checkpoint"):
        return best["checkpoint"]
    cps = cfg.get("checkpoints") or []
    if cps and cps[-1].get("sampler_path"):
        return cps[-1]["sampler_path"]
    raise ExportError(f"{cfg_path} has no best/checkpoints sampler path")


def build_plan(args: argparse.Namespace) -> Plan:
    if not args.run_id and not args.checkpoint:
        raise ExportError("give --run-id or --checkpoint")
    run_dir = RUNS / args.run_id if args.run_id else None
    checkpoint = args.checkpoint or checkpoint_from_run(run_dir)
    if not checkpoint.startswith("tinker://"):
        raise ExportError(f"checkpoint must be a tinker:// path, got {checkpoint!r}")
    n = args.version
    merged_root = args.merged_root
    out = args.out or MODELS / f"rakshak-detector-v{n}-q4km.gguf"
    llama_cpp = args.llama_cpp
    adapter_dir = merged_root / f"v{n}-adapter"
    return Plan(
        run_id=args.run_id,
        checkpoint=checkpoint,
        version=n,
        base=args.base,
        adapter_dir=adapter_dir,
        merge_adapter_dir=adapter_dir / "no_unembed" if args.unembed == "untied" else adapter_dir,
        merged_dir=merged_root / f"v{n}",
        f16=merged_root / f"rakshak-detector-v{n}-f16.gguf",
        out=out,
        unembed=args.unembed,
        keep_intermediate=args.keep_intermediate,
        llama_cpp=llama_cpp,
        convert_python=llama_cpp / ".venv" / "bin" / "python",
        export_json=(run_dir / "export.json") if run_dir else None,
        min_free_gb=args.min_free_gb,
    )


def convert_cmd(plan: Plan) -> list[str]:
    return [
        str(plan.convert_python),
        str(plan.llama_cpp / "convert_hf_to_gguf.py"),
        str(plan.merged_dir),
        "--outfile",
        str(plan.f16),
        "--outtype",
        "f16",
    ]


def quantize_cmd(plan: Plan) -> list[str]:
    return ["llama-quantize", str(plan.f16), str(plan.out), "Q4_K_M"]


def describe(plan: Plan) -> list[str]:
    """Human-readable plan (dry run)."""
    return [
        f"checkpoint  {plan.checkpoint}",
        f"base        {plan.base}",
        f"download    → {plan.adapter_dir}",
        f"merge       build_hf_model(base={plan.base}, adapter={plan.merge_adapter_dir}, "
        f"out={plan.merged_dir}, merge_strategy='shard')",
        f"unembed     {plan.unembed}"
        + (
            " (lm_head.weight written separately, tie_word_embeddings=false)"
            if plan.unembed == "untied"
            else ""
        ),
        "convert     " + " ".join(convert_cmd(plan)),
        "quantize    " + " ".join(quantize_cmd(plan)),
        "cleanup     "
        + (
            "keep intermediates"
            if plan.keep_intermediate
            else f"rm {plan.f16}, {plan.merged_dir}, {plan.adapter_dir}"
        ),
        f"export.json {plan.export_json or '(no --run-id: not written)'}",
    ]


# --- small helpers ---


def free_gb(path: Path) -> float:
    p = path
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return 0


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def run_cmd(cmd: list[str]) -> None:
    """Run a tool; output streams to the terminal (no message text is involved)."""
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        raise ExportError(f"{Path(cmd[0]).name} exited with {r.returncode}")


def step(plan: Plan, name: str, fn) -> object:
    t0 = time.perf_counter()
    try:
        result = fn()
    except Exception as e:
        sec = time.perf_counter() - t0
        plan.steps.append(
            {"step": name, "ok": False, "sec": round(sec, 1), "error": type(e).__name__}
        )
        print(f"FAIL  {name} ({sec:.0f}s): {type(e).__name__}: {e}", flush=True)
        raise ExportError(f"{name} failed") from e
    sec = time.perf_counter() - t0
    detail = f" — {result}" if isinstance(result, str) and result else ""
    plan.steps.append({"step": name, "ok": True, "sec": round(sec, 1)})
    print(f"PASS  {name} ({sec:.0f}s){detail}", flush=True)
    return result


# --- steps ---


def preflight(plan: Plan, *, dry_run: bool = False) -> str:
    problems = []
    if shutil.which("llama-quantize") is None:
        problems.append("llama-quantize not on PATH (brew install llama.cpp)")
    if not (plan.llama_cpp / "convert_hf_to_gguf.py").exists():
        problems.append(f"no {plan.llama_cpp}/convert_hf_to_gguf.py")
    if not plan.convert_python.exists():
        problems.append(f"no {plan.convert_python} (llama.cpp venv)")
    if not (plan.base / "config.json").exists() or not list(plan.base.glob("*.safetensors")):
        problems.append(f"base model missing in {plan.base} (bash scripts/smoke/qwen_gguf.sh)")
    for p in (plan.merged_dir, plan.f16, plan.out):
        if p.exists():
            problems.append(f"{p} already exists; remove it or bump --version")
    free = free_gb(plan.merged_dir.parent)
    if free < plan.min_free_gb:
        problems.append(f"only {free:.1f} GB free, need ≥ {plan.min_free_gb:.0f} GB")
    if not dry_run and not _tinker_key():
        problems.append("TINKER_API_KEY not set in env or .env")
    if problems:
        raise ExportError("; ".join(problems))
    return f"{free:.0f} GB free"


def _tinker_key() -> str | None:
    if os.environ.get("TINKER_API_KEY"):
        return os.environ["TINKER_API_KEY"]
    from hub.settings import get_settings

    k = get_settings().tinker_api_key
    return k.get_secret_value() if k else None


def download_adapter(plan: Plan) -> str:
    from tinker_cookbook import weights

    key = _tinker_key()
    if key:  # cookbook's download() builds ServiceClient() from the environment
        os.environ["TINKER_API_KEY"] = key
    if (plan.adapter_dir / "adapter_model.safetensors").exists():
        return f"reusing {plan.adapter_dir}"
    weights.download(tinker_path=plan.checkpoint, output_dir=str(plan.adapter_dir))
    for f in ("adapter_model.safetensors", "adapter_config.json"):
        if not (plan.adapter_dir / f).exists():
            raise ExportError(f"downloaded archive has no {f}")
    return gb(size_bytes(plan.adapter_dir))


def base_state_keys(base: Path) -> set[str]:
    idx = base / "model.safetensors.index.json"
    if idx.exists():
        return set(json.loads(idx.read_text("utf-8"))["weight_map"])
    from safetensors import safe_open

    keys: set[str] = set()
    for f in base.glob("*.safetensors"):
        with safe_open(str(f), "pt") as st:
            keys |= set(st.keys())
    return keys


def lora_modules(adapter_keys: list[str]) -> list[str]:
    """Distinct module names (layer index and lora_A/B stripped) for the summary."""
    out = set()
    for k in adapter_keys:
        if ".lora_A" not in k:
            continue
        parts = k.replace(".lora_A.weight", "").split(".")
        out.add(parts[-1] if parts[-1] != "weight" else parts[-2])
    return sorted(out)


def inspect_adapter(plan: Plan) -> str:
    """Plan every LoRA op against the base keys with the cookbook's Qwen3.5 profile; raises if
    any adapter module has no target (catches key-name / architecture mismatches early)."""
    from safetensors.torch import load_file
    from tinker_cookbook.weights import _merge_qwen3_5 as q35

    cfg = json.loads((plan.adapter_dir / "adapter_config.json").read_text("utf-8"))
    base_cfg = json.loads((plan.base / "config.json").read_text("utf-8"))
    keys = base_state_keys(plan.base)
    profile = q35.detect_profile(base_cfg, keys)
    if profile is None:
        raise ExportError(f"base model_type {base_cfg.get('model_type')!r} is not qwen3_5")
    w = load_file(str(plan.adapter_dir / "adapter_model.safetensors"))
    ops = q35.plan_merge_ops(w, cfg, keys, profile)
    missing = [k for k in ops if k not in keys]
    if missing:
        raise ExportError(f"{len(missing)} merge targets absent from base, e.g. {missing[:3]}")
    mods = lora_modules(list(w))
    has_unembed = any(UNEMBED_KEY in k for k in w)
    return (
        f"r={cfg.get('r')} alpha={cfg.get('lora_alpha')} {len(ops)} targets; modules "
        f"{','.join(mods)}; unembed LoRA {'yes' if has_unembed else 'no'}"
    )


def strip_unembed(plan: Plan) -> str:
    """Copy of the adapter without the unembed LoRA (merged separately as lm_head)."""
    from safetensors.torch import load_file, save_file

    w = load_file(str(plan.adapter_dir / "adapter_model.safetensors"))
    kept = {k: v for k, v in w.items() if UNEMBED_KEY not in k}
    plan.merge_adapter_dir.mkdir(parents=True, exist_ok=True)
    save_file(kept, str(plan.merge_adapter_dir / "adapter_model.safetensors"))
    shutil.copy2(plan.adapter_dir / "adapter_config.json", plan.merge_adapter_dir)
    return f"{len(w) - len(kept)} unembed tensors set aside"


def merge(plan: Plan) -> str:
    from tinker_cookbook import weights

    plan.merged_dir.parent.mkdir(parents=True, exist_ok=True)
    weights.build_hf_model(
        base_model=str(plan.base),
        adapter_path=str(plan.merge_adapter_dir),
        output_path=str(plan.merged_dir),
        merge_strategy="shard",
    )
    copied = copy_tokenizer_files(plan.base, plan.merged_dir)
    return f"{gb(size_bytes(plan.merged_dir))}" + (f"; copied {copied}" if copied else "")


def copy_tokenizer_files(base: Path, merged: Path) -> list[str]:
    """Tokenizer + chat template from the base when the merge didn't already write them."""
    copied = []
    for name in TOKENIZER_FILES:
        if (base / name).exists() and not (merged / name).exists():
            shutil.copy2(base / name, merged / name)
            copied.append(name)
    if not (merged / "chat_template.jinja").exists() and (base / "chat_template.jinja").exists():
        raise ExportError("chat_template.jinja missing from merged dir")
    return copied


def find_key_file(model_dir: Path, key: str) -> Path:
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        return model_dir / json.loads(idx.read_text("utf-8"))["weight_map"][key]
    files = list(model_dir.glob("*.safetensors"))
    if len(files) != 1:
        raise ExportError(f"cannot locate {key} in {model_dir}")
    return files[0]


def untie_lm_head(plan: Plan) -> str:
    """lm_head.weight = embed_tokens (base) + (alpha/r)·B·A in fp32, row-chunked, saved bf16."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    cfg = json.loads((plan.adapter_dir / "adapter_config.json").read_text("utf-8"))
    scaling = cfg["lora_alpha"] / cfg["r"]
    with safe_open(str(plan.adapter_dir / "adapter_model.safetensors"), "pt") as st:
        names = [k for k in st.keys() if UNEMBED_KEY in k]  # noqa: SIM118 (safe_open)
        a_key = next((k for k in names if ".lora_A" in k), None)
        b_key = next((k for k in names if ".lora_B" in k), None)
        if not a_key or not b_key:
            return "no unembed LoRA in adapter; nothing to do"
        lora_a = st.get_tensor(a_key).float()  # (r, hidden)
        lora_b = st.get_tensor(b_key).float() * scaling  # (vocab, r)

    keys = base_state_keys(plan.merged_dir)
    if "lm_head.weight" in keys:
        raise ExportError("merged model already has lm_head.weight (untied base?)")
    embed_key = next(k for k in keys if k.endswith("embed_tokens.weight") and "mtp" not in k)
    with safe_open(str(find_key_file(plan.merged_dir, embed_key)), "pt") as st:
        embed = st.get_tensor(embed_key)
    if embed.shape != (lora_b.shape[0], lora_a.shape[1]):
        raise ExportError(
            f"shape mismatch embed {tuple(embed.shape)} vs B·A ({lora_b.shape[0]}, {lora_a.shape[1]})"
        )
    head = torch.empty_like(embed)
    for i in range(0, embed.shape[0], ROW_CHUNK):
        j = i + ROW_CHUNK
        head[i:j] = (embed[i:j].float() + lora_b[i:j] @ lora_a).to(embed.dtype)
    save_file({"lm_head.weight": head.contiguous()}, str(plan.merged_dir / LM_HEAD_SHARD))
    del embed, head

    idx_path = plan.merged_dir / "model.safetensors.index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text("utf-8"))
    else:  # single-shard merge: build an index so both files load
        only = next(f for f in plan.merged_dir.glob("*.safetensors") if f.name != LM_HEAD_SHARD)
        with safe_open(str(only), "pt") as st:
            idx = {"metadata": {}, "weight_map": dict.fromkeys(st.keys(), only.name)}
    idx["weight_map"]["lm_head.weight"] = LM_HEAD_SHARD
    idx.setdefault("metadata", {})["total_size"] = sum(
        f.stat().st_size for f in plan.merged_dir.glob("*.safetensors")
    )
    idx_path.write_text(json.dumps(idx, indent=2) + "\n", "utf-8")
    set_untied(plan.merged_dir / "config.json")
    return f"lm_head {tuple(lora_b.shape[:1]) + tuple(lora_a.shape[1:])} scaling {scaling:g}"


def set_untied(config_path: Path) -> None:
    cfg = json.loads(config_path.read_text("utf-8"))
    cfg["tie_word_embeddings"] = False
    if isinstance(cfg.get("text_config"), dict):
        cfg["text_config"]["tie_word_embeddings"] = False
    config_path.write_text(json.dumps(cfg, indent=2) + "\n", "utf-8")


def convert(plan: Plan) -> str:
    run_cmd(convert_cmd(plan))
    if not plan.f16.exists():
        raise ExportError(f"{plan.f16} not written")
    return gb(size_bytes(plan.f16))


def quantize(plan: Plan) -> str:
    plan.out.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(quantize_cmd(plan))
    if not plan.out.exists():
        raise ExportError(f"{plan.out} not written")
    return gb(size_bytes(plan.out))


def cleanup(plan: Plan) -> str:
    if plan.keep_intermediate:
        return "kept intermediates"
    removed = []
    for p in (plan.f16, plan.merged_dir, plan.adapter_dir):
        if p.is_dir():
            shutil.rmtree(p)
            removed.append(p.name)
        elif p.exists():
            p.unlink()
            removed.append(p.name)
    return "removed " + ", ".join(removed)


# --- main ---


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def export(plan: Plan) -> dict:
    started = datetime.now(UTC)
    sizes: dict[str, int] = {}
    step(plan, "preflight", lambda: preflight(plan))
    step(plan, "download adapter", lambda: download_adapter(plan))
    sizes["adapter_bytes"] = size_bytes(plan.adapter_dir)
    step(plan, "inspect adapter vs base", lambda: inspect_adapter(plan))
    if plan.unembed == "untied":
        step(plan, "strip unembed LoRA", lambda: strip_unembed(plan))
    step(plan, "merge (bf16, shard)", lambda: merge(plan))
    if plan.unembed == "untied":
        step(plan, "write untied lm_head", lambda: untie_lm_head(plan))
    sizes["merged_bytes"] = size_bytes(plan.merged_dir)
    step(plan, "convert f16 GGUF", lambda: convert(plan))
    sizes["f16_bytes"] = size_bytes(plan.f16)
    step(plan, "quantize Q4_K_M", lambda: quantize(plan))
    sizes["gguf_bytes"] = size_bytes(plan.out)
    digest = step(plan, "sha256", lambda: sha256_file(plan.out))
    step(plan, "cleanup", lambda: cleanup(plan))
    record = {
        "run_id": plan.run_id,
        "checkpoint": plan.checkpoint,
        "version": plan.version,
        "detector_version": f"rakshak-detector-v{plan.version}-q4km",
        "base": rel(plan.base),
        "unembed": plan.unembed,
        "gguf": rel(plan.out),
        "gguf_sha256": digest,
        "sizes": sizes,
        "kept_intermediate": plan.keep_intermediate,
        "steps": plan.steps,
        "total_sec": round(sum(s["sec"] for s in plan.steps), 1),
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now(UTC).isoformat(timespec="seconds"),
        "tools": {"llama_cpp": str(plan.llama_cpp), "quantize": shutil.which("llama-quantize")},
    }
    if plan.export_json:
        plan.export_json.write_text(json.dumps(record, indent=2) + "\n", "utf-8")
    return record


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", help="training/runs/{run_id}; uses config.json best checkpoint")
    ap.add_argument("--checkpoint", help="tinker://... sampler checkpoint (overrides --run-id's)")
    ap.add_argument("--version", type=int, required=True, help="N in rakshak-detector-vN")
    ap.add_argument("--base", type=Path, default=MODELS / "base" / "qwen3.5-4b")
    ap.add_argument(
        "--out", type=Path, default=None, help="default models/rakshak-detector-v{N}-q4km.gguf"
    )
    ap.add_argument("--merged-root", type=Path, default=MODELS / "merged")
    ap.add_argument("--llama-cpp", type=Path, default=LLAMA_CPP)
    ap.add_argument("--unembed", choices=("untied", "tied"), default="untied")
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    ap.add_argument("--keep-intermediate", action="store_true")
    ap.add_argument(
        "--dry-run", action="store_true", help="print the plan and checks; write nothing"
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = build_plan(args)
        if args.dry_run:
            print("\n".join(describe(plan)))
            try:
                print(f"PASS  preflight — {preflight(plan, dry_run=True)}")
            except ExportError as e:
                print(f"FAIL  preflight: {e}")
                return 1
            print("dry run: nothing downloaded or written")
            return 0
        rec = export(plan)
    except ExportError as e:
        print(f"FAIL  {e}", file=sys.stderr)
        return 1
    s = rec["sizes"]
    print(
        f"\nRESULT: PASS  {rec['gguf']} {gb(s['gguf_bytes'])} sha256 {rec['gguf_sha256'][:12]}… "
        f"in {rec['total_sec']:.0f}s (merged {gb(s['merged_bytes'])}, f16 {gb(s['f16_bytes'])})"
    )
    for st in rec["steps"]:
        print(f"  {st['step']:<26} {st['sec']:>7.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
