"""End-to-end and per-node latency through the running hub API (PRD §12.1, NFR-2).

    uv run python -m eval.latency [--n 50] [--base http://127.0.0.1:8000] [--parent mom]

!!! WARNING: every request is a real check. Each SCAM verdict makes the hub publish an ntfy
!!! alert to NTFY_TOPIC (FR-17), and every check is stored in the hub's DB. Only run this
!!! against a hub started with NTFY_TOPIC unset or NTFY_SERVER pointed at an unreachable port
!!! (and preferably a throwaway DB_PATH). This script cannot switch alerts off for you.

Sends 3 warm-up checks (the last dev items, not measured), then the first N dev items one at a
time to POST {base}/api/check as `parent_id` (channel, sender and text recovered from the
item's §8.2 user message by `eval.systems`). Records client wall time per request and the
hub's own `timings_ms` per node from each Verdict. Writes `eval/results/latency.json` with
p50/p95 per node and end-to-end, the detector version from /health and the hardware string.
Prints numbers only, never message text.
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import httpx

from eval.metrics import percentile
from eval.run_eval import git_sha, is_provisional, sha256_file
from eval.systems import Item, load_items

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "data" / "splits" / "dev.jsonl"
OUT = ROOT / "eval" / "results" / "latency.json"
WARMUP = 3
TIMEOUT_S = 120.0


def _run(cmd: list[str]) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (r.stdout.strip() or None) if r.returncode == 0 else None


def hardware() -> str:
    """e.g. "MacBook Pro M3 Pro 18 GB" (macOS sysctl / system_profiler), else platform info."""
    chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    mem = _run(["sysctl", "-n", "hw.memsize"])
    if not chip:
        return f"{platform.system()} {platform.machine()}".strip()
    name = None
    sp = _run(["system_profiler", "SPHardwareDataType", "-json"])
    if sp:
        try:
            name = json.loads(sp)["SPHardwareDataType"][0].get("machine_name")
        except (ValueError, KeyError, IndexError):
            name = None
    parts = [name, chip.removeprefix("Apple ").strip()]
    if mem and mem.isdigit():
        parts.append(f"{round(int(mem) / 2**30)} GB")
    return " ".join(p for p in parts if p)


def payload(item: Item, parent_id: str) -> dict:
    body = {"parent_id": parent_id, "text": item.text, "channel": item.channel}
    if item.sender:
        body["sender"] = item.sender
    return body


def check(client: httpx.Client, base: str, item: Item, parent_id: str) -> dict:
    """One timed check. Returns {wall_ms, timings_ms, verdict} or {wall_ms, error}."""
    t0 = time.perf_counter()
    try:
        r = client.post(f"{base}/api/check", json=payload(item, parent_id))
        r.raise_for_status()
        v = r.json()
    except (httpx.HTTPError, ValueError) as e:
        return {"wall_ms": (time.perf_counter() - t0) * 1000, "error": type(e).__name__}
    return {
        "wall_ms": (time.perf_counter() - t0) * 1000,
        "timings_ms": v.get("timings_ms") or {},
        "verdict": v.get("verdict"),
    }


def health(client: httpx.Client, base: str) -> dict:
    try:
        r = client.get(f"{base}/health")
        r.raise_for_status()
        return r.json()
    except (httpx.HTTPError, ValueError):
        return {}


def _pcts(values: list[float]) -> dict:
    return {
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "max_ms": max(values) if values else None,
        "n": len(values),
    }


def summarize(records: list[dict]) -> dict:
    ok = [r for r in records if "error" not in r]
    nodes: dict[str, list[float]] = defaultdict(list)
    by_verdict: dict[str, list[float]] = defaultdict(list)
    for r in ok:
        for node, ms in r["timings_ms"].items():
            if isinstance(ms, int | float):
                nodes[node].append(float(ms))
        by_verdict[str(r["verdict"])].append(r["wall_ms"])
    return {
        "end_to_end_wall": _pcts([r["wall_ms"] for r in ok]),
        "nodes": {k: _pcts(v) for k, v in sorted(nodes.items())},
        "end_to_end_by_verdict": {k: _pcts(v) for k, v in sorted(by_verdict.items())},
        "errors": dict(sorted(Counter(r["error"] for r in records if "error" in r).items())),
    }


def run(
    items: list[Item],
    n: int,
    base: str,
    parent_id: str,
    client: httpx.Client,
    warmup: int = WARMUP,
) -> tuple[list[dict], dict]:
    """Warm up on the last `warmup` items, then measure the first `n`, sequentially."""
    measured = items[:n]
    ids = {it.id for it in measured}
    warm = [it for it in items[-warmup:] if it.id not in ids] if warmup else []
    if len(warm) < warmup:  # tiny split: reuse measured items for warm-up
        warm = items[:warmup]
    for it in warm:
        check(client, base, it, parent_id)
    records = []
    for i, it in enumerate(measured, 1):
        rec = check(client, base, it, parent_id)
        records.append(rec)
        status = rec.get("error") or rec.get("verdict")
        print(f"{i}/{len(measured)} {rec['wall_ms']:.0f} ms {status}", flush=True)
    return records, health(client, base)


def build_result(
    records: list[dict], hl: dict, items: list[Item], dev: Path, base: str, warmup: int
) -> dict:
    provisional = is_provisional(items)
    return {
        "provisional": provisional,
        "split": "dev_synthetic" if provisional else "dev",
        "n": len(records),
        "warmup": warmup,
        "base_url": base,
        "hardware": hardware(),
        "detector_version": (hl.get("model_versions") or {}).get("detector"),
        "gemma_model": (hl.get("model_versions") or {}).get("gemma"),
        "hub_status": hl.get("status"),
        **summarize(records),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "data": {"path": str(dev), "sha256": sha256_file(dev)},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--parent", default="mom")
    ap.add_argument("--dev", type=Path, default=DEV)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    if args.dev.name == "test.jsonl":
        print("refused: latency runs on dev only", file=sys.stderr)
        return 2
    base = args.base.rstrip("/")
    items = load_items(args.dev)
    with httpx.Client(timeout=TIMEOUT_S) as client:
        records, hl = run(items, args.n, base, args.parent, client, args.warmup)
    result = build_result(records, hl, items[: args.n], args.dev, base, args.warmup)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    e2e = result["end_to_end_wall"]
    tag = "PROVISIONAL (synthetic dev) " if result["provisional"] else ""
    print(f"\n{tag}latency n={result['n']} on {result['hardware']}")
    print(f"end-to-end p50 {e2e['p50_ms']} / p95 {e2e['p95_ms']} ms; errors {result['errors']}")
    for node, s in result["nodes"].items():
        print(f"  {node:<14} p50 {s['p50_ms']} / p95 {s['p95_ms']} ms")
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
