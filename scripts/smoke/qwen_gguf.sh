#!/usr/bin/env bash
# Phase 1: can llama.cpp convert, quantize and serve base Qwen3.5-4B? (PRD §16 Q5)
#
#   bash scripts/smoke/qwen_gguf.sh            # leaves llama-server running on :8081 for latency.py
#   STOP=1 bash scripts/smoke/qwen_gguf.sh     # stops the server at the end
#
# Steps are skipped when their output already exists. Needs: ~/src/llama.cpp with its .venv
# (IMPLEMENTATION.md §0.2), brew llama.cpp (llama-quantize, llama-server), ~10 GB free disk.
# Override paths with LLAMA_CPP=..., BASE_DIR=..., PORT=...
set -uo pipefail

cd "$(dirname "$0")/../.."
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-4B}"
BASE_DIR="${BASE_DIR:-models/base/qwen3.5-4b}"
F16="${BASE_DIR}-f16.gguf"
Q4="${BASE_DIR}-q4km.gguf"
LLAMA_CPP="${LLAMA_CPP:-$HOME/src/llama.cpp}"
CONVERT_PY="${LLAMA_CPP}/.venv/bin/python"
PORT="${PORT:-8081}"
LOG=var/smoke/llama-server.log

fail() { echo "FAIL  $1"; echo; echo "RESULT: FAIL"; exit 1; }
pass() { echo "PASS  $1"; }
secs() { echo $(( $(date +%s) - $1 ))s; }

for bin in llama-quantize llama-server; do
  command -v "$bin" >/dev/null || fail "$bin not on PATH (brew install llama.cpp)"
done
[ -f "${LLAMA_CPP}/convert_hf_to_gguf.py" ] || fail "no ${LLAMA_CPP}/convert_hf_to_gguf.py (git clone llama.cpp, see IMPLEMENTATION.md §0.2)"
[ -x "$CONVERT_PY" ] || fail "no ${CONVERT_PY} (create the llama.cpp venv, see IMPLEMENTATION.md §0.2)"

# HF_TOKEN from .env if not already exported (Qwen3.5-4B is public; the token just avoids rate limits)
if [ -z "${HF_TOKEN:-}" ] && [ -f .env ]; then
  HF_TOKEN="$(grep -E '^HF_TOKEN=' .env | cut -d= -f2- | awk '{print $1}')"
  [ -n "$HF_TOKEN" ] && export HF_TOKEN
fi

echo "=== 1. download ${MODEL_ID} → ${BASE_DIR} ==="
t=$(date +%s)
if [ -f "${BASE_DIR}/config.json" ] && ls "${BASE_DIR}"/*.safetensors >/dev/null 2>&1; then
  pass "download (already present)"
else
  uv run --extra train hf download "$MODEL_ID" --local-dir "$BASE_DIR" || fail "download"
  pass "download ($(secs $t))"
fi

echo "=== 2. convert_hf_to_gguf.py → ${F16} ==="
t=$(date +%s)
if [ -f "$F16" ]; then
  pass "convert (already present)"
else
  "$CONVERT_PY" "${LLAMA_CPP}/convert_hf_to_gguf.py" "$BASE_DIR" --outfile "$F16" --outtype f16 \
    || fail "convert (Plan B: MLX-LM fine-tune, IMPLEMENTATION.md Phase 4B)"
  pass "convert ($(secs $t), $(du -h "$F16" | cut -f1))"
fi

echo "=== 3. llama-quantize Q4_K_M → ${Q4} ==="
t=$(date +%s)
if [ -f "$Q4" ]; then
  pass "quantize (already present)"
else
  llama-quantize "$F16" "$Q4" Q4_K_M || fail "quantize"
  pass "quantize ($(secs $t), $(du -h "$Q4" | cut -f1))"
fi

echo "=== 4. llama-server :${PORT} (-ngl 99) ==="
if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
  fail "something already listens on :${PORT}; stop it first (pkill -f 'llama-server.*--port ${PORT}')"
fi
mkdir -p var/smoke
llama-server -m "$Q4" --host 127.0.0.1 --port "$PORT" -ngl 99 --jinja -c 4096 >"$LOG" 2>&1 &
SERVER_PID=$!
t=$(date +%s)
until curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; do
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -20 "$LOG"; fail "llama-server exited (log: $LOG)"; }
  [ $(( $(date +%s) - t )) -gt 180 ] && { kill "$SERVER_PID"; fail "llama-server not healthy after 180s (log: $LOG)"; }
  sleep 1
done
pass "llama-server healthy in $(secs $t) (pid $SERVER_PID, log $LOG)"

echo "=== 5. A.1 prompt (synthetic message) ==="
DETECTOR_URL="http://127.0.0.1:${PORT}/v1" uv run python - <<'EOF'
import sys
from scripts.smoke._common import check_detector_json, detector_chat, verdict_line

text, ms = detector_chat()
print(text)
print(f"\nlatency {ms:.0f} ms")
ok, detail = check_detector_json(text)
sys.exit(0 if verdict_line(ok, "server answers A.1 with JSON", detail) else 1)
EOF
RC=$?

if [ "${STOP:-0}" = "1" ]; then
  kill "$SERVER_PID" && echo "stopped llama-server"
else
  echo "llama-server left running for latency.py; stop with: kill $SERVER_PID"
fi
echo
[ $RC -eq 0 ] && echo "RESULT: PASS" || echo "RESULT: FAIL"
exit $RC
