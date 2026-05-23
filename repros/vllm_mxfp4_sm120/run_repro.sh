#!/usr/bin/env bash
set -euo pipefail

MODEL=""
TP="1"
PORT="8000"
DTYPE="auto"
QUANTIZATION=""
DRY_RUN="0"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: bash repros/vllm_mxfp4_sm120/run_repro.sh --model MODEL [options]

Options:
  --model MODEL       Model ID or local path. Required.
  --tp N             Tensor parallel size. Default: 1.
  --port PORT        Server port. Default: 8000.
  --dtype DTYPE      vLLM dtype argument. Default: auto.
  --quantization Q   Optional vLLM quantization argument.
  --extra ARG        Extra argument passed to vLLM. Repeat for multiple args.
  --dry-run          Print planned command and write command metadata only.
  --help             Show this help.

GPU launches must be wrapped by scripts/run_with_gpu_lock.py.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --tp) TP="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --quantization) QUANTIZATION="$2"; shift 2 ;;
    --extra) EXTRA_ARGS+=("$2"); shift 2 ;;
    --dry-run) DRY_RUN="1"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "--model is required" >&2
  exit 2
fi

RUN_DIR="${BLACKWELL_INFERENCE_GPU_RUN_DIR:-${SM120_LAB_RUN_DIR:-results/repros/vllm_mxfp4_sm120/$(date -u +%Y%m%dT%H%M%SZ)}}"
mkdir -p "$RUN_DIR"

SERVER_CMD=(
  python -m vllm.entrypoints.openai.api_server
  --model "$MODEL"
  --tensor-parallel-size "$TP"
  --dtype "$DTYPE"
  --port "$PORT"
)
if [[ -n "$QUANTIZATION" ]]; then
  SERVER_CMD+=(--quantization "$QUANTIZATION")
fi
SERVER_CMD+=("${EXTRA_ARGS[@]}")

printf '%q ' "${SERVER_CMD[@]}" > "$RUN_DIR/command.txt"
printf '\n' >> "$RUN_DIR/command.txt"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "dry_run" | tee "$RUN_DIR/status.txt"
  cat "$RUN_DIR/command.txt"
  exit 0
fi

if [[ "${BLACKWELL_INFERENCE_GPU_LOCKED:-${SM120_LAB_GPU_LOCKED:-0}}" != "1" ]]; then
  echo "Refusing to launch vLLM without scripts/run_with_gpu_lock.py." >&2
  echo "Use: python scripts/run_with_gpu_lock.py --gpus 0 --min-free-gb 70 --wait --label vllm_mxfp4_repro -- bash repros/vllm_mxfp4_sm120/run_repro.sh --model <MODEL>" >&2
  echo "missing_gpu_lock" | tee "$RUN_DIR/status.txt"
  exit 4
fi

python scripts/verify_blackwell.py --out "$RUN_DIR/verify_blackwell.json" || true

PID=""
cleanup() {
  if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
    kill "$PID" 2>/dev/null || true
    wait "$PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# This script launches the server for a short smoke window. For full benchmarks,
# run the server manually under run_with_gpu_lock.py and then run benchmarks/serve_bench.py.
set +e
"${SERVER_CMD[@]}" > "$RUN_DIR/server_stdout.log" 2> "$RUN_DIR/server_stderr.log" &
PID=$!
set -e

echo "$PID" > "$RUN_DIR/server.pid"

# Wait for startup or failure.
READY="0"
for i in $(seq 1 120); do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "server exited early" | tee "$RUN_DIR/status.txt"
    wait "$PID" || true
    grep -RniE "mxfp4|nvfp4|fp4|marlin|flashinfer|cutlass|backend|quant|error|exception" "$RUN_DIR" > "$RUN_DIR/backend_grep.txt" || true
    python scripts/extract_vllm_backend_evidence.py \
      --grep-file "$RUN_DIR/backend_grep.txt" \
      --out "$RUN_DIR/backend_summary.json" \
      --status-file "$RUN_DIR/status.txt" \
      --model "$MODEL" \
      --dtype "$DTYPE" \
      --quantization "${QUANTIZATION:-unknown}" \
      --tp "$TP" || true
    exit 3
  fi
  if grep -qiE "Uvicorn running|Application startup complete|Started server" "$RUN_DIR/server_stdout.log" "$RUN_DIR/server_stderr.log" 2>/dev/null; then
    READY="1"
    break
  fi
  sleep 2
done

if [[ "$READY" != "1" ]]; then
  echo "server startup timeout" | tee "$RUN_DIR/status.txt"
  grep -RniE "mxfp4|nvfp4|fp4|marlin|flashinfer|cutlass|backend|quant|error|exception" "$RUN_DIR" > "$RUN_DIR/backend_grep.txt" || true
  python scripts/extract_vllm_backend_evidence.py \
    --grep-file "$RUN_DIR/backend_grep.txt" \
    --out "$RUN_DIR/backend_summary.json" \
    --status-file "$RUN_DIR/status.txt" \
    --model "$MODEL" \
    --dtype "$DTYPE" \
    --quantization "${QUANTIZATION:-unknown}" \
    --tp "$TP" || true
  exit 5
fi

set +e
python benchmarks/serve_bench.py \
  --framework vllm \
  --base-url "http://127.0.0.1:${PORT}" \
  --model "$MODEL" \
  --backend "unknown_collect_from_logs" \
  --dtype "$DTYPE" \
  --quantization "${QUANTIZATION:-unknown}" \
  --tp "$TP" \
  --input-words 128 \
  --max-tokens 64 \
  --warmup 2 \
  --requests 5 \
  --metadata "$RUN_DIR/verify_blackwell.json" \
  --out "$RUN_DIR/bench_raw.jsonl" \
  > "$RUN_DIR/bench_summary_stdout.json" 2> "$RUN_DIR/bench_stderr.log"
BENCH_RC=$?
set -e

# Capture likely backend-selection lines.
grep -RniE "mxfp4|nvfp4|fp4|marlin|flashinfer|cutlass|backend|quant|error|exception" "$RUN_DIR" > "$RUN_DIR/backend_grep.txt" || true
if [[ "$BENCH_RC" -eq 0 ]]; then
  echo "completed" | tee "$RUN_DIR/status.txt"
else
  echo "benchmark_failed rc=$BENCH_RC" | tee "$RUN_DIR/status.txt"
fi
python scripts/extract_vllm_backend_evidence.py \
  --grep-file "$RUN_DIR/backend_grep.txt" \
  --out "$RUN_DIR/backend_summary.json" \
  --status-file "$RUN_DIR/status.txt" \
  --model "$MODEL" \
  --dtype "$DTYPE" \
  --quantization "${QUANTIZATION:-unknown}" \
  --tp "$TP" || true

exit "$BENCH_RC"
