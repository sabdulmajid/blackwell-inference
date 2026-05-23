#!/usr/bin/env bash
set -euo pipefail

MODEL=""
TP="1"
PORT="8000"
ATTN_BACKEND="triton"
FP8_BACKEND="triton"
LINEAR_ATTN_BACKEND=""
LINEAR_ATTN_DECODE_BACKEND=""
LINEAR_ATTN_PREFILL_BACKEND=""
CONTEXT_LENGTH="8192"
MEM_FRACTION="0.90"
DRY_RUN="0"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: bash repros/sglang_attention_backend_sm120/run_repro.sh --model MODEL [options]

Options:
  --model MODEL               Model ID or local path. Required.
  --tp N                      Tensor parallel size. Default: 1.
  --port PORT                 Server port. Default: 8000.
  --attention-backend NAME    SGLang attention backend. Default: triton.
  --fp8-gemm-backend NAME     SGLang FP8 GEMM backend. Default: triton.
  --linear-attn-backend NAME  Optional SGLang linear attention backend.
  --linear-attn-decode-backend NAME
                              Optional SGLang linear attention decode backend.
  --linear-attn-prefill-backend NAME
                              Optional SGLang linear attention prefill backend.
  --context-length N          Context length. Default: 8192.
  --mem-fraction-static X     Static memory fraction. Default: 0.90.
  --extra ARG                 Extra argument passed to SGLang. Repeat for multiple args.
  --dry-run                   Print planned command and write command metadata only.
  --help                      Show this help.

GPU launches must be wrapped by scripts/run_with_gpu_lock.py.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --tp) TP="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --attention-backend) ATTN_BACKEND="$2"; shift 2 ;;
    --fp8-gemm-backend) FP8_BACKEND="$2"; shift 2 ;;
    --linear-attn-backend) LINEAR_ATTN_BACKEND="$2"; shift 2 ;;
    --linear-attn-decode-backend) LINEAR_ATTN_DECODE_BACKEND="$2"; shift 2 ;;
    --linear-attn-prefill-backend) LINEAR_ATTN_PREFILL_BACKEND="$2"; shift 2 ;;
    --context-length) CONTEXT_LENGTH="$2"; shift 2 ;;
    --mem-fraction-static) MEM_FRACTION="$2"; shift 2 ;;
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

RUN_DIR="${BLACKWELL_INFERENCE_GPU_RUN_DIR:-${SM120_LAB_RUN_DIR:-results/repros/sglang_attention_backend_sm120/$(date -u +%Y%m%dT%H%M%SZ)}}"
mkdir -p "$RUN_DIR"

SERVER_ARGS=(
  python -m sglang.launch_server
  --model-path "$MODEL"
  --host 127.0.0.1
  --port "$PORT"
  --tensor-parallel-size "$TP"
  --context-length "$CONTEXT_LENGTH"
  --mem-fraction-static "$MEM_FRACTION"
  --attention-backend "$ATTN_BACKEND"
  --fp8-gemm-backend "$FP8_BACKEND"
)
if [[ -n "$LINEAR_ATTN_BACKEND" ]]; then
  SERVER_ARGS+=(--linear-attn-backend "$LINEAR_ATTN_BACKEND")
fi
if [[ -n "$LINEAR_ATTN_DECODE_BACKEND" ]]; then
  SERVER_ARGS+=(--linear-attn-decode-backend "$LINEAR_ATTN_DECODE_BACKEND")
fi
if [[ -n "$LINEAR_ATTN_PREFILL_BACKEND" ]]; then
  SERVER_ARGS+=(--linear-attn-prefill-backend "$LINEAR_ATTN_PREFILL_BACKEND")
fi
SERVER_ARGS+=("${EXTRA_ARGS[@]}")

printf '%q ' "${SERVER_ARGS[@]}" > "$RUN_DIR/command.txt"
printf '\n' >> "$RUN_DIR/command.txt"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "dry_run" | tee "$RUN_DIR/status.txt"
  cat "$RUN_DIR/command.txt"
  exit 0
fi

if [[ "${BLACKWELL_INFERENCE_GPU_LOCKED:-${SM120_LAB_GPU_LOCKED:-0}}" != "1" ]]; then
  echo "Refusing to launch SGLang without scripts/run_with_gpu_lock.py." >&2
  echo "Use: python scripts/run_with_gpu_lock.py --gpus 0 --min-free-gb 70 --wait --label sglang_fp8_repro -- bash repros/sglang_attention_backend_sm120/run_repro.sh --model <MODEL>" >&2
  echo "missing_gpu_lock" | tee "$RUN_DIR/status.txt"
  exit 4
fi

python scripts/verify_blackwell.py --probe-cuda --out "$RUN_DIR/verify_blackwell.json" || true

collect_backend_evidence() {
  grep -RniE "OutOfResources|shared memory|Hardware limit|attention|flashinfer|triton|fp8|nan|DeepGemm|backend|gdn|linear|error|exception" "$RUN_DIR" > "$RUN_DIR/backend_grep.txt" || true
  python scripts/extract_sglang_backend_evidence.py \
    --grep-file "$RUN_DIR/backend_grep.txt" \
    --out "$RUN_DIR/backend_summary.json" \
    --status-file "$RUN_DIR/status.txt" \
    --model "$MODEL" \
    --attention-backend "$ATTN_BACKEND" \
    --fp8-gemm-backend "$FP8_BACKEND" \
    --linear-attn-backend "${LINEAR_ATTN_BACKEND:-auto}" \
    --linear-attn-decode-backend "${LINEAR_ATTN_DECODE_BACKEND:-auto}" \
    --linear-attn-prefill-backend "${LINEAR_ATTN_PREFILL_BACKEND:-auto}" \
    --tp "$TP" \
    > "$RUN_DIR/backend_summary_stdout.json" 2> "$RUN_DIR/backend_summary_stderr.log" || true
}

PID=""
cleanup() {
  if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
    kill "$PID" 2>/dev/null || true
    wait "$PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

set +e
"${SERVER_ARGS[@]}" > "$RUN_DIR/server_stdout.log" 2> "$RUN_DIR/server_stderr.log" &
PID=$!
set -e

echo "$PID" > "$RUN_DIR/server.pid"

READY="0"
for i in $(seq 1 160); do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "server exited early" | tee "$RUN_DIR/status.txt"
    wait "$PID" || true
    collect_backend_evidence
    exit 3
  fi
  if grep -qiE "Uvicorn running|Application startup complete|server started|The server is fired up" "$RUN_DIR/server_stdout.log" "$RUN_DIR/server_stderr.log" 2>/dev/null; then
    READY="1"
    break
  fi
  sleep 2
done

if [[ "$READY" != "1" ]]; then
  echo "server startup timeout" | tee "$RUN_DIR/status.txt"
  collect_backend_evidence
  exit 5
fi

set +e
python benchmarks/serve_bench.py \
  --framework sglang \
  --base-url "http://127.0.0.1:${PORT}" \
  --model "$MODEL" \
  --backend "$ATTN_BACKEND/fp8_${FP8_BACKEND}" \
  --dtype "fp8" \
  --quantization "fp8" \
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

if [[ "$BENCH_RC" -eq 0 ]]; then
  echo "completed" | tee "$RUN_DIR/status.txt"
else
  echo "benchmark_failed rc=$BENCH_RC" | tee "$RUN_DIR/status.txt"
fi
collect_backend_evidence
exit "$BENCH_RC"
