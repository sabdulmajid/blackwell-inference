#!/usr/bin/env bash
set -euo pipefail

ITERATIONS=1
PROMPT_FILE="prompts/master_goal_loop.md"
LOG_DIR="results/codex_goal_loop"
SANDBOX="workspace-write"
MODEL_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --prompt) PROMPT_FILE="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --sandbox) SANDBOX="$2"; shift 2 ;;
    --model-arg) MODEL_ARGS+=("$2"); shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$LOG_DIR"

if ! command -v codex >/dev/null 2>&1; then
  echo "codex CLI not found. Install with: npm i -g @openai/codex" >&2
  exit 1
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "prompt file not found: $PROMPT_FILE" >&2
  exit 1
fi

for i in $(seq 1 "$ITERATIONS"); do
  TS=$(date -u +%Y%m%dT%H%M%SZ)
  OUT="$LOG_DIR/${TS}_iteration_${i}.jsonl"
  MSG="$LOG_DIR/${TS}_iteration_${i}_last_message.md"
  META="$LOG_DIR/${TS}_iteration_${i}_meta.txt"
  echo "[goal-loop] iteration $i/$ITERATIONS at $TS"
  {
    echo "iteration=$i"
    echo "start_utc=$TS"
    echo "prompt_file=$PROMPT_FILE"
    echo "sandbox=$SANDBOX"
  } > "$META"

  # Always snapshot GPU state before the agent gets a chance to schedule GPU work.
  python scripts/gpu_guard.py status --out "$LOG_DIR/${TS}_gpu_status_before.json" || true
  git status --short > "$LOG_DIR/${TS}_git_status_before.txt" || true
  git diff --stat > "$LOG_DIR/${TS}_git_diff_stat_before.txt" || true
  cp TASK_BOARD.md "$LOG_DIR/${TS}_TASK_BOARD_before.md"

  codex exec --json --sandbox "$SANDBOX" "$(cat "$PROMPT_FILE")" \
    -o "$MSG" "${MODEL_ARGS[@]}" | tee "$OUT"

  END_TS=$(date -u +%Y%m%dT%H%M%SZ)
  python scripts/gpu_guard.py status --out "$LOG_DIR/${TS}_gpu_status_after.json" || true
  cp TASK_BOARD.md "$LOG_DIR/${TS}_TASK_BOARD_after.md"
  diff -u "$LOG_DIR/${TS}_TASK_BOARD_before.md" "$LOG_DIR/${TS}_TASK_BOARD_after.md" \
    > "$LOG_DIR/${TS}_TASK_BOARD_delta.diff" || true
  git diff --stat > "$LOG_DIR/${TS}_git_diff_stat_after.txt" || true

  git status --short | tee "$LOG_DIR/${TS}_git_status.txt"
  {
    echo "end_utc=$END_TS"
    echo "jsonl=$OUT"
    echo "last_message=$MSG"
    echo "gpu_status_before=$LOG_DIR/${TS}_gpu_status_before.json"
    echo "gpu_status_after=$LOG_DIR/${TS}_gpu_status_after.json"
    echo "task_board_delta=$LOG_DIR/${TS}_TASK_BOARD_delta.diff"
    echo "git_diff_stat_after=$LOG_DIR/${TS}_git_diff_stat_after.txt"
  } >> "$META"
  echo "[goal-loop] last message: $MSG"
  echo "[goal-loop] jsonl: $OUT"
  echo "[goal-loop] meta: $META"
  echo

done
