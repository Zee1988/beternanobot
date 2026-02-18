#!/usr/bin/env bash
# poll-task.sh — Check task status and output progress
# Usage: poll-task.sh <task_id> [--tail N]
set -euo pipefail

DATA_DIR="$HOME/.claude/data/claude-code-results"

TASK_ID="${1:-}"
TAIL_LINES="${3:-20}"

if [[ -z "$TASK_ID" ]]; then
  # List recent tasks if no task_id given
  echo "Recent tasks:"
  for dir in $(ls -td "$DATA_DIR"/*/ 2>/dev/null | head -5); do
    if [[ -f "$dir/task-meta.json" ]]; then
      python3 -c "
import json
d = json.load(open('$dir/task-meta.json'))
print(f\"  {d.get('status','?'):10s} {d.get('name','') or d.get('task_id','')}  ({d.get('task_id','')})\")
" 2>/dev/null || true
    fi
  done
  exit 0
fi

TASK_DIR="$DATA_DIR/$TASK_ID"

if [[ ! -d "$TASK_DIR" ]]; then
  echo "{\"error\": \"task not found\", \"task_id\": \"$TASK_ID\"}"
  exit 1
fi

# Read metadata
META="$(cat "$TASK_DIR/task-meta.json" 2>/dev/null || echo "{}")"
STATUS="$(echo "$META" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null || echo "unknown")"

# Read latest output
OUTPUT=""
if [[ -f "$TASK_DIR/task-output.txt" ]]; then
  OUTPUT="$(tail -n "$TAIL_LINES" "$TASK_DIR/task-output.txt" 2>/dev/null || echo "")"
fi

# Build response
python3 -c "
import json, sys
meta = json.loads('''$META''') if '''$META''' else {}
result = {
  'task_id': '$TASK_ID',
  'status': '$STATUS',
  'name': meta.get('name', ''),
  'latest_output': sys.stdin.read().strip(),
}
if '$STATUS' == 'completed' or '$STATUS' == 'failed':
  result['exit_code'] = meta.get('exit_code')
  result['completed_at'] = meta.get('completed_at')
print(json.dumps(result, ensure_ascii=False, indent=2))
" <<< "$OUTPUT"
