#!/usr/bin/env bash
# notify-openclaw.sh — Claude Code Stop/SessionEnd hook callback
# Reads event from stdin, collects results, notifies OpenClaw Gateway.
set -euo pipefail

DATA_DIR="$HOME/.claude/data/claude-code-results"
LOCK_FILE="$HOME/.claude/data/.hook-lock"

# Read hook event from stdin
EVENT_JSON="$(cat)"
SESSION_ID="$(echo "$EVENT_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('session_id',''))" 2>/dev/null || echo "")"
EVENT_TYPE="$(echo "$EVENT_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('event',''))" 2>/dev/null || echo "")"
CWD="$(echo "$EVENT_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('cwd',''))" 2>/dev/null || echo "")"

# 30s dedup lock to prevent Stop+SessionEnd double-fire
if [[ -f "$LOCK_FILE" ]]; then
  LOCK_AGE=$(( $(date +%s) - $(stat -f %m "$LOCK_FILE" 2>/dev/null || echo 0) ))
  if [[ $LOCK_AGE -lt 30 ]]; then
    exit 0
  fi
fi
touch "$LOCK_FILE"

# Find the most recent running task
LATEST_TASK=""
LATEST_TIME=0
for meta_file in "$DATA_DIR"/*/task-meta.json; do
  [[ -f "$meta_file" ]] || continue
  STATUS="$(python3 -c "import json; print(json.load(open('$meta_file')).get('status',''))" 2>/dev/null || echo "")"
  if [[ "$STATUS" == "running" ]]; then
    CREATED="$(stat -f %m "$meta_file" 2>/dev/null || echo 0)"
    if [[ "$CREATED" -gt "$LATEST_TIME" ]]; then
      LATEST_TIME="$CREATED"
      LATEST_TASK="$(dirname "$meta_file")"
    fi
  fi
done

# If no running task found, try most recent task dir
if [[ -z "$LATEST_TASK" ]]; then
  LATEST_TASK="$(ls -td "$DATA_DIR"/*/ 2>/dev/null | head -1)"
fi

[[ -z "$LATEST_TASK" ]] && exit 0

TASK_ID="$(basename "$LATEST_TASK")"

# Collect output
OUTPUT=""
if [[ -f "$LATEST_TASK/task-output.txt" ]]; then
  OUTPUT="$(tail -c 4000 "$LATEST_TASK/task-output.txt")"
fi

# Read task metadata
TASK_NAME=""
TASK_GROUP=""
if [[ -f "$LATEST_TASK/task-meta.json" ]]; then
  TASK_NAME="$(python3 -c "import json; print(json.load(open('$LATEST_TASK/task-meta.json')).get('name',''))" 2>/dev/null || echo "")"
  TASK_GROUP="$(python3 -c "import json; print(json.load(open('$LATEST_TASK/task-meta.json')).get('group',''))" 2>/dev/null || echo "")"
  # Update meta status
  python3 -c "
import json
with open('$LATEST_TASK/task-meta.json','r') as f: d=json.load(f)
d['status']='completed'
d['completed_at']='$(date -u +%Y-%m-%dT%H:%M:%SZ)'
with open('$LATEST_TASK/task-meta.json','w') as f: json.dump(d,f,indent=2)
" 2>/dev/null || true
fi

# Write result.json
python3 -c "
import json, sys
result = {
  'session_id': '$SESSION_ID',
  'task_id': '$TASK_ID',
  'task_name': '$TASK_NAME',
  'timestamp': '$(date -u +%Y-%m-%dT%H:%M:%SZ)',
  'output': sys.stdin.read(),
  'status': 'completed',
  'event': '$EVENT_TYPE'
}
with open('$LATEST_TASK/result.json', 'w') as f:
  json.dump(result, f, indent=2)
" <<< "$OUTPUT" 2>/dev/null || true

# Send result to Telegram via openclaw message send
if [[ -n "$TASK_GROUP" ]] && command -v openclaw &>/dev/null; then
  SUMMARY="✅ Task ${TASK_NAME:-$TASK_ID} completed."
  if [[ -n "$OUTPUT" ]]; then
    SUMMARY="$SUMMARY

${OUTPUT: -500}"
  fi
  openclaw message send --channel telegram -t "$TASK_GROUP" -m "$SUMMARY" 2>/dev/null || true
fi

# Cleanup lock
rm -f "$LOCK_FILE" 2>/dev/null || true

