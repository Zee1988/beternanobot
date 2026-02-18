#!/usr/bin/env bash
# dispatch-claude-code.sh — Async task dispatcher for Claude Code
# Launches Claude Code in background and immediately returns task_id.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$HOME/.claude/data/claude-code-results"

# Defaults
PROMPT=""
TASK_NAME=""
GROUP=""
WORKDIR=""
AGENT_TEAMS=""
PERMISSION_MODE="plan"
MODEL=""
REPORT_CHANNEL=""
REPORT_CHAT_ID=""
EXTRA_ARGS=()

usage() {
  echo "Usage: $0 -p <prompt> [-n name] [-g group] [-w workdir] [--report channel:chat_id] [--permission-mode MODE]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--prompt)       PROMPT="$2"; shift 2 ;;
    -n|--name)         TASK_NAME="$2"; shift 2 ;;
    -g|--group)        GROUP="$2"; shift 2 ;;
    -w|--workdir)      WORKDIR="$2"; shift 2 ;;
    --agent-teams)     AGENT_TEAMS="--agent-teams"; shift ;;
    --permission-mode) PERMISSION_MODE="$2"; shift 2 ;;
    --model)           MODEL="$2"; shift 2 ;;
    --report)          IFS=':' read -r REPORT_CHANNEL REPORT_CHAT_ID <<< "$2"; shift 2 ;;
    *)                 EXTRA_ARGS+=("$1"); shift ;;
  esac
done

[[ -z "$PROMPT" ]] && { echo "ERROR: -p <prompt> is required"; usage; }

# Generate task_id
TASK_ID="$(date +%Y%m%d-%H%M%S)-$(head -c 4 /dev/urandom | xxd -p)"
TASK_DIR="$DATA_DIR/$TASK_ID"
mkdir -p "$TASK_DIR"

# Write task metadata
cat > "$TASK_DIR/task-meta.json" <<METAEOF
{
  "task_id": "$TASK_ID",
  "prompt": $(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$PROMPT"),
  "name": "$TASK_NAME",
  "group": "$GROUP",
  "workdir": "${WORKDIR:-$(pwd)}",
  "status": "running",
  "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "completed_at": null,
  "exit_code": null
}
METAEOF

# Build runner args
RUNNER_ARGS=(-p "$PROMPT" --output-file "$TASK_DIR/task-output.txt")
[[ -n "$WORKDIR" ]] && RUNNER_ARGS+=(-w "$WORKDIR")
[[ -n "$PERMISSION_MODE" ]] && RUNNER_ARGS+=(--permission-mode "$PERMISSION_MODE")
[[ -n "$AGENT_TEAMS" ]] && RUNNER_ARGS+=(--agent-teams)
[[ -n "$MODEL" ]] && RUNNER_ARGS+=(--model "$MODEL")

# Launch Claude Code in background
(
  python3 "$SCRIPT_DIR/claude_code_run.py" "${RUNNER_ARGS[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>"$TASK_DIR/task-stderr.txt"
  EXIT_CODE=$?

  # Update task metadata with completion info
  python3 -c "
import json
with open('$TASK_DIR/task-meta.json', 'r') as f:
    d = json.load(f)
d['exit_code'] = $EXIT_CODE
d['completed_at'] = '$(date -u +%Y-%m-%dT%H:%M:%SZ)'
if $EXIT_CODE == 0:
    d['status'] = 'completed'
else:
    d['status'] = 'failed'
with open('$TASK_DIR/task-meta.json', 'w') as f:
    json.dump(d, f, indent=2)
" 2>/dev/null || true
) &

# Launch progress watcher if --report was specified
if [[ -n "$REPORT_CHANNEL" && -n "$REPORT_CHAT_ID" ]]; then
  python3 "$SCRIPT_DIR/progress-watcher.py" "$TASK_DIR" "$REPORT_CHANNEL" "$REPORT_CHAT_ID" \
    2>"$TASK_DIR/watcher-stderr.txt" &
fi

# Return task_id immediately (async core)
echo "{\"task_id\": \"$TASK_ID\", \"status\": \"dispatched\", \"task_dir\": \"$TASK_DIR\"}"
