#!/bin/bash
# dispatch-claude-code.sh — Dispatch a task to Claude Code with auto-callback
#
# Usage:
#   dispatch-claude-code.sh [OPTIONS] -p "your prompt here"
#
# Options:
#   -p, --prompt TEXT        Task prompt (required)
#   -n, --name NAME          Task name (for tracking)
#   -g, --group ID           Telegram group ID for result delivery
#   -s, --session KEY        Callback session key (AGI session to notify)
#   -w, --workdir DIR        Working directory for Claude Code
#   --agent-teams            Enable Agent Teams (lead + sub-agents)
#   --teammate-mode MODE     Agent Teams display mode (auto/in-process/tmux)
#   --permission-mode MODE   Claude Code permission mode
#   --allowed-tools TOOLS    Allowed tools string
#   --model MODEL            Model override
#
# The script:
#   1. Writes task metadata to task-meta.json (hook reads this)
#   2. Runs Claude Code via claude_code_run.py
#   3. When Claude Code finishes, Stop hook fires automatically
#   4. Hook reads meta, writes results, wakes AGI
#   5. AGI reads results and relays to Telegram group

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULT_DIR="$HOME/.claude/data/claude-code-results"
META_FILE="${RESULT_DIR}/task-meta.json"
OUTPUT_FILE="/tmp/claude-code-output.txt"
TASK_OUTPUT="${RESULT_DIR}/task-output.txt"
RUNNER="$SCRIPT_DIR/claude_code_run.py"

# Defaults
PROMPT=""
TASK_NAME="adhoc-$(date +%s)"
TELEGRAM_GROUP=""
CALLBACK_SESSION=""
WORKDIR="$HOME"
AGENT_TEAMS=""
TEAMMATE_MODE=""
PERMISSION_MODE=""
ALLOWED_TOOLS=""
MODEL=""

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--prompt) PROMPT="$2"; shift 2;;
        -n|--name) TASK_NAME="$2"; shift 2;;
        -g|--group) TELEGRAM_GROUP="$2"; shift 2;;
        -s|--session) CALLBACK_SESSION="$2"; shift 2;;
        -w|--workdir) WORKDIR="$2"; shift 2;;
        --agent-teams) AGENT_TEAMS="1"; shift;;
        --teammate-mode) TEAMMATE_MODE="$2"; shift 2;;
        --permission-mode) PERMISSION_MODE="$2"; shift 2;;
        --allowed-tools) ALLOWED_TOOLS="$2"; shift 2;;
        --model) MODEL="$2"; shift 2;;
        *) echo "Unknown option: $1" >&2; exit 1;;
    esac
done

if [ -z "$PROMPT" ]; then
    echo "Error: --prompt is required" >&2
    exit 1
fi

# ---- 1. Write task metadata ----
mkdir -p "$RESULT_DIR"

jq -n \
    --arg name "$TASK_NAME" \
    --arg group "$TELEGRAM_GROUP" \
    --arg session "$CALLBACK_SESSION" \
    --arg prompt "$PROMPT" \
    --arg workdir "$WORKDIR" \
    --arg ts "$(date -Iseconds)" \
    --arg agent_teams "${AGENT_TEAMS:-0}" \
    '{task_name: $name, telegram_group: $group, callback_session: $session, prompt: $prompt, workdir: $workdir, started_at: $ts, agent_teams: ($agent_teams == "1"), status: "running"}' \
    > "$META_FILE"

echo "📋 Task metadata written: $META_FILE"
echo "   Task: $TASK_NAME"
echo "   Group: ${TELEGRAM_GROUP:-none}"
echo "   Agent Teams: ${AGENT_TEAMS:-no}"

# ---- 2. Clear previous output ----
> "$OUTPUT_FILE"
> "$TASK_OUTPUT"

# ---- 3. Build runner command ----
CMD=(python3 "$RUNNER" -p "$PROMPT" --cwd "$WORKDIR")

if [ -n "$AGENT_TEAMS" ]; then
    CMD+=(--agent-teams)
fi
if [ -n "$TEAMMATE_MODE" ]; then
    CMD+=(--teammate-mode "$TEAMMATE_MODE")
fi
if [ -n "$PERMISSION_MODE" ]; then
    CMD+=(--permission-mode "$PERMISSION_MODE")
fi
if [ -n "$ALLOWED_TOOLS" ]; then
    CMD+=(--allowedTools "$ALLOWED_TOOLS")
fi

# ---- 4. Set environment ----
# Allow nested Claude Code sessions
unset CLAUDECODE

if [ -n "$MODEL" ]; then
    export ANTHROPIC_MODEL="$MODEL"
fi

# ---- 5. Run Claude Code (output tee'd for hook) ----
echo "🚀 Launching Claude Code..."
echo "   Command: ${CMD[*]}"
echo ""

# Use tee to capture output while also displaying it
"${CMD[@]}" 2>&1 | tee "$TASK_OUTPUT"
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "✅ Claude Code exited with code: $EXIT_CODE"
echo "   Hook should have fired automatically."
echo "   Results: ${RESULT_DIR}/latest.json"

# Update meta with completion
if [ -f "$META_FILE" ]; then
    jq --arg code "$EXIT_CODE" --arg ts "$(date -Iseconds)" \
        '. + {exit_code: ($code | tonumber), completed_at: $ts, status: "done"}' \
        "$META_FILE" > "${META_FILE}.tmp" && mv "${META_FILE}.tmp" "$META_FILE"
fi

exit $EXIT_CODE
