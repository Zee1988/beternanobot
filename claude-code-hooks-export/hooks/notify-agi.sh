#!/bin/bash
# Claude Code Stop Hook: 任务完成后通知 AGI
# 触发时机: Stop (生成停止) + SessionEnd (会话结束)
# 支持 Agent Teams: lead 完成后自动触发

set -uo pipefail

HOOK_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULT_DIR="$HOME/.claude/data/claude-code-results"
RESULT_DIR_OPENCLAW="$HOME/.openclaw/claude-code/data/results"
OPENCLAW_BIN="/Users/defi/.npm-global/bin/openclaw"

# Find the most recent task result directory
find_result_dir() {
    # Check both locations and use the most recently modified one
    local latest_dir=""
    local latest_time=0

    # Check primary location
    if [ -f "${RESULT_DIR}/task-meta.json" ]; then
        local time
        time=$(stat -f %m "${RESULT_DIR}/task-meta.json" 2>/dev/null || stat -c %Y "${RESULT_DIR}/task-meta.json" 2>/dev/null || echo 0)
        if [ "$time" -gt "$latest_time" ]; then
            latest_time=$time
            latest_dir="$RESULT_DIR"
        fi
    fi

    # Check openclaw location
    if [ -d "$RESULT_DIR_OPENCLAW" ]; then
        local latest
        latest=$(ls -td "$RESULT_DIR_OPENCLAW"/*/ 2>/dev/null | head -1)
        if [ -n "$latest" ] && [ -f "${latest}task-meta.json" ]; then
            local time
            time=$(stat -f %m "${latest}task-meta.json" 2>/dev/null || stat -c %Y "${latest}task-meta.json" 2>/dev/null || echo 0)
            if [ "$time" -gt "$latest_time" ]; then
                latest_time=$time
                latest_dir="$latest"
            fi
        fi
    fi

    if [ -n "$latest_dir" ]; then
        echo "$latest_dir"
    else
        echo "$RESULT_DIR"
    fi
}

RESULT_DIR=$(find_result_dir)
META_FILE="${RESULT_DIR}/task-meta.json"

mkdir -p "$RESULT_DIR"

LOG="${RESULT_DIR}/hook.log"

log() { echo "[$(date -Iseconds)] $*" >> "$LOG"; }

log "=== Hook fired ==="

# ---- 读 stdin ----
INPUT=""
if [ -t 0 ]; then
    log "stdin is tty, skip"
elif [ -e /dev/stdin ]; then
    INPUT=$(timeout 2 cat /dev/stdin 2>/dev/null || true)
fi

SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // "unknown"' 2>/dev/null || echo "unknown")
CWD=$(echo "$INPUT" | jq -r '.cwd // ""' 2>/dev/null || echo "")
EVENT=$(echo "$INPUT" | jq -r '.hook_event_name // "unknown"' 2>/dev/null || echo "unknown")

log "session=$SESSION_ID cwd=$CWD event=$EVENT"

# ---- 防重复：只处理第一个事件（Stop），跳过后续的 SessionEnd ----
LOCK_FILE="${RESULT_DIR}/.hook-lock"
LOCK_AGE_LIMIT=30  # 30秒内重复触发视为同一任务

if [ -f "$LOCK_FILE" ]; then
    # macOS: stat -f %m, Linux: stat -c %Y
    LOCK_TIME=$(stat -f %m "$LOCK_FILE" 2>/dev/null || stat -c %Y "$LOCK_FILE" 2>/dev/null || echo 0)
    NOW=$(date +%s)
    AGE=$(( NOW - LOCK_TIME ))
    if [ "$AGE" -lt "$LOCK_AGE_LIMIT" ]; then
        log "Duplicate hook within ${AGE}s, skipping"
        exit 0
    fi
fi
touch "$LOCK_FILE"

# ---- 读取 Claude Code 输出 ----
OUTPUT=""

# 等待 tee 管道 flush（hook 可能在 pipe 写完前触发）
sleep 1

# 来源1: task-output.txt (dispatch 脚本 tee 写入)
TASK_OUTPUT="${RESULT_DIR}/task-output.txt"
if [ -f "$TASK_OUTPUT" ] && [ -s "$TASK_OUTPUT" ]; then
    OUTPUT=$(tail -c 4000 "$TASK_OUTPUT")
    log "Output from task-output.txt (${#OUTPUT} chars)"
fi

# 来源2: /tmp/claude-code-output.txt
if [ -z "$OUTPUT" ] && [ -f "/tmp/claude-code-output.txt" ] && [ -s "/tmp/claude-code-output.txt" ]; then
    OUTPUT=$(tail -c 4000 /tmp/claude-code-output.txt)
    log "Output from /tmp fallback (${#OUTPUT} chars)"
fi

# 来源3: 工作目录
if [ -z "$OUTPUT" ] && [ -n "$CWD" ] && [ -d "$CWD" ]; then
    FILES=$(ls -1t "$CWD" 2>/dev/null | head -20 | tr '\n' ', ')
    OUTPUT="Working dir: ${CWD}\nFiles: ${FILES}"
    log "Output from dir listing"
fi

# ---- 读取任务元数据 ----
TASK_NAME="unknown"
TELEGRAM_GROUP=""

if [ -f "$META_FILE" ]; then
    # Support both task_name (new) and name (openclaw) fields
    TASK_NAME=$(jq -r '.task_name // .name // "unknown"' "$META_FILE" 2>/dev/null || echo "unknown")
    TELEGRAM_GROUP=$(jq -r '.telegram_group // ""' "$META_FILE" 2>/dev/null || echo "")
    log "Meta: task=$TASK_NAME group=$TELEGRAM_GROUP"
fi

# ---- 写入结果 JSON ----
jq -n \
    --arg sid "$SESSION_ID" \
    --arg ts "$(date -Iseconds)" \
    --arg cwd "$CWD" \
    --arg event "$EVENT" \
    --arg output "$OUTPUT" \
    --arg task "$TASK_NAME" \
    --arg group "$TELEGRAM_GROUP" \
    '{session_id: $sid, timestamp: $ts, cwd: $cwd, event: $event, output: $output, task_name: $task, telegram_group: $group, status: "done"}' \
    > "${RESULT_DIR}/latest.json" 2>/dev/null

log "Wrote latest.json"

# ---- 方式1: 直接发 Telegram 消息（如果有目标群组）----
if [ -n "$TELEGRAM_GROUP" ] && [ -x "$OPENCLAW_BIN" ]; then
    SUMMARY=$(echo "$OUTPUT" | tail -c 1000 | tr '\n' ' ')
    MSG="🤖 *Claude Code 任务完成*
📋 任务: ${TASK_NAME}
📝 结果摘要:
\`\`\`
${SUMMARY:0:800}
\`\`\`"

    "$OPENCLAW_BIN" message send \
        --channel telegram \
        --target "$TELEGRAM_GROUP" \
        --message "$MSG" 2>/dev/null && log "Sent Telegram message to $TELEGRAM_GROUP" || log "Telegram send failed"
fi

# ---- 方式2: 唤醒 AGI 主会话 ----
# 写入 wake 标记文件，AGI 在下次 heartbeat 时读取
WAKE_FILE="${RESULT_DIR}/pending-wake.json"
jq -n \
    --arg task "$TASK_NAME" \
    --arg group "$TELEGRAM_GROUP" \
    --arg ts "$(date -Iseconds)" \
    --arg summary "$(echo "$OUTPUT" | head -c 500 | tr '\n' ' ')" \
    '{task_name: $task, telegram_group: $group, timestamp: $ts, summary: $summary, processed: false}' \
    > "$WAKE_FILE" 2>/dev/null

log "Wrote pending-wake.json"

log "=== Hook completed ==="
exit 0
