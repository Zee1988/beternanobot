---
name: claude-code
description: Dispatch coding tasks to Claude Code asynchronously. Returns task_id immediately, progress reported via hooks.
metadata: {"clawdbot":{"emoji":"🤖","requires":{"bins":["python3","claude"]}}}
---

# Claude Code

Dispatch coding tasks to Claude Code. Tasks run in background, progress is auto-reported to the user's channel.

## IMPORTANT: How to Use

**ALWAYS use the dispatch script below. NEVER call `claude` CLI directly.**

When a user asks you to use Claude Code for a task, you MUST:
1. Use `bash {baseDir}/scripts/dispatch-claude-code.sh` with the `--report` flag
2. Pass the user's channel and chat_id via `--report <channel>:<chat_id>` so progress is auto-reported
3. Use `--permission-mode bypassPermissions` for fully automated execution
4. Tell the user the task has been dispatched and they will receive progress updates

## Dispatch Command

```bash
bash {baseDir}/scripts/dispatch-claude-code.sh \
  -p "coding task prompt" \
  -w /path/to/workdir \
  -n "task-name" \
  --permission-mode bypassPermissions \
  --report <channel>:<chat_id>
```

## Options

| Flag | Description |
|------|-------------|
| `-p, --prompt` | **(required)** The coding task prompt |
| `-w, --workdir` | Working directory (default: `~/.nanobot/workspace`) |
| `-n, --name` | Human-readable task name |
| `--permission-mode` | Use `bypassPermissions` for auto mode |
| `--report` | **(recommended)** `channel:chat_id` for progress updates |
| `--model` | Model override |
| `--agent-teams` | Enable Agent Teams mode |
| `-g, --group` | Legacy: Telegram group for openclaw notification |

## Channel Mapping for --report

| Channel | Format | Example |
|---------|--------|---------|
| Feishu  | `feishu:<open_id>` | `feishu:ou_dd016da88e4105cfd8e2c170d4b2a696` |
| Telegram | `telegram:<chat_id>` | `telegram:7635862678` |

## Examples

### Feishu user asks to write code
```bash
bash {baseDir}/scripts/dispatch-claude-code.sh \
  -p "Create a racing game with canvas" \
  -w ~/.nanobot/workspace/racing-game \
  -n "racing-game" \
  --permission-mode bypassPermissions \
  --report feishu:ou_dd016da88e4105cfd8e2c170d4b2a696
```

### Telegram user asks to fix a bug
```bash
bash {baseDir}/scripts/dispatch-claude-code.sh \
  -p "Fix the login bug in auth.ts" \
  -w ~/projects/myapp \
  -n "fix-login" \
  --permission-mode bypassPermissions \
  --report telegram:7635862678
```

## What Happens After Dispatch

1. Script returns `task_id` immediately
2. Claude Code runs in background
3. Progress watcher auto-sends updates to the user's channel every 60s
4. Final result (success/failure) is sent when task completes
5. **You do NOT need to poll or check status** — the watcher handles everything

## Notes

- Results saved to `~/.claude/data/claude-code-results/<task_id>/`
- Requires `claude` CLI installed and authenticated
- Default timeout: 600 seconds
