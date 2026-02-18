# Claude Code Hooks

Claude Code 任务派发和回调 Hook 系统。

## 包含内容

- `scripts/claude_code_run.py` - Claude Code 运行器
- `scripts/dispatch-claude-code.sh` - 任务派发脚本
- `hooks/notify-agi.sh` - Stop/SessionEnd Hook
- `nanobot-skill/` - nanobot 集成（可选）
- `settings-hooks.json` - Claude Code 配置

## 安装

```bash
./install.sh
```

## 使用

### 派发任务

```bash
~/.claude/scripts/dispatch-claude-code.sh -p "Create a hello.py file" -n my-task
```

### 参数

| 参数 | 描述 |
|------|------|
| `-p` | 任务提示（必填） |
| `-n` | 任务名称 |
| `-w` | 工作目录 |
| `--agent-teams` | 启用 Agent Teams |
| `--permission-mode` | 权限模式 |

### 结果

任务结果保存在：
- `~/.claude/data/claude-code-results/latest.json`
- `~/.claude/data/claude-code-results/pending-wake.json`

## nanobot 集成

如果使用 nanobot，skill 已配置为使用此系统。当用户请求使用 Claude Code 时，nanobot 会自动调用 dispatch 脚本。

## 依赖

- python3
- claude CLI
- jq (用于 settings.json 合并)
