<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-11 | Updated: 2026-02-11 -->

# tools

## Purpose

Built-in tool implementations for the agent. Tools are capabilities the LLM can invoke to interact with the environment (files, shell, web, messaging, etc.).

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Package exports |
| `base.py` | `BaseTool` abstract class defining the tool interface |
| `registry.py` | `ToolRegistry` for managing available tools |
| `filesystem.py` | File tools: `ReadFileTool`, `WriteFileTool`, `EditFileTool`, `ListDirTool` |
| `shell.py` | Shell execution: `ExecTool` |
| `web.py` | Web tools: `WebSearchTool`, `WebFetchTool` |
| `message.py` | Messaging: `MessageTool` for sending responses |
| `spawn.py` | Background tasks: `SpawnTool` for subagents |
| `cron.py` | Scheduling: `CronTool` for task scheduling |

## For AI Agents

### Working In This Directory

- All tools extend `BaseTool` and implement `execute()`
- Tools define their JSON schema for the LLM
- Tools can be restricted (e.g., filesystem limited to workspace)
- Registry manages tool discovery and invocation

### Adding a New Tool

1. Create `newtool.py` implementing `BaseTool`
2. Define `name`, `description`, `parameters` (JSON schema)
3. Implement `async execute(**kwargs) -> str`
4. Register in `AgentLoop._register_default_tools()`

### Tool Interface

```python
class NewTool(BaseTool):
    name = "new_tool"
    description = "What this tool does"
    parameters = {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "..."}
        },
        "required": ["param1"]
    }

    async def execute(self, param1: str) -> str:
        # Do something
        return "result"
```

### Testing Requirements

- Test with valid and invalid parameters
- Test error handling
- Test security restrictions (workspace limits, etc.)

### Common Patterns

- Return string results for LLM to process
- Raise exceptions for errors (caught by loop)
- Optional `allowed_dir` for filesystem sandboxing
- Async execution for I/O operations

## Dependencies

### Internal

- `nanobot.config` - Tool configuration
- `nanobot.bus` - MessageTool uses bus

### External

| Package | Tool |
|---------|------|
| `httpx` | WebFetchTool |
| `readability-lxml` | WebFetchTool (HTML parsing) |

<!-- MANUAL: -->
