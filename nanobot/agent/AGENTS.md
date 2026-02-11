<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-11 | Updated: 2026-02-11 -->

# agent

## Purpose

The core agent implementation. Contains the main processing loop that receives messages, builds context, calls the LLM, executes tools, and sends responses. This is where the "intelligence" of nanobot lives.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Package exports |
| `loop.py` | Main `AgentLoop` class - the core processing engine |
| `context.py` | `ContextBuilder` for assembling LLM prompts with history, memory, skills |
| `memory.py` | Memory management for persistent agent knowledge |
| `skills.py` | Skill loading and execution |
| `subagent.py` | `SubagentManager` for spawning background tasks |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `tools/` | Built-in tool implementations (see `tools/AGENTS.md`) |

## For AI Agents

### Working In This Directory

- `AgentLoop` is the central class - understand it first
- Tools are registered in `_register_default_tools()`
- Context building happens in `ContextBuilder`
- The loop processes messages from the bus and calls the LLM provider

### Testing Requirements

- Mock the LLM provider for unit tests
- Mock the message bus for isolated testing
- Test tool execution with controlled inputs

### Common Patterns

```python
# The agent loop flow:
# 1. Receive InboundMessage from bus
# 2. Build context with ContextBuilder
# 3. Call LLM provider
# 4. Execute tool calls if present
# 5. Loop until no more tool calls
# 6. Send OutboundMessage to bus
```

- Max iterations limit prevents infinite loops
- Session manager tracks conversation history
- Subagents run as independent background tasks

## Dependencies

### Internal

- `nanobot.bus` - Message bus for communication
- `nanobot.providers` - LLM providers for inference
- `nanobot.session` - Session management
- `nanobot.config` - Configuration
- `nanobot.cron` - Scheduled task integration

### External

- `loguru` - Logging
- `asyncio` - Async runtime

<!-- MANUAL: -->
