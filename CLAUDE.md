# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nanobot (`nanobot-ai` on PyPI) is a lightweight personal AI assistant framework in Python 3.11+. It connects multiple chat platforms to LLM providers through an async message bus, with built-in tools for file ops, shell execution, web access, and task scheduling. The WhatsApp bridge is a separate Node.js/TypeScript component in `bridge/`.

## Commands

```bash
# Install (dev)
pip install -e ".[dev]"

# Run
nanobot onboard              # First-time config setup
nanobot agent -m "message"   # Single message
nanobot agent                # Interactive chat
nanobot gateway              # Start multi-channel gateway (long-running)
nanobot status               # System status

# Lint
ruff check nanobot/
ruff check --fix nanobot/

# Test
pytest tests/
pytest tests/test_tool_validation.py          # Single test file
pytest tests/test_tool_validation.py -k "test_name"  # Single test

# Docker
docker build -t nanobot .
docker run -v ~/.nanobot:/root/.nanobot nanobot gateway

# WhatsApp bridge (bridge/)
cd bridge && npm install && npm run build
```

## Architecture

```
Channels (telegram, discord, whatsapp, feishu, dingtalk, slack, email, qq, mochat)
    ↓ InboundMessage
MessageBus (async queues)
    ↓
AgentLoop (loop.py) → ContextBuilder → LLMProvider → tool calls → response
    ↓ OutboundMessage
MessageBus → Channel.send()
```

Key flow: channels receive messages → publish `InboundMessage` to bus → `AgentLoop` subscribes, builds context (history + memory + skills), calls LLM, executes tool calls in a loop (max 20 iterations) → publishes `OutboundMessage` back through bus → channel delivers.

### Core Abstractions

- **`BaseChannel`** (`channels/base.py`): ABC with `start()`, `stop()`, `send()`, `is_allowed()`. All 9 channel implementations follow this interface.
- **`Tool`** (`agent/tools/base.py`): ABC with `name`, `description`, `parameters` (JSON Schema), `execute()`, `validate_params()`, `to_schema()`. Tools auto-register via `ToolRegistry`.
- **`LLMProvider`** (`providers/base.py`): ABC with `chat()` returning `LLMResponse` (content + tool_calls). The `LiteLLMProvider` wraps all providers through litellm.
- **`MessageBus`** (`bus/queue.py`): Async pub/sub with `InboundMessage`/`OutboundMessage` dataclasses.
- **`SessionManager`** (`session/manager.py`): Per-chat conversation history keyed by `{channel}:{chat_id}`.

### Module Layout

| Directory | Purpose |
|-----------|---------|
| `nanobot/agent/` | AgentLoop, ContextBuilder, memory, skills, subagent |
| `nanobot/agent/tools/` | Built-in tools: filesystem, shell, web, message, spawn, cron |
| `nanobot/channels/` | Chat platform integrations (9 channels) |
| `nanobot/providers/` | LLM provider interface + LiteLLM implementation |
| `nanobot/bus/` | Message bus events and queue |
| `nanobot/session/` | Conversation session management |
| `nanobot/config/` | Pydantic config schema + loader |
| `nanobot/cli/` | Typer CLI app |
| `nanobot/cron/` | Scheduled task service |
| `nanobot/skills/` | Markdown-based skill definitions loaded into agent context |
| `bridge/` | Node.js WhatsApp bridge (WebSocket server) |
| `workspace/` | Runtime agent instructions (AGENTS.md, HEARTBEAT.md) |

## Configuration

Runtime config lives at `~/.nanobot/config.json` (Pydantic-validated via `config/schema.py`). Schema covers:
- `providers` — API keys/bases for OpenRouter, DeepSeek, Groq, Gemini, MiniMax, etc.
- `agents.defaults` — model selection, behavior
- `channels` — per-channel enable/token/allowFrom
- `tools` — workspace restrictions, exec limits

## Code Conventions

- Python 3.11+ with type hints (`str | None` union syntax)
- Async/await throughout — all channels, tools, and the agent loop are async
- Pydantic v2 for config validation, dataclasses for bus events
- Ruff for linting (line-length 100, rules: E, F, I, N, W; E501 ignored)
- pytest with `asyncio_mode = "auto"` — async tests run without explicit markers
- Loguru for structured logging (`from loguru import logger`)

## Adding a New Channel

1. Create `nanobot/channels/{name}.py` implementing `BaseChannel`
2. Add config model to `config/schema.py`
3. Register in `channels/manager.py`

## Adding a New Tool

1. Create class extending `Tool` in `nanobot/agent/tools/`
2. Implement `name`, `description`, `parameters` (JSON Schema), `execute()`
3. Import and register in `agent/loop.py` constructor
