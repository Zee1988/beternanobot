<!-- Generated: 2026-02-11 | Updated: 2026-02-11 -->

# nanobot-ai

## Purpose

An ultra-lightweight personal AI assistant framework (~4,000 lines of core agent code). Nanobot provides a minimal yet powerful foundation for building AI-powered chatbots that can integrate with multiple messaging platforms, execute tools, maintain memory, and schedule tasks.

## Key Files

| File | Description |
|------|-------------|
| `pyproject.toml` | Python package configuration, dependencies, and build settings |
| `README.md` | Comprehensive documentation with setup guides and feature overview |
| `Dockerfile` | Container configuration for Docker deployment |
| `COMMUNICATION.md` | Community links (Feishu, WeChat, Discord) |
| `SECURITY.md` | Security policy and vulnerability reporting |
| `LICENSE` | MIT license |
| `core_agent_lines.sh` | Script to count core agent lines of code |
| `.gitignore` | Git ignore patterns |
| `.dockerignore` | Docker ignore patterns |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `nanobot/` | Main Python package with all core functionality (see `nanobot/AGENTS.md`) |
| `bridge/` | TypeScript bridge for WhatsApp integration via Node.js (see `bridge/AGENTS.md`) |
| `tests/` | Test suite for the project (see `tests/AGENTS.md`) |
| `workspace/` | Runtime workspace with agent configuration files (see `workspace/AGENTS.md`) |
| `case/` | Demo GIFs showcasing features |

## For AI Agents

### Working In This Directory

- This is a Python 3.11+ project using `hatch` as the build backend
- Install with `pip install -e .` for development
- Configuration lives in `~/.nanobot/config.json`
- The CLI entry point is `nanobot` (defined in pyproject.toml)

### Testing Requirements

- Run `pytest` for Python tests
- Tests are in `tests/` directory
- Use `pytest-asyncio` for async test support

### Common Patterns

- Pydantic for configuration schema validation
- Abstract base classes define interfaces (providers, channels, tools)
- Async/await throughout for non-blocking I/O
- Loguru for structured logging
- Message bus pattern for decoupled communication

## Dependencies

### External (Key Python Packages)

| Package | Purpose |
|---------|---------|
| `litellm` | Unified LLM API interface |
| `pydantic` | Configuration validation |
| `typer` | CLI framework |
| `httpx` | Async HTTP client |
| `websockets` | WebSocket support |
| `loguru` | Logging |
| `python-telegram-bot` | Telegram integration |
| `slack-sdk` | Slack integration |
| `dingtalk-stream` | DingTalk integration |
| `lark-oapi` | Feishu/Lark integration |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Chat Channels                           │
│  (Telegram, Discord, WhatsApp, Feishu, DingTalk, Slack...)  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     Message Bus                             │
│              (InboundMessage / OutboundMessage)             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     Agent Loop                              │
│  (Context → LLM → Tool Execution → Response)                │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     LLM Providers                           │
│  (OpenRouter, DeepSeek, vLLM, Moonshot, Qwen...)            │
└─────────────────────────────────────────────────────────────┘
```

<!-- MANUAL: -->
