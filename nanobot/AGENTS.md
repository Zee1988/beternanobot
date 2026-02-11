<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-11 | Updated: 2026-02-11 -->

# nanobot

## Purpose

The main Python package containing all core functionality for the nanobot AI assistant. This is the heart of the framework, implementing the agent loop, chat channels, LLM providers, tools, and configuration management.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initialization with version and logo |
| `__main__.py` | Entry point for `python -m nanobot` |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `agent/` | Core agent logic: loop, context, memory, tools (see `agent/AGENTS.md`) |
| `bus/` | Message bus for decoupled communication (see `bus/AGENTS.md`) |
| `channels/` | Chat platform integrations (see `channels/AGENTS.md`) |
| `cli/` | Command-line interface (see `cli/AGENTS.md`) |
| `config/` | Configuration loading and schema (see `config/AGENTS.md`) |
| `cron/` | Scheduled task service (see `cron/AGENTS.md`) |
| `heartbeat/` | Periodic heartbeat service (see `heartbeat/AGENTS.md`) |
| `providers/` | LLM provider implementations (see `providers/AGENTS.md`) |
| `session/` | Conversation session management (see `session/AGENTS.md`) |
| `skills/` | Skill definitions loaded by the agent (see `skills/AGENTS.md`) |
| `utils/` | Utility functions and helpers (see `utils/AGENTS.md`) |

## For AI Agents

### Working In This Directory

- All imports should be relative within the package or use `nanobot.` prefix
- Each subdirectory has an `__init__.py` for clean exports
- Follow existing patterns for new modules

### Testing Requirements

- Unit tests go in `tests/` at project root
- Use `pytest-asyncio` for async functions
- Mock external APIs in tests

### Common Patterns

- Abstract base classes in `base.py` files define interfaces
- Concrete implementations follow the base class contract
- Configuration is passed via Pydantic models
- Async/await for all I/O operations
- Loguru logger for all logging

## Dependencies

### Internal

- All subdirectories depend on `config/` for settings
- `agent/` depends on `bus/`, `providers/`, `session/`
- `channels/` depend on `bus/` for message routing

### External

- `loguru` - Logging
- `pydantic` - Data validation
- `asyncio` - Async runtime

<!-- MANUAL: -->
