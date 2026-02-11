<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-11 | Updated: 2026-02-11 -->

# cli

## Purpose

Command-line interface for nanobot. Provides commands for initialization, running the agent, managing scheduled tasks, and configuration.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Package exports |
| `commands.py` | Typer CLI application with all commands |

## For AI Agents

### Working In This Directory

- The CLI uses Typer for command parsing
- Entry point is `app` which is registered in pyproject.toml as `nanobot`
- Commands include: `onboard`, `agent`, `serve`, `cron`

### Available Commands

| Command | Description |
|---------|-----------|
| `nanobot onboard` | Initialize configuration |
| `nanobot agent -m "message"` | Send a message to the agent |
| `nanobot serve` | Start the bot with all enabled channels |
| `nanobot cron` | Manage scheduled tasks |

### Testing Requirements

- Use Typer's testing utilities
- Test command parsing and option handling
- Mock the agent and channels for integration tests

### Common Patterns

```python
@app.command()
def mycommand(option: str = typer.Option(...)):
    """Command description."""
    pass
```

- Rich console for formatted output
- Typer Options for command parameters
- Async functions called via `asyncio.run()`

## Dependencies

### Internal

- `nanobot.config` - Load configuration
- `nanobot.agent` - Agent loop
- `nanobot.channels` - Channel manager
- `nanobot.cron` - Cron service

### External

- `typer` - CLI framework
- `rich` - Console formatting

<!-- MANUAL: -->
