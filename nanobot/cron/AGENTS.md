<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-11 | Updated: 2026-02-11 -->

# cron

## Purpose

Scheduled task service. Allows users to schedule one-time or recurring reminders and tasks that the agent will execute at specified times.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Package exports |
| `service.py` | `CronService` for scheduling and executing tasks |
| `types.py` | Type definitions for scheduled tasks |

## For AI Agents

### Working In This Directory

- Tasks are stored persistently and survive restarts
- One-time tasks execute once at specified time
- Recurring tasks use cron expressions
- Tasks can deliver messages to specific channels

### CLI Commands

```bash
nanobot cron add --name "reminder" --message "Hello" --at "2024-01-15T10:00:00"
nanobot cron list
nanobot cron remove --name "reminder"
```

### Testing Requirements

- Test task scheduling and persistence
- Test cron expression parsing
- Mock datetime for predictable tests

### Common Patterns

- Use `croniter` for cron expression parsing
- Tasks stored in `~/.nanobot/cron.json`
- Background asyncio task checks for due tasks

## Dependencies

### Internal

- `nanobot.bus` - Send messages when tasks trigger
- `nanobot.config` - Load cron configuration

### External

- `croniter` - Cron expression parsing
- `asyncio` - Background task scheduling

<!-- MANUAL: -->
