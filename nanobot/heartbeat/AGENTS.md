<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-11 | Updated: 2026-02-11 -->

# heartbeat

## Purpose

Periodic heartbeat service. Runs a background task that periodically checks `HEARTBEAT.md` in the workspace and processes any tasks defined there. This enables recurring automated workflows.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Package exports |
| `service.py` | `HeartbeatService` implementation |

## For AI Agents

### Working In This Directory

- The heartbeat runs every 30 minutes by default
- Reads `workspace/HEARTBEAT.md` for task definitions
- Tasks are markdown checkboxes that the agent processes
- Used for recurring checks (calendar, weather, inbox, etc.)

### Task Format

```markdown
- [ ] Check calendar and remind of upcoming events
- [ ] Scan inbox for urgent emails
- [ ] Check weather forecast for today
```

### Testing Requirements

- Mock the file system for `HEARTBEAT.md`
- Test task parsing and execution
- Verify interval timing

### Common Patterns

- Background asyncio task with sleep interval
- Reads file, sends to agent for processing
- Completes silently if no tasks defined

## Dependencies

### Internal

- `nanobot.agent` - Process heartbeat tasks
- `nanobot.bus` - Send messages

### External

- `asyncio` - Background task scheduling

<!-- MANUAL: -->
