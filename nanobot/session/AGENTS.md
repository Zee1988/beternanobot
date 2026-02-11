<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-11 | Updated: 2026-02-11 -->

# session

## Purpose

Conversation session management. Tracks conversation history per chat, manages context windows, and handles session persistence.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Package exports |
| `manager.py` | `SessionManager` for conversation tracking |

## For AI Agents

### Working In This Directory

- Sessions are keyed by `{channel}:{chat_id}`
- History is maintained per session
- Sessions can be persisted to disk
- Context window management prevents token overflow

### Session Key Format

```python
session_key = f"{channel}:{chat_id}"
# Examples:
# "telegram:123456789"
# "discord:987654321"
# "whatsapp:1234567890@s.whatsapp.net"
```

### Testing Requirements

- Test session creation and retrieval
- Test history truncation
- Test persistence and restoration

### Common Patterns

- In-memory session storage with optional persistence
- History as list of message dicts
- Automatic cleanup of old sessions

## Dependencies

### Internal

- Used by `nanobot.agent` for conversation tracking

### External

- `json` - Session serialization

<!-- MANUAL: -->
