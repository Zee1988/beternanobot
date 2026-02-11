<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-11 | Updated: 2026-02-11 -->

# bus

## Purpose

Message bus implementation for decoupled communication between components. Channels publish inbound messages to the bus, and the agent loop subscribes to process them. Outbound messages flow the reverse direction.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Package exports |
| `events.py` | `InboundMessage` and `OutboundMessage` dataclasses |
| `queue.py` | `MessageBus` implementation with async queues |

## For AI Agents

### Working In This Directory

- `InboundMessage` represents messages from users (via channels)
- `OutboundMessage` represents responses to send back
- `MessageBus` uses asyncio queues for pub/sub
- The `session_key` property on messages identifies conversations

### Testing Requirements

- Test queue behavior with multiple producers/consumers
- Verify message routing by channel and chat_id

### Common Patterns

```python
# Message flow:
InboundMessage(channel="telegram", sender_id="123", chat_id="456", content="Hello")
# → AgentLoop processes
# →
OutboundMessage(channel="telegram", chat_id="456", content="Hi there!")
```

- Dataclasses with default factories for optional fields
- Timestamp auto-set on message creation
- Metadata dict for channel-specific data

## Dependencies

### Internal

- Used by `nanobot.channels` (publish inbound)
- Used by `nanobot.agent` (subscribe inbound, publish outbound)

### External

- `asyncio` - Async queues
- `dataclasses` - Message definitions

<!-- MANUAL: -->
