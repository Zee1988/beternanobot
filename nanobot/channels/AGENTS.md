<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-11 | Updated: 2026-02-11 -->

# channels

## Purpose

Chat platform integrations. Each channel connects to a messaging platform (Telegram, Discord, WhatsApp, etc.), listens for incoming messages, and sends outbound responses. All channels implement the `BaseChannel` interface.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Package exports and channel registration |
| `base.py` | `BaseChannel` abstract base class defining the interface |
| `manager.py` | `ChannelManager` for starting/stopping multiple channels |
| `telegram.py` | Telegram bot integration via python-telegram-bot |
| `discord.py` | Discord bot integration via raw WebSocket gateway |
| `whatsapp.py` | WhatsApp integration via Node.js bridge |
| `feishu.py` | Feishu/Lark integration via WebSocket |
| `dingtalk.py` | DingTalk integration via stream mode |
| `slack.py` | Slack integration via Slack SDK |
| `email.py` | Email integration via IMAP/SMTP |
| `qq.py` | QQ bot integration |
| `mochat.py` | Mochat (Claw-compatible) integration |

## For AI Agents

### Working In This Directory

- All channels extend `BaseChannel` and implement `start()`, `stop()`, `send()`
- Use `is_allowed()` to check sender permissions from config
- Forward messages to the bus via `_handle_message()`
- Each channel has its own config model in `nanobot/config/schema.py`

### Adding a New Channel

1. Create `newchannel.py` implementing `BaseChannel`
2. Add config model in `config/schema.py`
3. Add to `ChannelManager` in `manager.py`
4. Update `__init__.py` exports

### Testing Requirements

- Mock the message bus
- Mock external platform APIs
- Test `is_allowed()` with various allow lists

### Common Patterns

```python
class NewChannel(BaseChannel):
    name = "newchannel"

    async def start(self) -> None:
        # Connect to platform, listen for messages
        pass

    async def stop(self) -> None:
        # Disconnect, cleanup
        pass

    async def send(self, msg: OutboundMessage) -> None:
        # Send message via platform API
        pass
```

- Proxy support for platforms in restricted regions
- Allow lists for access control
- Graceful reconnection on connection loss

## Dependencies

### Internal

- `nanobot.bus` - Message bus
- `nanobot.config` - Channel configurations

### External

| Package | Channel |
|---------|---------|
| `python-telegram-bot` | Telegram |
| `slack-sdk` | Slack |
| `dingtalk-stream` | DingTalk |
| `lark-oapi` | Feishu |
| `websockets` | Discord, WhatsApp |
| `qq-botpy` | QQ |

<!-- MANUAL: -->
