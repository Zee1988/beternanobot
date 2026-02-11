<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-11 | Updated: 2026-02-11 -->

# config

## Purpose

Configuration management. Defines the schema for all configuration options using Pydantic models and provides loading/validation from `~/.nanobot/config.json`.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Package exports |
| `loader.py` | Configuration loading from JSON file |
| `schema.py` | Pydantic models for all configuration options |

## For AI Agents

### Working In This Directory

- All config models are defined in `schema.py`
- Each channel has its own config model (e.g., `TelegramConfig`, `DiscordConfig`)
- Provider configs define API keys and base URLs
- Agent config defines model and behavior settings

### Configuration Structure

```json
{
  "providers": {
    "openrouter": { "apiKey": "..." }
  },
  "agents": {
    "defaults": { "model": "anthropic/claude-opus-4-5" }
  },
  "channels": {
    "telegram": { "enabled": true, "token": "..." }
  }
}
```

### Adding New Configuration

1. Add a Pydantic model in `schema.py`
2. Add to parent config model
3. Document default values
4. Update README with example

### Testing Requirements

- Test validation of required fields
- Test default value handling
- Test loading from file with missing fields

### Common Patterns

```python
class NewConfig(BaseModel):
    enabled: bool = False
    api_key: str = ""
    options: list[str] = Field(default_factory=list)
```

- BaseModel for all configs
- Default values for optional fields
- Field with default_factory for mutable defaults
- Sensitive fields (api keys, tokens) are strings

## Dependencies

### Internal

- Used by all other modules for configuration

### External

- `pydantic` - Data validation
- `pydantic-settings` - Settings management

<!-- MANUAL: -->
