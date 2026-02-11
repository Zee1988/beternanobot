<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-11 | Updated: 2026-02-11 -->

# providers

## Purpose

LLM provider implementations. Each provider handles communication with a specific LLM API (OpenRouter, DeepSeek, vLLM, etc.). All providers implement the `LLMProvider` interface for consistent usage.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Package exports |
| `base.py` | `LLMProvider` abstract base class and `LLMResponse` dataclass |
| `registry.py` | `ProviderRegistry` for provider lookup by name |
| `litellm_provider.py` | Universal provider using LiteLLM library |
| `transcription.py` | Audio transcription provider (Whisper, etc.) |

## For AI Agents

### Working In This Directory

- `LLMProvider` defines the interface all providers must implement
- `LLMResponse` contains the response with content, tool calls, and usage
- `ToolCallRequest` represents a tool call from the LLM
- LiteLLM provider handles most APIs with unified interface

### Adding a New Provider

1. Create `newprovider.py` implementing `LLMProvider`
2. Implement `chat()` and `get_default_model()`
3. Register in `registry.py`
4. Add config in `config/schema.py`

### Testing Requirements

- Mock HTTP responses from provider APIs
- Test tool call parsing
- Test error handling for API failures

### Common Patterns

```python
class NewProvider(LLMProvider):
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        # Call provider API
        # Parse response into LLMResponse
        pass

    def get_default_model(self) -> str:
        return "default-model-name"
```

- Async HTTP with httpx
- Handle rate limiting and retries
- Support for reasoning_content (DeepSeek-R1, Kimi)

## Dependencies

### Internal

- `nanobot.config` - Provider configuration

### External

- `litellm` - Universal LLM interface
- `httpx` - Async HTTP client

<!-- MANUAL: -->
