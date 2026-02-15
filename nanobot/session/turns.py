"""Provider-specific turn validation and repair.

Inspired by openclaw's turns.ts: validates and repairs conversation turn
sequences to satisfy provider-specific ordering constraints.

- Anthropic: cannot handle consecutive user messages
- Gemini: cannot handle consecutive assistant messages
- MiniMax: similar to Anthropic (strict alternation)
"""

from typing import Any


def validate_anthropic_turns(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fix turn ordering for Anthropic-style APIs.

    Merges consecutive user messages by concatenating their content.
    Providers like Anthropic and MiniMax require strict user->assistant
    alternation and reject consecutive user messages.

    Args:
        messages: The message list (not mutated).

    Returns:
        A new list with consecutive user messages merged.
    """
    if not messages:
        return messages

    result: list[dict[str, Any]] = []
    last_role: str | None = None

    for msg in messages:
        role = msg.get("role")
        if not role:
            result.append(msg)
            continue

        if role == "user" and last_role == "user" and result:
            # Merge into the previous user message
            prev = result[-1]
            merged_content = _merge_content(prev.get("content", ""), msg.get("content", ""))
            result[-1] = {**prev, "content": merged_content}
        else:
            result.append(msg)

        last_role = role

    return result


def validate_gemini_turns(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fix turn ordering for Gemini API.

    Merges consecutive assistant messages by concatenating their content.
    Gemini requires strict alternation and rejects consecutive assistant
    messages.

    Args:
        messages: The message list (not mutated).

    Returns:
        A new list with consecutive assistant messages merged.
    """
    if not messages:
        return messages

    result: list[dict[str, Any]] = []
    last_role: str | None = None

    for msg in messages:
        role = msg.get("role")
        if not role:
            result.append(msg)
            continue

        if role == "assistant" and last_role == "assistant" and result:
            # Merge into the previous assistant message
            prev = result[-1]
            merged_content = _merge_content(prev.get("content", ""), msg.get("content", ""))
            # Keep tool_calls from the message that has them
            merged = {**prev, "content": merged_content}
            if msg.get("tool_calls") and not prev.get("tool_calls"):
                merged["tool_calls"] = msg["tool_calls"]
            result[-1] = merged
        else:
            result.append(msg)

        last_role = role

    return result


def _merge_content(a: Any, b: Any) -> str:
    """Merge two message contents into one string."""
    str_a = a if isinstance(a, str) else str(a) if a else ""
    str_b = b if isinstance(b, str) else str(b) if b else ""
    if str_a and str_b:
        return f"{str_a}\n\n{str_b}"
    return str_a or str_b
