"""Tool call ID sanitization for provider compatibility.

Inspired by openclaw's tool-call-id.ts: sanitizes tool call IDs across
a message transcript to satisfy provider-specific constraints (e.g.
alphanumeric-only, fixed length) while preventing ID collisions.
"""

import hashlib
import re
import time
from typing import Any

_NON_ALNUM = re.compile(r"[^a-zA-Z0-9]")


def sanitize_tool_call_id(raw_id: str, mode: str = "strict") -> str:
    """Sanitize a single tool call ID.

    Args:
        raw_id: The original tool call ID.
        mode: "strict" (alphanumeric, any length) or
              "strict9" (exactly 9 alphanumeric chars, for Mistral).

    Returns:
        A sanitized ID string.
    """
    cleaned = _NON_ALNUM.sub("", raw_id or "")

    if mode == "strict9":
        if len(cleaned) == 9:
            return cleaned
        if len(cleaned) > 9:
            return cleaned[:9]
        # Too short — hash to get deterministic 9 chars
        if cleaned:
            h = hashlib.sha1(raw_id.encode()).hexdigest()[:9]
            # Ensure alphanumeric (sha1 hex is already)
            return h
        return "defaultid"

    # strict mode
    if not cleaned:
        return "sanitizedtoolid"
    if len(cleaned) > 40:
        return cleaned[:40]
    return cleaned


def _make_unique(base: str, used: set[str], mode: str = "strict") -> str:
    """Generate a unique ID that doesn't collide with the used set."""
    if base not in used:
        return base

    # Try hash suffix
    h = hashlib.sha1(base.encode()).hexdigest()[:8]
    max_len = 9 if mode == "strict9" else 40

    candidate = (base + h)[:max_len]
    if candidate not in used:
        return candidate

    # Numeric suffix fallback
    for i in range(2, 1000):
        suffix = str(i)
        candidate = (base + suffix)[:max_len]
        if candidate not in used:
            return candidate

    # Final fallback: timestamp
    ts = str(int(time.time() * 1000))[-8:]
    return (base + ts)[:max_len]


def sanitize_tool_call_ids(
    messages: list[dict[str, Any]],
    mode: str = "strict",
) -> list[dict[str, Any]]:
    """Sanitize all tool call IDs in a message transcript.

    Builds a transcript-wide stable mapping from original IDs to sanitized
    IDs, then rewrites all references. Returns the original list if no
    changes were needed.

    Args:
        messages: The message list (will NOT be mutated).
        mode: "strict" or "strict9".

    Returns:
        A new message list with sanitized IDs (or the original if unchanged).
    """
    # First pass: collect all tool call IDs that need rewriting
    id_map: dict[str, str] = {}
    used: set[str] = set()
    needs_rewrite = False

    for msg in messages:
        role = msg.get("role")

        if role == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id", "")
                if tc_id and tc_id not in id_map:
                    sanitized = sanitize_tool_call_id(tc_id, mode)
                    sanitized = _make_unique(sanitized, used, mode)
                    used.add(sanitized)
                    id_map[tc_id] = sanitized
                    if sanitized != tc_id:
                        needs_rewrite = True

        elif role == "tool":
            tc_id = msg.get("tool_call_id", "")
            if tc_id and tc_id not in id_map:
                sanitized = sanitize_tool_call_id(tc_id, mode)
                sanitized = _make_unique(sanitized, used, mode)
                used.add(sanitized)
                id_map[tc_id] = sanitized
                if sanitized != tc_id:
                    needs_rewrite = True

    if not needs_rewrite:
        return messages

    # Second pass: rewrite IDs immutably
    result = []
    for msg in messages:
        role = msg.get("role")

        if role == "assistant" and msg.get("tool_calls"):
            new_tcs = []
            for tc in msg["tool_calls"]:
                old_id = tc.get("id", "")
                new_id = id_map.get(old_id, old_id)
                new_tcs.append({**tc, "id": new_id})
            result.append({**msg, "tool_calls": new_tcs})

        elif role == "tool":
            old_id = msg.get("tool_call_id", "")
            new_id = id_map.get(old_id, old_id)
            result.append({**msg, "tool_call_id": new_id})

        else:
            result.append(msg)

    return result
