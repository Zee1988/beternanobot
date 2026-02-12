"""Context compression utilities — pure functions, no classes."""
import copy
import json

import litellm

from nanobot.storage.chunker import estimate_tokens

# ── Constants (not exposed as config) ──────────────────────
SOFT_TRIM_RATIO = 0.3       # single tool result token ratio cap
HARD_TRIM_RATIO = 0.5       # all tool results token ratio cap
SOFT_TRIM_HEAD_LINES = 20   # soft trim keep first N lines
SOFT_TRIM_TAIL_LINES = 10   # soft trim keep last N lines
DEFAULT_CONTEXT_WINDOW = 128_000
REASONING_KEEP_THRESHOLD = 2000  # reasoning_content keep threshold (chars)

def get_context_window(model: str, override: int | None = None) -> int:
    """Get model context window size. Priority: override > litellm > default."""
    if override is not None:
        return override
    try:
        info = litellm.get_model_info(model)
        return info.get("max_input_tokens") or info.get("max_tokens") or DEFAULT_CONTEXT_WINDOW
    except Exception:
        return DEFAULT_CONTEXT_WINDOW


def count_tokens(messages: list[dict], model: str | None = None) -> int:
    """Estimate total token count for messages. Prefer litellm, fallback to heuristic."""
    if model:
        try:
            return litellm.token_counter(model=model, messages=messages)
        except Exception:
            pass
    total = 0
    for msg in messages:
        content = msg.get("content") or ""
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += estimate_tokens(part.get("text", ""))
                elif isinstance(part, dict) and part.get("type") == "image_url":
                    total += 300
        if "tool_calls" in msg:
            total += estimate_tokens(json.dumps(msg["tool_calls"], ensure_ascii=False))
        rc = msg.get("reasoning_content")
        if rc and isinstance(rc, str):
            total += estimate_tokens(rc)
    return int(total * 1.10)

def trim_tool_result(result: str, max_chars: int = 15_000) -> str:
    """Soft trim a single tool result: keep head + tail, omit middle."""
    if len(result) <= max_chars:
        return result
    lines = result.split("\n")
    if len(lines) <= SOFT_TRIM_HEAD_LINES + SOFT_TRIM_TAIL_LINES:
        return result[:max_chars] + f"\n...[已省略 {len(result) - max_chars} 字符]"
    head = "\n".join(lines[:SOFT_TRIM_HEAD_LINES])
    tail = "\n".join(lines[-SOFT_TRIM_TAIL_LINES:])
    omitted = len(lines) - SOFT_TRIM_HEAD_LINES - SOFT_TRIM_TAIL_LINES
    trimmed = f"{head}\n\n...[已省略 {omitted} 行]...\n\n{tail}"
    if len(trimmed) > max_chars:
        trimmed = trimmed[:max_chars] + "\n...[已截断]"
    return trimmed


def _find_safe_cut_point(messages: list[dict], keep_last_n: int) -> int:
    """Find safe cut point that doesn't break tool_call/tool pairing.
    Returns the start index of messages to keep."""
    if keep_last_n >= len(messages):
        return 0
    candidate = len(messages) - keep_last_n
    while candidate > 0:
        msg = messages[candidate]
        role = msg.get("role", "")
        if role == "user":
            break
        if role == "assistant" and "tool_calls" not in msg:
            break
        candidate -= 1
    return max(candidate, 0)


def _find_last_user_index(messages: list[dict]) -> int:
    """Find index of the last user message. Returns len(messages) if not found."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return len(messages)


def compress_messages(
    messages: list[dict],
    context_window: int,
    model: str | None = None,
) -> list[dict]:
    """
    Main compression entry point (called before each LLM call).
    Strategy:
    1. system prompt: never trim
    2. last round (from last user message to end): never trim
    3. reasoning_content: keep only last round, remove others above threshold
    4. tool results old→new: soft trim then hard trim (stop when under limit)
    5. long user/assistant messages: soft trim (>30000 chars)
    """
    result = []
    for msg in messages:
        if "tool_calls" in msg:
            result.append(copy.deepcopy(msg))
        else:
            result.append(msg.copy())

    last_user_idx = _find_last_user_index(result)

    # Step 1: reasoning_content — keep only last round
    for i, msg in enumerate(result):
        if i >= last_user_idx:
            continue
        rc = msg.get("reasoning_content")
        if rc and isinstance(rc, str) and len(rc) > REASONING_KEEP_THRESHOLD:
            del msg["reasoning_content"]

    # Step 2: soft trim tool results
    soft_token_limit = int(context_window * SOFT_TRIM_RATIO)
    for msg in result:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if isinstance(content, str) and estimate_tokens(content) > soft_token_limit:
                max_chars = soft_token_limit * 3
                msg["content"] = trim_tool_result(content, max_chars)

    # Step 3: hard trim — stop when under limit
    hard_token_limit = int(context_window * HARD_TRIM_RATIO)
    total_tool_tokens = sum(
        estimate_tokens(msg.get("content", ""))
        for msg in result
        if msg.get("role") == "tool"
    )
    if total_tool_tokens > hard_token_limit:
        protected_indices = set()
        for i in range(last_user_idx, len(result)):
            if result[i].get("role") == "tool":
                protected_indices.add(i)

        for i, msg in enumerate(result):
            if total_tool_tokens <= hard_token_limit:
                break
            if msg.get("role") == "tool" and i not in protected_indices:
                old_tokens = estimate_tokens(msg.get("content", ""))
                name = msg.get("name", "tool")
                orig_len = len(msg.get("content", ""))
                placeholder = f"[工具 {name} 的结果已省略，原始长度 {orig_len} 字符]"
                msg["content"] = placeholder
                new_tokens = estimate_tokens(placeholder)
                total_tool_tokens -= (old_tokens - new_tokens)

    # Step 4: long user/assistant message soft trim
    for i, msg in enumerate(result):
        if i == 0:
            continue
        if i >= last_user_idx:
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and len(content) > 30_000:
            keep = int(30_000 * 0.6)
            tail_len = int(30_000 * 0.2)
            msg["content"] = (
                content[:keep]
                + f"\n...[已省略 {len(content) - keep - tail_len} 字符]...\n"
                + content[-tail_len:]
            )

    return result


def emergency_compress(
    messages: list[dict],
    context_window: int,
    model: str | None = None,
) -> list[dict]:
    """
    Emergency compression (called after overflow):
    1. Replace all tool results with one-line summary
    2. Remove all reasoning_content
    3. Safe truncate to last ~5 rounds
    4. If still too large, truncate to last ~2 rounds
    """
    result = []
    if messages and messages[0].get("role") == "system":
        result.append(messages[0].copy())
        rest = messages[1:]
    else:
        rest = messages[:]

    cleaned = []
    for msg in rest:
        if "tool_calls" in msg:
            m = copy.deepcopy(msg)
        else:
            m = msg.copy()
        m.pop("reasoning_content", None)
        if m.get("role") == "tool":
            name = m.get("name", "tool")
            m["content"] = f"[{name} 结果已省略]"
        cleaned.append(m)

    cut = _find_safe_cut_point(cleaned, 10)
    recent = cleaned[cut:]
    result.extend(recent)

    estimated = count_tokens(result, model=model)
    if estimated > int(context_window * 0.6):
        result = result[:1] if result and result[0].get("role") == "system" else []
        cut2 = _find_safe_cut_point(cleaned, 4)
        result.extend(cleaned[cut2:])

    return result
