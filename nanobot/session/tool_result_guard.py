"""Write-time tool result guard for session integrity.

Inspired by openclaw's session-tool-result-guard.ts: intercepts message
writes to ensure every assistant tool_call has a matching tool result
before other messages are persisted. Synthesizes missing results for
unmatched tool calls.
"""

from typing import Any

from loguru import logger

# Max characters for a single tool result content
HARD_MAX_TOOL_RESULT_CHARS = 30_000

_TRUNCATION_SUFFIX = "\n\n[Output truncated. Use offset/limit parameters for large results.]"


def cap_tool_result_size(content: str) -> str:
    """Truncate oversized tool result content.

    Tries to cut at a newline boundary for cleaner output.
    """
    if not content or len(content) <= HARD_MAX_TOOL_RESULT_CHARS:
        return content

    budget = HARD_MAX_TOOL_RESULT_CHARS - len(_TRUNCATION_SUFFIX)
    # Try to cut at a newline within 80% of budget
    cut_search_start = int(budget * 0.8)
    newline_pos = content.rfind("\n", cut_search_start, budget)
    cut_at = newline_pos if newline_pos > cut_search_start else budget

    return content[:cut_at] + _TRUNCATION_SUFFIX


def make_missing_tool_result(tool_call_id: str, tool_name: str = "unknown") -> dict[str, Any]:
    """Create a synthetic tool result for an unmatched tool call."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": "[Tool result missing — call may have been interrupted]",
    }


class ToolResultGuard:
    """Tracks pending tool calls and ensures result pairing on writes.

    Usage:
        guard = ToolResultGuard()

        # For each message being saved to session:
        messages_to_save = guard.process(msg)
        for m in messages_to_save:
            session.add_message(...)
    """

    def __init__(self) -> None:
        # Map of tool_call_id -> tool_name for calls awaiting results
        self._pending: dict[str, str] = {}

    @property
    def pending_ids(self) -> set[str]:
        return set(self._pending.keys())

    def flush_pending(self) -> list[dict[str, Any]]:
        """Synthesize missing results for all pending tool calls."""
        results = []
        for tc_id, tc_name in self._pending.items():
            logger.debug(f"Guard: synthesizing missing result for {tc_id} ({tc_name})")
            results.append(make_missing_tool_result(tc_id, tc_name))
        self._pending.clear()
        return results

    def process(self, msg: dict[str, Any]) -> list[dict[str, Any]]:
        """Process a message through the guard.

        Returns a list of messages to persist (may include synthetic
        results flushed before the current message).

        Args:
            msg: The message dict with at least "role".

        Returns:
            List of messages to write to session, in order.
        """
        role = msg.get("role")
        output: list[dict[str, Any]] = []

        if role == "assistant" and msg.get("tool_calls"):
            # Flush any stale pending calls first
            if self._pending:
                output.extend(self.flush_pending())

            # Register new pending tool calls
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id", "")
                tc_name = tc.get("function", {}).get("name", "unknown")
                if tc_id:
                    self._pending[tc_id] = tc_name

            output.append(msg)

        elif role == "tool":
            tc_id = msg.get("tool_call_id", "")
            if tc_id in self._pending:
                self._pending.pop(tc_id)
            # Cap oversized results
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > HARD_MAX_TOOL_RESULT_CHARS:
                msg = {**msg, "content": cap_tool_result_size(content)}
            output.append(msg)

        else:
            # Non-tool message — flush pending before persisting
            if self._pending:
                output.extend(self.flush_pending())
            output.append(msg)

        return output
