"""Observation hook for recording tool execution results."""

import hashlib
import time

from .redact import redact_text, safe_preview
from .worker import IndexTask, IndexWorker


class ObservationHook:
    """
    Records tool execution observations for memory indexing.

    v5 fixes:
    - safe_preview parameter: max_chars (not max_len) (#18)
    - doc_id includes timestamp + content hash for uniqueness (#24)
    - Skip low-value periodic file reads (HEARTBEAT.md etc.)
    """

    # Files that are read periodically and produce low-value observations
    _SKIP_PATHS = {"HEARTBEAT.md", "heartbeat.md"}

    def __init__(self, worker: IndexWorker):
        self.worker = worker

    def _should_skip(self, tool_name: str, arguments: dict) -> bool:
        """Skip low-value periodic observations to improve signal-to-noise ratio."""
        if tool_name == "read_file":
            path = arguments.get("path", "")
            filename = path.rsplit("/", 1)[-1] if "/" in path else path
            if filename in self._SKIP_PATHS:
                return True
        return False

    async def on_tool_executed(
        self, tool_name: str, arguments: dict, result: object
    ) -> None:
        if self._should_skip(tool_name, arguments):
            return

        # v5 fix: max_chars not max_len (#18)
        preview = safe_preview(result, max_chars=500)
        text = redact_text(f"Tool: {tool_name}\nArgs: {arguments}\nResult: {preview}")

        # v5 fix: unique doc_id with timestamp + content hash (#24)
        ts = int(time.time() * 1000)
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:8]
        source_path = f"tool:{tool_name}:{ts}:{content_hash}"

        await self.worker.submit(IndexTask(
            source_type="observation",
            source_path=source_path,
            text=text,
            title=f"tool_exec:{tool_name}",
        ))
