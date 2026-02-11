"""Background periodic conversation summarization."""

import asyncio
from typing import Any, Awaitable, Callable

from loguru import logger

from .redact import redact_text
from .worker import IndexTask, IndexWorker

# LLM summarizer callback type
LLMSummarizer = Callable[[str], Awaitable[str]]


class SummaryTimer:
    """
    Background asyncio.Task for periodic conversation summaries.

    v5 enhancements:
    - mode="llm": calls LLM for real summarization
    - mode="concat": v4-compatible concatenation (default)
    - LLM timeout/failure auto-degrades to concat
    """

    LLM_TIMEOUT = 10.0  # LLM summarization timeout

    def __init__(
        self,
        worker: IndexWorker,
        interval: float = 300.0,
        mode: str = "concat",  # "concat" | "llm"
        llm_provider: Any = None,  # Must implement async summarize(text) -> str
    ):
        self.worker = worker
        self.interval = interval
        self.mode = mode
        self._llm_provider = llm_provider
        self._task: asyncio.Task | None = None
        self._messages: list[dict] = []

    def start(self):
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def feed_message(self, role: str, content: str):
        """Called from main flow, append to buffer (sync, microsecond-level)."""
        self._messages.append({"role": role, "content": content})

    async def _loop(self):
        while True:
            await asyncio.sleep(self.interval)
            if len(self._messages) < 4:
                continue
            try:
                await self._generate_summary()
            except Exception as e:
                logger.error(f"Summary generation failed: {e}")

    async def _generate_summary(self):
        """Generate summary from message buffer and submit for indexing."""
        msgs = self._messages.copy()
        self._messages.clear()

        # Build raw text from recent messages
        parts = []
        for m in msgs[-20:]:
            parts.append(f"{m['role']}: {m['content'][:300]}")
        raw_text = "\n".join(parts)

        # Try LLM summarization (#27)
        summary = None
        if self.mode == "llm" and self._llm_provider:
            try:
                summary = await asyncio.wait_for(
                    self._llm_provider.summarize(raw_text),
                    timeout=self.LLM_TIMEOUT,
                )
                logger.debug("LLM summary generated successfully")
            except asyncio.TimeoutError:
                logger.warning("LLM summary timed out, falling back to concat")
            except Exception as e:
                logger.warning(f"LLM summary failed ({e}), falling back to concat")

        # Fallback: concat mode
        if summary is None:
            summary = raw_text

        safe_summary = redact_text(summary)
        await self.worker.submit(IndexTask(
            source_type="summary",
            source_path="session_summary",
            text=safe_summary,
            title="conversation_summary",
        ))
        logger.debug(f"Summary generated from {len(msgs)} messages (mode={self.mode})")
