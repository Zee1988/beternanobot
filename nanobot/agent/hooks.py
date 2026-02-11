"""Lightweight hook manager for agent lifecycle events."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from loguru import logger

HookFn = Callable[..., Awaitable[None]]


class HookManager:
    """
    Lightweight hook manager.

    All hooks are non-blocking (fire-and-forget or queue submission).
    Exceptions in hooks are logged but never propagate to the caller.
    """

    def __init__(self):
        self._hooks: dict[str, list[HookFn]] = {}

    def register(self, event: str, fn: HookFn) -> None:
        self._hooks.setdefault(event, []).append(fn)

    async def emit(self, event: str, **kwargs: Any) -> None:
        for fn in self._hooks.get(event, []):
            try:
                await fn(**kwargs)
            except Exception as e:
                logger.error(f"Hook {fn.__name__} failed on {event}: {e}")
