"""Run lifecycle manager."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable
from uuid import uuid4

from nanobot.agent.run_events import RunEvent, RunStatus

Subscriber = Callable[[RunEvent], Awaitable[None] | None]


@dataclass
class RunRecord:
    """Record for a single run."""

    id: str
    session_key: str
    status: RunStatus = RunStatus.QUEUED
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    result: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    done_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)


class RunManager:
    """Manage run lifecycle state and waiters."""

    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._subscribers: list[Subscriber] = []
        self._lock = asyncio.Lock()

    def create_run(self, session_key: str, metadata: dict[str, Any] | None = None) -> str:
        run_id = uuid4().hex
        record = RunRecord(id=run_id, session_key=session_key, metadata=metadata or {})
        self._runs[run_id] = record
        self.emit(RunEvent(run_id=run_id, kind="created", status=record.status))
        return run_id

    def get(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    def subscribe(self, callback: Subscriber) -> None:
        self._subscribers.append(callback)

    def emit(self, event: RunEvent) -> None:
        for cb in list(self._subscribers):
            try:
                result = cb(event)
                if inspect.isawaitable(result):
                    asyncio.create_task(result)
            except Exception:
                continue

    def mark_started(self, run_id: str) -> None:
        record = self._runs.get(run_id)
        if not record:
            return
        record.status = RunStatus.RUNNING
        record.started_at = datetime.now()
        self.emit(RunEvent(run_id=run_id, kind="started", status=record.status))

    def mark_completed(self, run_id: str, result: str | None = None) -> None:
        record = self._runs.get(run_id)
        if not record:
            return
        record.status = RunStatus.COMPLETED
        record.ended_at = datetime.now()
        record.result = result
        record.done_event.set()
        self.emit(RunEvent(run_id=run_id, kind="completed", status=record.status, content=result))

    def mark_failed(self, run_id: str, error: str | None = None) -> None:
        record = self._runs.get(run_id)
        if not record:
            return
        record.status = RunStatus.FAILED
        record.ended_at = datetime.now()
        record.result = error
        record.done_event.set()
        self.emit(RunEvent(run_id=run_id, kind="failed", status=record.status, content=error))

    def mark_cancelled(self, run_id: str, reason: str | None = None) -> None:
        record = self._runs.get(run_id)
        if not record:
            return
        record.status = RunStatus.CANCELLED
        record.ended_at = datetime.now()
        record.result = reason
        record.done_event.set()
        self.emit(RunEvent(run_id=run_id, kind="cancelled", status=record.status, content=reason))

    async def wait(self, run_id: str, timeout: float | None = None) -> str | None:
        record = self._runs.get(run_id)
        if not record:
            raise KeyError(f"Run {run_id} not found")
        if record.done_event.is_set():
            return record.result
        await asyncio.wait_for(record.done_event.wait(), timeout=timeout)
        return record.result
