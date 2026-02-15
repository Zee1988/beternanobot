"""Session queue manager for serialized message processing."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from loguru import logger

from nanobot.agent.run_manager import RunManager
from nanobot.bus.events import InboundMessage
from nanobot.config.schema import QueueConfig


@dataclass
class RunContext:
    run_id: str
    cancel_event: asyncio.Event
    queue_mode: str


@dataclass
class SessionLane:
    queue: deque[InboundMessage] = field(default_factory=deque)
    collect_buffer: list[InboundMessage] = field(default_factory=list)
    collect_task: asyncio.Task | None = None
    pending_steer: InboundMessage | None = None
    active_task: asyncio.Task | None = None
    cancel_event: asyncio.Event | None = None
    worker_task: asyncio.Task | None = None


ProcessFn = Callable[[InboundMessage, RunContext], Awaitable[str | None]]


class QueueManager:
    """Queue manager that serializes processing per session."""

    def __init__(
        self,
        process_message: ProcessFn,
        run_manager: RunManager,
        config: QueueConfig,
    ) -> None:
        self._process_message = process_message
        self._run_manager = run_manager
        self._config = config
        self._lanes: dict[str, SessionLane] = {}

    def _get_lane(self, session_key: str) -> SessionLane:
        if session_key not in self._lanes:
            self._lanes[session_key] = SessionLane()
        return self._lanes[session_key]

    async def enqueue(self, msg: InboundMessage, mode: str | None = None) -> None:
        if not self._config.enabled:
            await self._process_direct(msg, mode or self._config.mode)
            return

        session_key = msg.session_key
        lane = self._get_lane(session_key)
        queue_mode = mode or msg.queue_mode or self._config.mode

        if queue_mode == "collect":
            lane.collect_buffer.append(msg)
            if lane.collect_task and not lane.collect_task.done():
                lane.collect_task.cancel()
            lane.collect_task = asyncio.create_task(
                self._flush_collect(session_key)
            )
            return

        if queue_mode in {"steer", "steer-backlog"}:
            if lane.cancel_event:
                lane.cancel_event.set()
            if queue_mode == "steer-backlog" and lane.pending_steer:
                lane.pending_steer = msg
            else:
                lane.pending_steer = msg
            self._ensure_worker(session_key)
            return

        lane.queue.append(msg)
        self._ensure_worker(session_key)

    async def wait_idle(self, session_key: str) -> None:
        lane = self._lanes.get(session_key)
        if not lane:
            return
        while True:
            tasks = []
            if lane.collect_task and not lane.collect_task.done():
                tasks.append(lane.collect_task)
            if lane.worker_task and not lane.worker_task.done():
                tasks.append(lane.worker_task)
            if not tasks:
                return
            await asyncio.gather(*tasks)

    def has_active_runs(self) -> bool:
        return any(
            lane.active_task and not lane.active_task.done()
            for lane in self._lanes.values()
        )

    async def _process_direct(self, msg: InboundMessage, queue_mode: str) -> None:
        run_id = self._run_manager.create_run(msg.session_key)
        cancel_event = asyncio.Event()
        context = RunContext(run_id=run_id, cancel_event=cancel_event, queue_mode=queue_mode)
        self._run_manager.mark_started(run_id)
        try:
            result = await self._process_message(msg, context)
            self._run_manager.mark_completed(run_id, result)
        except Exception as exc:
            self._run_manager.mark_failed(run_id, str(exc))

    def _ensure_worker(self, session_key: str) -> None:
        lane = self._get_lane(session_key)
        if lane.worker_task and not lane.worker_task.done():
            return
        lane.worker_task = asyncio.create_task(self._run_lane(session_key))

    async def _run_lane(self, session_key: str) -> None:
        lane = self._get_lane(session_key)
        try:
            while True:
                next_msg = self._next_message(lane)
                if not next_msg:
                    break

                run_id = self._run_manager.create_run(session_key)
                cancel_event = asyncio.Event()
                lane.cancel_event = cancel_event
                context = RunContext(run_id=run_id, cancel_event=cancel_event, queue_mode=next_msg.queue_mode or self._config.mode)

                self._run_manager.mark_started(run_id)
                lane.active_task = asyncio.create_task(
                    self._process_message(next_msg, context)
                )
                try:
                    result = await lane.active_task
                    if cancel_event.is_set():
                        self._run_manager.mark_cancelled(run_id, result)
                    else:
                        self._run_manager.mark_completed(run_id, result)
                except asyncio.CancelledError:
                    self._run_manager.mark_cancelled(run_id, "cancelled")
                    raise
                except Exception as exc:
                    self._run_manager.mark_failed(run_id, str(exc))
                finally:
                    lane.active_task = None
                    lane.cancel_event = None
        finally:
            lane.worker_task = None

    def _next_message(self, lane: SessionLane) -> InboundMessage | None:
        if lane.pending_steer:
            msg = lane.pending_steer
            lane.pending_steer = None
            return msg
        if lane.queue:
            return lane.queue.popleft()
        return None

    async def _flush_collect(self, session_key: str) -> None:
        lane = self._get_lane(session_key)
        try:
            await asyncio.sleep(self._config.debounce_ms / 1000)
        except asyncio.CancelledError:
            return

        buffer = list(lane.collect_buffer)
        lane.collect_buffer.clear()
        lane.collect_task = None

        if not buffer:
            return

        merged = self._merge_collect(buffer)
        lane.queue.append(merged)
        self._ensure_worker(session_key)

    @staticmethod
    def _merge_collect(buffer: list[InboundMessage]) -> InboundMessage:
        base = buffer[0]
        content = "\n".join(msg.content for msg in buffer)
        metadata = dict(base.metadata)
        metadata["collected_count"] = len(buffer)
        return InboundMessage(
            channel=base.channel,
            sender_id=base.sender_id,
            chat_id=base.chat_id,
            content=content,
            media=base.media,
            metadata=metadata,
            queue_mode=base.queue_mode,
        )
