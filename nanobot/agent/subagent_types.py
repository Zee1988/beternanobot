"""Subagent type definitions and state management."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any


class SubagentStatus(enum.Enum):
    """子代理生命周期状态 (幂等状态转换)."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

    def is_terminal(self) -> bool:
        return self in (
            SubagentStatus.COMPLETED,
            SubagentStatus.FAILED,
            SubagentStatus.TIMEOUT,
        )


@dataclass
class SubagentEntry:
    """单个子代理的跟踪记录."""
    task_id: str
    label: str
    task: str
    origin_channel: str
    origin_chat_id: str
    status: SubagentStatus = SubagentStatus.PENDING
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    result: str | None = None
    error: str | None = None

    def mark_running(self) -> None:
        if self.status.is_terminal():
            return  # 幂等: 终态不可逆
        self.status = SubagentStatus.RUNNING

    def mark_completed(self, result: str) -> None:
        if self.status.is_terminal():
            return
        self.status = SubagentStatus.COMPLETED
        self.result = result
        self.finished_at = time.time()

    def mark_failed(self, error: str) -> None:
        if self.status.is_terminal():
            return
        self.status = SubagentStatus.FAILED
        self.error = error
        self.finished_at = time.time()

    def mark_timeout(self) -> None:
        if self.status.is_terminal():
            return
        self.status = SubagentStatus.TIMEOUT
        self.error = "Task exceeded timeout limit"
        self.finished_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "label": self.label,
            "task": self.task,
            "status": self.status.value,
            "origin_channel": self.origin_channel,
            "origin_chat_id": self.origin_chat_id,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
        }
