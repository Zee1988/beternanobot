"""Run lifecycle event types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class RunStatus(str, Enum):
    """Status of a run."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class RunEvent:
    """Lifecycle or output event for a run."""

    run_id: str
    kind: str
    status: RunStatus | None = None
    content: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
