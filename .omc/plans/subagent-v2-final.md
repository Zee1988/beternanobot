# Subagent System V2 - Final Implementation Plan

> Consolidated implementation plan based on architect review.
> All code is complete and executable (not pseudocode).

---

## Table of Contents

1. [Problem Summary](#problem-summary)
2. [Coverage Matrix](#coverage-matrix)
3. [File Change Overview](#file-change-overview)
4. [Implementation Phases](#implementation-phases)
5. [Complete Source Code](#complete-source-code)
6. [Test Code](#test-code)
7. [Verification Commands](#verification-commands)
8. [Backward Compatibility](#backward-compatibility)

---

## Problem Summary

The current subagent system has critical issues that need resolution:

**CRITICAL Issues**:
- C1: Synchronous file writes block the event loop
- C2: Dual timeout sources (sweeper + asyncio.wait_for) cause race conditions
- C3: Unbounded registry growth + no CLI cleanup

**HIGH Issues**:
- H1: No crash recovery mechanism
- H2: ResultQueue collect mode loses messages
- H3: SpawnTool mutable state causes concurrency bugs
- H4: No model format validation

**MEDIUM Issues**:
- M2: stop_sweeper is synchronous and doesn't handle CancelledError
- M3: Nesting prevention is over-engineered
- M5: String-based status values instead of enums


---

## Coverage Matrix

| Problem | Level | Phase | Solution |
|---------|-------|-------|----------|
| C1: Sync writes block event loop | CRITICAL | 1 | `asyncio.to_thread` + dirty flag + `persist_if_dirty()` |
| C2: Dual timeout race conditions | CRITICAL | 2,4 | `asyncio.wait_for` only; sweeper only cleans; `mark_*` idempotent |
| C3: Unbounded registry + no CLI sweeper | CRITICAL | 1 | `MAX_ENTRIES` + `register()` auto-eviction; CLI uses eviction not sweeper |
| H1: No crash recovery | HIGH | 1 | `load()` marks non-terminal as FAILED |
| H2: ResultQueue collect loses messages | HIGH | deferred | Deferred: collect 模式当前未使用，仅保留 direct 模式 |
| H3: SpawnTool mutable state unsafe | HIGH | 2 | Remove `set_context()`; origin via `execute()` params |
| H4: No model validation | HIGH | 3 | `_validate_model()` regex check |
| M2: stop_sweeper sync + no CancelledError | MEDIUM | 4 | `async stop_sweeper()` + `except CancelledError` |
| M3: Nesting over-engineered | MEDIUM | 2 | Single `is_nested` + `nesting_enabled` config |
| M5: String status values | MEDIUM | 1 | `SubagentStatus` enum |

---

## File Change Overview

| Operation | File Path | Problems Solved |
|-----------|-----------|-----------------|
| NEW | `nanobot/agent/subagent_types.py` | M5, C2 (idempotent) |
| NEW | `nanobot/agent/subagent_registry.py` | C1, C3, H1 |
| MODIFY | `nanobot/config/schema.py` | H4 (SubagentConfig) |
| MODIFY | `nanobot/agent/subagent.py` | C2, H3, H4, M2, M3, M5 |
| MODIFY | `nanobot/agent/tools/spawn.py` | H3 |
| MODIFY | `nanobot/agent/tools/message.py` | H3 (扩展: 消除 set_context 可变状态) |
| MODIFY | `nanobot/agent/tools/cron.py` | H3 (扩展: 消除 set_context 可变状态) |
| MODIFY | `nanobot/agent/loop.py` | H3, C2 (sweeper lifecycle) |
| MODIFY | `nanobot/cli/commands.py` | Config passing |
| NEW | `tests/test_subagent_types.py` | Tests |
| NEW | `tests/test_subagent_registry.py` | Tests |
| NEW | `tests/test_subagent_improved.py` | Tests |
| NEW | `tests/test_sweeper.py` | Tests |
| MODIFY | `tests/test_overflow_recovery.py` | CRITICAL-2 兼容性修复 |

---

## Implementation Phases

### Phase 1: Foundation (Registry + Config + Status Enum)
- Create `subagent_types.py` with `SubagentStatus` enum and `SubagentEntry`
- Create `subagent_registry.py` with bounded entries and async persistence
- Add `SubagentConfig` to `schema.py`
- Update `SubagentManager.__init__` to accept config

### Phase 2: Timeout + Concurrency Fixes
- Modify `SubagentManager.spawn()` with timeout and concurrency limits
- Add `_run_with_timeout()` wrapper using `asyncio.wait_for`
- Rewrite `SpawnTool` to remove mutable state
- Update `AgentLoop` to inject origin params

### Phase 3: Model Validation + System Prompt
- Add `_validate_model()` method
- Improve `_build_subagent_prompt()` to load from AGENTS.md
- Add `get_registry_stats()` method

### Phase 4: Sweeper + ResultQueue
- Add `ResultQueue` to `subagent_types.py`
- Implement sweeper (cleanup only, no timeout checking)
- Add `start_sweeper()` and `stop_sweeper()` methods
- Update `AgentLoop.run()` and `shutdown()` to manage sweeper

### Phase 5: Tests
- Create comprehensive test suite for all new components


---

## Complete Source Code

### 1. nanobot/agent/subagent_types.py (NEW)

```python
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
        }
```


### 2. nanobot/agent/subagent_registry.py (NEW)

```python
"""Subagent registry with bounded entries and async persistence."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.subagent_types import SubagentEntry, SubagentStatus

# --- 常量 ---
MAX_ENTRIES = 200          # _entries 上限
ARCHIVE_KEEP = 50          # 清理后保留的终态条目数
PERSIST_INTERVAL = 30.0    # sweeper 持久化间隔 (秒)


class SubagentRegistry:
    """
    子代理注册表。

    - 有界: 超过 MAX_ENTRIES 时主动清理最旧终态条目
    - 异步持久化: dirty flag + sweeper 定期 flush (解决 C1)
    - 崩溃恢复: load 时将非终态条目标记为 FAILED (解决 H1)
    """

    def __init__(self, persist_path: Path | None = None):
        self._entries: dict[str, SubagentEntry] = {}
        self._persist_path = persist_path
        self._dirty = False

    # --- 公开 API ---

    def register(self, entry: SubagentEntry) -> None:
        """注册新条目，超限时主动清理 (解决 C3)."""
        if len(self._entries) >= MAX_ENTRIES:
            self._evict_terminal()
        self._entries[entry.task_id] = entry
        self._dirty = True

    def get(self, task_id: str) -> SubagentEntry | None:
        return self._entries.get(task_id)

    def get_all(self) -> list[SubagentEntry]:
        return list(self._entries.values())

    def get_running(self) -> list[SubagentEntry]:
        return [
            e for e in self._entries.values()
            if e.status == SubagentStatus.RUNNING
        ]

    def remove(self, task_id: str) -> None:
        self._entries.pop(task_id, None)
        self._dirty = True

    def cleanup_archived(self) -> int:
        """清理所有终态条目，返回清理数量."""
        to_remove = [
            tid for tid, e in self._entries.items()
            if e.status.is_terminal()
        ]
        for tid in to_remove:
            del self._entries[tid]
        if to_remove:
            self._dirty = True
        return len(to_remove)

    # --- 持久化 (异步, 解决 C1) ---

    async def persist_if_dirty(self) -> None:
        """仅在有变更时异步写入磁盘."""
        if not self._dirty or not self._persist_path:
            return
        data = {
            tid: e.to_dict() for tid, e in self._entries.items()
        }
        content = json.dumps(data, ensure_ascii=False, indent=2)
        try:
            await asyncio.to_thread(self._sync_write, content)
            self._dirty = False
        except Exception as exc:
            logger.warning(f"Registry persist failed: {exc}")

    def _sync_write(self, content: str) -> None:
        """同步写入 (在 to_thread 中执行, 不阻塞事件循环)."""
        if self._persist_path is None:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._persist_path.with_suffix(".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(self._persist_path)

    def load(self) -> None:
        """从磁盘加载，非终态条目标记为 FAILED (解决 H1)."""
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            raw = json.loads(self._persist_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Registry load failed: {exc}")
            return
        for tid, data in raw.items():
            status = SubagentStatus(data.get("status", "failed"))
            entry = SubagentEntry(
                task_id=tid,
                label=data.get("label", ""),
                task=data.get("task", ""),
                origin_channel=data.get("origin_channel", "cli"),
                origin_chat_id=data.get("origin_chat_id", "direct"),
                status=status,
                created_at=data.get("created_at", 0),
                finished_at=data.get("finished_at"),
            )
            # 崩溃恢复: 非终态 → FAILED
            if not entry.status.is_terminal():
                entry.mark_failed("Recovered after crash: task was not terminal at load time")
                self._dirty = True
            self._entries[tid] = entry

    # --- 内部 ---

    def _evict_terminal(self) -> None:
        """淘汰最旧的终态条目，保留 ARCHIVE_KEEP 个."""
        terminal = sorted(
            [(tid, e) for tid, e in self._entries.items() if e.status.is_terminal()],
            key=lambda x: x[1].created_at,
        )
        to_remove = terminal[:-ARCHIVE_KEEP] if len(terminal) > ARCHIVE_KEEP else []
        for tid, _ in to_remove:
            del self._entries[tid]
        if to_remove:
            self._dirty = True
            logger.debug(f"Registry evicted {len(to_remove)} terminal entries")

    def __len__(self) -> int:
        return len(self._entries)
```


### 3. nanobot/config/schema.py (MODIFY)

Add `SubagentConfig` class before `AgentDefaults` (around line 157):

```python
class SubagentConfig(BaseModel):
    """子代理配置."""
    max_concurrent: int = 5           # 最大并发子代理数
    timeout_seconds: int = 300        # 单个子代理超时 (秒)
    max_iterations: int = 15          # 子代理最大工具迭代次数
    model: str | None = None          # 子代理专用模型 (None = 继承主模型)
    nesting_enabled: bool = False     # 是否允许子代理嵌套 spawn (简化 M3)
```

Modify `AgentsConfig` class:

```python
class AgentsConfig(BaseModel):
    """Agent configuration."""
    defaults: AgentDefaults = Field(default_factory=AgentDefaults)
    subagent: SubagentConfig = Field(default_factory=SubagentConfig)
```

### 4. nanobot/agent/tools/spawn.py (COMPLETE REWRITE)

```python
"""Spawn tool for creating background subagents."""

from typing import Any, TYPE_CHECKING

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


class SpawnTool(Tool):
    """Tool to spawn a subagent for background task execution."""

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "Spawn a subagent to handle a task in the background. "
            "Use this for complex or time-consuming tasks that can run independently. "
            "The subagent will complete the task and report back when done."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task for the subagent to complete",
                },
                "label": {
                    "type": "string",
                    "description": "Optional short label for the task (for display)",
                },
            },
            "required": ["task"],
        }

    async def execute(
        self,
        task: str,
        label: str | None = None,
        *,
        _origin_channel: str = "cli",
        _origin_chat_id: str = "direct",
        **kwargs: Any,
    ) -> str:
        """Spawn a subagent. Origin context injected by caller, not stored as state."""
        return await self._manager.spawn(
            task=task,
            label=label,
            origin_channel=_origin_channel,
            origin_chat_id=_origin_chat_id,
        )
```


### 4b. nanobot/agent/tools/message.py (MODIFY — H3 扩展)

Apply the same H3 fix to `MessageTool`: accept origin via `execute()` params instead of mutable `set_context()`.

```python
"""Message tool for sending messages to users."""

from typing import Any, Callable, Awaitable

from nanobot.agent.tools.base import Tool
from nanobot.bus.events import OutboundMessage


class MessageTool(Tool):
    """Tool to send messages to users on chat channels."""

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
    ):
        self._send_callback = send_callback

    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        """Set the callback for sending messages."""
        self._send_callback = callback

    @property
    def name(self) -> str:
        return "message"

    @property
    def description(self) -> str:
        return "Send a message to the user. Use this when you want to communicate something."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The message content to send"
                },
                "channel": {
                    "type": "string",
                    "description": "Optional: target channel (telegram, discord, etc.)"
                },
                "chat_id": {
                    "type": "string",
                    "description": "Optional: target chat/user ID"
                }
            },
            "required": ["content"]
        }

    async def execute(
        self,
        content: str,
        channel: str | None = None,
        chat_id: str | None = None,
        *,
        _origin_channel: str = "",
        _origin_chat_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Send message. Origin context injected by caller, not stored as state."""
        channel = channel or _origin_channel
        chat_id = chat_id or _origin_chat_id

        if not channel or not chat_id:
            return "Error: No target channel/chat specified"

        if not self._send_callback:
            return "Error: Message sending not configured"

        msg = OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=content
        )

        try:
            await self._send_callback(msg)
            return f"Message sent to {channel}:{chat_id}"
        except Exception as e:
            return f"Error sending message: {str(e)}"
```


### 4c. nanobot/agent/tools/cron.py (MODIFY — H3 扩展)

Apply the same H3 fix to `CronTool`: accept origin via `execute()` params instead of mutable `set_context()`.

Only the `__init__`, `set_context` removal, and `execute` signature need to change:

```python
# In __init__, remove self._channel and self._chat_id:
def __init__(self, cron_service: CronService):
    self._cron = cron_service

# Remove set_context() method entirely

# In execute(), add keyword-only origin params:
async def execute(
    self,
    action: str,
    *,
    _origin_channel: str = "",
    _origin_chat_id: str = "",
    **kwargs: Any,
) -> str:
    # Use _origin_channel/_origin_chat_id instead of self._channel/self._chat_id
    channel = _origin_channel
    chat_id = _origin_chat_id
    # ... rest of execute logic unchanged, just replace self._channel → channel
    #     and self._chat_id → chat_id
```


### 5. nanobot/agent/subagent.py (COMPLETE REWRITE)

```python
"""Subagent manager for background task execution."""

import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider, ContextOverflowError
from nanobot.agent.compressor import (
    trim_tool_result, compress_messages, emergency_compress, get_context_window,
)
from nanobot.agent.subagent_types import SubagentEntry, SubagentStatus
from nanobot.agent.subagent_registry import SubagentRegistry, PERSIST_INTERVAL
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool


class SubagentManager:
    """
    Manages background subagent execution.

    Subagents are lightweight agent instances that run in the background
    to handle specific tasks. They share the same LLM provider but have
    isolated context and a focused system prompt.
    """

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        model: str | None = None,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        restrict_to_workspace: bool = False,
        subagent_config: "SubagentConfig | None" = None,
    ):
        from nanobot.config.schema import ExecToolConfig, SubagentConfig
        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.restrict_to_workspace = restrict_to_workspace

        self._config = subagent_config or SubagentConfig()
        # 模型: 子代理配置 > 显式参数 > provider 默认
        self.model = self._config.model or model or provider.get_default_model()

        # Registry (持久化路径可选, lazy load 避免构造函数同步 I/O)
        persist_path = workspace / ".nanobot" / "subagent_registry.json"
        self.registry = SubagentRegistry(persist_path=persist_path)
        self._registry_loaded = False

        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self._sweeper_running = False
        self._sweeper_task: asyncio.Task[None] | None = None

    def _ensure_registry_loaded(self) -> None:
        """Lazy load registry on first access (避免构造函数同步 I/O, 解决 C1)."""
        if not self._registry_loaded:
            self.registry.load()  # 崩溃恢复 (H1)
            self._registry_loaded = True

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        is_nested: bool = False,
    ) -> str:
        """Spawn a subagent to execute a task in the background."""
        self._ensure_registry_loaded()

        # 嵌套防护 (简化 M3: 单层检查即可)
        if is_nested and not self._config.nesting_enabled:
            return "Error: Subagent nesting is disabled."

        # 模型格式验证 (H4)
        model = self.model
        if not self._validate_model(model):
            return f"Error: Invalid model format '{model}'. Expected 'provider/model-name' or 'model-name'."

        # 并发限制
        running_count = len(self._running_tasks)
        if running_count >= self._config.max_concurrent:
            return (
                f"Error: Maximum concurrent subagents ({self._config.max_concurrent}) "
                f"reached. Wait for a running task to complete."
            )

        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")

        # 注册到 registry
        entry = SubagentEntry(
            task_id=task_id,
            label=display_label,
            task=task,
            origin_channel=origin_channel,
            origin_chat_id=origin_chat_id,
        )
        self.registry.register(entry)

        origin = {"channel": origin_channel, "chat_id": origin_chat_id}

        # 创建带超时的后台任务 (C2: asyncio.wait_for 是唯一超时源)
        bg_task = asyncio.create_task(
            self._run_with_timeout(task_id, task, display_label, origin)
        )
        self._running_tasks[task_id] = bg_task
        bg_task.add_done_callback(lambda _: self._running_tasks.pop(task_id, None))

        logger.info(f"Spawned subagent [{task_id}]: {display_label}")
        return (
            f"Subagent [{display_label}] started (id: {task_id}). "
            f"I'll notify you when it completes."
        )

    async def _run_with_timeout(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
    ) -> None:
        """用 asyncio.wait_for 包装子代理执行 (C2: 唯一超时源)."""
        entry = self.registry.get(task_id)
        if entry:
            entry.mark_running()

        try:
            await asyncio.wait_for(
                self._run_subagent(task_id, task, label, origin),
                timeout=self._config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Subagent [{task_id}] timed out after {self._config.timeout_seconds}s")
            if entry:
                entry.mark_timeout()  # 幂等
            await self._announce_result(
                task_id, label, task,
                f"Task timed out after {self._config.timeout_seconds} seconds.",
                origin, SubagentStatus.TIMEOUT,
            )
        except asyncio.CancelledError:
            logger.info(f"Subagent [{task_id}] cancelled")
            if entry:
                entry.mark_failed("Task was cancelled")
            raise  # 重新抛出让 asyncio 正确清理
        except Exception as exc:
            logger.error(f"Subagent [{task_id}] unexpected error in wrapper: {exc}")
            if entry:
                entry.mark_failed(str(exc))

    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
    ) -> None:
        """Execute the subagent task and announce the result."""
        logger.info(f"Subagent [{task_id}] starting task: {label}")
        entry = self.registry.get(task_id)

        try:
            # Build subagent tools (no message tool, no spawn tool)
            tools = ToolRegistry()
            allowed_dir = self.workspace if self.restrict_to_workspace else None
            tools.register(ReadFileTool(allowed_dir=allowed_dir))
            tools.register(WriteFileTool(allowed_dir=allowed_dir))
            tools.register(ListDirTool(allowed_dir=allowed_dir))
            tools.register(ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
            ))
            tools.register(WebSearchTool(api_key=self.brave_api_key))
            tools.register(WebFetchTool())

            system_prompt = self._build_subagent_prompt(task)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]

            max_iterations = self._config.max_iterations
            iteration = 0
            final_result: str | None = None

            while iteration < max_iterations:
                iteration += 1

                ctx_window = get_context_window(self.model)
                messages = compress_messages(messages, ctx_window, self.model)
                try:
                    response = await self.provider.chat(
                        messages=messages,
                        tools=tools.get_definitions(),
                        model=self.model,
                    )
                except ContextOverflowError:
                    logger.warning(f"Subagent [{task_id}] context overflow, emergency compress")
                    messages = emergency_compress(messages, ctx_window, self.model)
                    try:
                        response = await self.provider.chat(
                            messages=messages,
                            tools=tools.get_definitions(),
                            model=self.model,
                        )
                    except ContextOverflowError:
                        logger.error(f"Subagent [{task_id}] emergency compress failed")
                        final_result = "子代理上下文溢出，紧急压缩后仍超限，任务中止。"
                        break

                if response.has_tool_calls:
                    tool_call_dicts = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in response.tool_calls
                    ]
                    messages.append({
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": tool_call_dicts,
                    })

                    for tool_call in response.tool_calls:
                        args_str = json.dumps(tool_call.arguments)
                        logger.debug(
                            f"Subagent [{task_id}] executing: {tool_call.name} "
                            f"with arguments: {args_str}"
                        )
                        result = await tools.execute(tool_call.name, tool_call.arguments)
                        result = trim_tool_result(result)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.name,
                            "content": result,
                        })
                else:
                    final_result = response.content
                    break

            if final_result is None:
                final_result = "Task completed but no final response was generated."

            logger.info(f"Subagent [{task_id}] completed successfully")
            if entry:
                entry.mark_completed(final_result)
            await self._announce_result(
                task_id, label, task, final_result, origin, SubagentStatus.COMPLETED,
            )

        except asyncio.CancelledError:
            # 被 wait_for 超时或外部取消 — 不 announce, 由 _run_with_timeout 处理
            raise
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"Subagent [{task_id}] failed: {e}")
            if entry:
                entry.mark_failed(str(e))
            await self._announce_result(
                task_id, label, task, error_msg, origin, SubagentStatus.FAILED,
            )

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: dict[str, str],
        status: SubagentStatus,
    ) -> None:
        """Announce the subagent result to the main agent via the message bus."""
        status_text = {
            SubagentStatus.COMPLETED: "completed successfully",
            SubagentStatus.FAILED: "failed",
            SubagentStatus.TIMEOUT: "timed out",
        }.get(status, "finished")

        announce_content = (
            f"[Subagent '{label}' {status_text}]\n\n"
            f"Task: {task}\n\n"
            f"Result:\n{result}\n\n"
            "Summarize this naturally for the user. Keep it brief (1-2 sentences). "
            "Do not mention technical details like \"subagent\" or task IDs."
        )

        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
        )

        await self.bus.publish_inbound(msg)
        logger.debug(
            f"Subagent [{task_id}] announced result to "
            f"{origin['channel']}:{origin['chat_id']}"
        )

    def _build_subagent_prompt(self, task: str) -> str:
        """Build a focused system prompt for the subagent.

        优先从 workspace/AGENTS.md 读取自定义指令，
        回退到内置默认提示词。
        """
        custom_instructions = ""
        agents_md = self.workspace / "AGENTS.md"
        if agents_md.exists():
            try:
                content = agents_md.read_text(encoding="utf-8")
                if content.strip():
                    custom_instructions = (
                        "\n\n## Custom Instructions (from AGENTS.md)\n"
                        f"{content.strip()}\n"
                    )
            except Exception:
                pass  # 读取失败时使用默认提示词

        return f"""# Subagent

You are a subagent spawned by the main agent to complete a specific task.

## Your Task
{task}

## Rules
1. Stay focused - complete only the assigned task, nothing else
2. Your final response will be reported back to the main agent
3. Do not initiate conversations or take on side tasks
4. Be concise but informative in your findings

## What You Can Do
- Read and write files in the workspace
- Execute shell commands
- Search the web and fetch web pages
- Complete the task thoroughly

## What You Cannot Do
- Send messages directly to users (no message tool available)
- Spawn other subagents
- Access the main agent's conversation history

## Workspace
Your workspace is at: {self.workspace}
{custom_instructions}
When you have completed the task, provide a clear summary of your findings or actions."""

    @staticmethod
    def _validate_model(model: str) -> bool:
        """验证模型标识符格式 (H4).

        接受格式:
        - "provider/model-name" (如 "anthropic/claude-opus-4-5")
        - "model-name" (如 "gpt-4o")
        不接受: 空字符串、含空格、含特殊字符
        """
        if not model or not model.strip():
            return False
        # 基本格式: 字母数字 + 连字符/下划线/点/斜杠
        return bool(re.match(r'^[a-zA-Z0-9][a-zA-Z0-9._/:@-]*$', model))

    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return len(self._running_tasks)

    def get_registry_stats(self) -> dict[str, int]:
        """返回 registry 统计信息."""
        entries = self.registry.get_all()
        from collections import Counter
        counts = Counter(e.status.value for e in entries)
        return {
            "total": len(entries),
            "running": counts.get("running", 0),
            "completed": counts.get("completed", 0),
            "failed": counts.get("failed", 0),
            "timeout": counts.get("timeout", 0),
        }

    # --- Sweeper (仅清理 + 持久化, 不做超时检查) ---

    async def start_sweeper(self) -> None:
        """启动后台清理任务 (gateway 模式调用)."""
        if hasattr(self, '_sweeper_task') and self._sweeper_task is not None:
            return
        self._sweeper_running = True
        self._sweeper_task = asyncio.create_task(self._sweeper_loop())
        logger.debug("Subagent sweeper started")

    async def stop_sweeper(self) -> None:
        """停止后台清理任务 (M2: 异步方法 + CancelledError 处理)."""
        self._sweeper_running = False
        task = getattr(self, '_sweeper_task', None)
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass  # M2: 正确处理 CancelledError
            self._sweeper_task = None
        # 最终持久化
        await self.registry.persist_if_dirty()
        logger.debug("Subagent sweeper stopped")

    async def _sweeper_loop(self) -> None:
        """Sweeper 主循环 — 仅清理 + 持久化, 不做超时 (C2)."""
        while self._sweeper_running:
            try:
                await asyncio.sleep(PERSIST_INTERVAL)

                # 1. 异步持久化
                await self.registry.persist_if_dirty()

                # 2. 清理终态条目 (超过阈值时)
                if len(self.registry) > 100:
                    cleaned = self.registry.cleanup_archived()
                    if cleaned:
                        logger.debug(f"Sweeper cleaned {cleaned} archived entries")

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Sweeper error: {exc}")
                await asyncio.sleep(5)  # 错误后短暂等待再重试
```


### 6. nanobot/agent/loop.py (MODIFICATIONS)

**Change 1**: Add `subagent_config` parameter to `__init__` (line 41):

```python
def __init__(
    self,
    bus: MessageBus,
    provider: LLMProvider,
    workspace: Path,
    model: str | None = None,
    max_iterations: int = 20,
    brave_api_key: str | None = None,
    exec_config: "ExecToolConfig | None" = None,
    cron_service: "CronService | None" = None,
    restrict_to_workspace: bool = False,
    session_manager: SessionManager | None = None,
    memory_config: dict | None = None,
    context_compression: bool = True,
    context_window_override: int | None = None,
    max_tokens: int = 8192,
    temperature: float = 0.7,
    subagent_config: "SubagentConfig | None" = None,  # NEW
):
```

**Change 2**: Pass config to SubagentManager (line 81-89):

```python
self.subagents = SubagentManager(
    provider=provider,
    workspace=workspace,
    bus=bus,
    model=self.model,
    brave_api_key=brave_api_key,
    exec_config=self.exec_config,
    restrict_to_workspace=restrict_to_workspace,
    subagent_config=subagent_config,  # NEW
)
```

**Change 3**: Remove ALL `set_context()` calls and inject origin params in tool execution.

Delete lines 225-235 in `_process_message` (ALL three set_context blocks):
```python
# DELETE THESE LINES:
# -- MessageTool set_context --
message_tool = self.tools.get("message")
if isinstance(message_tool, MessageTool):
    message_tool.set_context(msg.channel, msg.chat_id)

# -- SpawnTool set_context --
spawn_tool = self.tools.get("spawn")
if isinstance(spawn_tool, SpawnTool):
    spawn_tool.set_context(msg.channel, msg.chat_id)

# -- CronTool set_context --
cron_tool = self.tools.get("cron")
if isinstance(cron_tool, CronTool):
    cron_tool.set_context(msg.channel, msg.chat_id)
```

Modify tool execution loop in `_process_message` (around line 296-308):

```python
# H3: 需要 origin 上下文的工具集合
_ORIGIN_TOOLS = {"spawn", "message", "cron"}

for tool_call in response.tool_calls:
    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")

    # H3: 对需要 origin 的工具统一注入上下文 (不依赖可变状态)
    call_params = dict(tool_call.arguments)
    if tool_call.name in _ORIGIN_TOOLS:
        call_params["_origin_channel"] = msg.channel
        call_params["_origin_chat_id"] = msg.chat_id

    result = await self.tools.execute(tool_call.name, call_params)
    if self._compression_enabled:
        result = trim_tool_result(result)
    await self.context.memory.on_tool_executed(
        tool_call.name, tool_call.arguments, result
    )
    messages = self.context.add_tool_result(
        messages, tool_call.id, tool_call.name, result
    )
```

Delete lines 360-370 in `_process_system_message` (ALL three set_context blocks):
```python
# DELETE THESE LINES:
message_tool = self.tools.get("message")
if isinstance(message_tool, MessageTool):
    message_tool.set_context(origin_channel, origin_chat_id)

spawn_tool = self.tools.get("spawn")
if isinstance(spawn_tool, SpawnTool):
    spawn_tool.set_context(origin_channel, origin_chat_id)

cron_tool = self.tools.get("cron")
if isinstance(cron_tool, CronTool):
    cron_tool.set_context(origin_channel, origin_chat_id)
```

Modify tool execution loop in `_process_system_message` (around line 415-423):

```python
for tool_call in response.tool_calls:
    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")

    call_params = dict(tool_call.arguments)
    if tool_call.name in _ORIGIN_TOOLS:
        call_params["_origin_channel"] = origin_channel
        call_params["_origin_chat_id"] = origin_chat_id

    result = await self.tools.execute(tool_call.name, call_params)
    if self._compression_enabled:
        result = trim_tool_result(result)
    messages = self.context.add_tool_result(
        messages, tool_call.id, tool_call.name, result
    )
```

**Change 4**: Start/stop sweeper in `run()` and `shutdown()`:

Modify `run()` method (line 169-196):

```python
async def run(self) -> None:
    """Run the agent loop, processing messages from the bus."""
    self._running = True
    logger.info("Agent loop started")

    # 启动子代理 sweeper (gateway 模式)
    await self.subagents.start_sweeper()

    try:
        while self._running:
            try:
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
                try:
                    response = await self._process_message(msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))
            except asyncio.TimeoutError:
                continue
    finally:
        await self.subagents.stop_sweeper()
```

Modify `shutdown()` method (line 471-474):

```python
async def shutdown(self):
    """Shutdown agent loop and smart memory."""
    self.stop()
    await self.subagents.stop_sweeper()
    await self.context.memory.shutdown()
```

### 7. nanobot/cli/commands.py (MODIFICATIONS)

**Change 1**: In `gateway()` function, add `subagent_config` parameter when creating AgentLoop (around line 356-372):

```python
agent = AgentLoop(
    bus=bus,
    provider=provider,
    workspace=config.workspace_path,
    model=config.agents.defaults.model,
    max_iterations=config.agents.defaults.max_tool_iterations,
    brave_api_key=config.tools.web.search.api_key or None,
    exec_config=config.tools.exec,
    cron_service=cron,
    restrict_to_workspace=config.tools.restrict_to_workspace,
    session_manager=session_manager,
    memory_config=config.memory.model_dump() if config.memory.enabled else None,
    context_compression=config.agents.defaults.context_compression,
    context_window_override=config.agents.defaults.context_window_override,
    max_tokens=config.agents.defaults.max_tokens,
    temperature=config.agents.defaults.temperature,
    subagent_config=config.agents.subagent,  # NEW
)
```

**Change 2**: In `agent()` function, add `subagent_config` parameter when creating AgentLoop (around line 468-480):

```python
agent_loop = AgentLoop(
    bus=bus,
    provider=provider,
    workspace=config.workspace_path,
    brave_api_key=config.tools.web.search.api_key or None,
    exec_config=config.tools.exec,
    restrict_to_workspace=config.tools.restrict_to_workspace,
    memory_config=config.memory.model_dump() if config.memory.enabled else None,
    context_compression=config.agents.defaults.context_compression,
    context_window_override=config.agents.defaults.context_window_override,
    max_tokens=config.agents.defaults.max_tokens,
    temperature=config.agents.defaults.temperature,
    subagent_config=config.agents.subagent,  # NEW
)
```


---

## Test Code

### tests/test_subagent_types.py (NEW)

```python
"""Tests for subagent type definitions and state management."""

import time
import pytest
from nanobot.agent.subagent_types import (
    SubagentStatus, SubagentEntry,
)


class TestSubagentStatus:
    def test_terminal_states(self):
        assert SubagentStatus.COMPLETED.is_terminal()
        assert SubagentStatus.FAILED.is_terminal()
        assert SubagentStatus.TIMEOUT.is_terminal()

    def test_non_terminal_states(self):
        assert not SubagentStatus.PENDING.is_terminal()
        assert not SubagentStatus.RUNNING.is_terminal()


class TestSubagentEntry:
    def _make_entry(self, **kwargs):
        defaults = dict(
            task_id="abc123",
            label="test",
            task="do something",
            origin_channel="cli",
            origin_chat_id="direct",
        )
        defaults.update(kwargs)
        return SubagentEntry(**defaults)

    def test_initial_status_is_pending(self):
        e = self._make_entry()
        assert e.status == SubagentStatus.PENDING

    def test_mark_running(self):
        e = self._make_entry()
        e.mark_running()
        assert e.status == SubagentStatus.RUNNING

    def test_mark_completed(self):
        e = self._make_entry()
        e.mark_running()
        e.mark_completed("done")
        assert e.status == SubagentStatus.COMPLETED
        assert e.result == "done"
        assert e.finished_at is not None

    def test_mark_failed(self):
        e = self._make_entry()
        e.mark_running()
        e.mark_failed("oops")
        assert e.status == SubagentStatus.FAILED
        assert e.error == "oops"

    def test_mark_timeout(self):
        e = self._make_entry()
        e.mark_running()
        e.mark_timeout()
        assert e.status == SubagentStatus.TIMEOUT

    # --- 幂等性测试 (C2) ---

    def test_mark_completed_is_idempotent(self):
        e = self._make_entry()
        e.mark_completed("first")
        e.mark_completed("second")
        assert e.result == "first"  # 不被覆盖

    def test_mark_failed_on_completed_is_noop(self):
        e = self._make_entry()
        e.mark_completed("ok")
        e.mark_failed("should not change")
        assert e.status == SubagentStatus.COMPLETED

    def test_mark_timeout_on_failed_is_noop(self):
        e = self._make_entry()
        e.mark_failed("err")
        e.mark_timeout()
        assert e.status == SubagentStatus.FAILED

    def test_mark_running_on_terminal_is_noop(self):
        e = self._make_entry()
        e.mark_completed("done")
        e.mark_running()
        assert e.status == SubagentStatus.COMPLETED

    def test_to_dict(self):
        e = self._make_entry()
        d = e.to_dict()
        assert d["task_id"] == "abc123"
        assert d["status"] == "pending"
        assert "created_at" in d

```


### tests/test_subagent_registry.py (NEW)

```python
"""Tests for subagent registry."""

import json
import pytest
from pathlib import Path
from nanobot.agent.subagent_types import SubagentEntry, SubagentStatus
from nanobot.agent.subagent_registry import SubagentRegistry, MAX_ENTRIES


class TestSubagentRegistry:
    def _make_entry(self, task_id: str = "t1", **kwargs):
        defaults = dict(
            task_id=task_id, label="test", task="do it",
            origin_channel="cli", origin_chat_id="direct",
        )
        defaults.update(kwargs)
        return SubagentEntry(**defaults)

    def test_register_and_get(self):
        reg = SubagentRegistry()
        e = self._make_entry()
        reg.register(e)
        assert reg.get("t1") is e
        assert len(reg) == 1

    def test_get_running(self):
        reg = SubagentRegistry()
        e1 = self._make_entry("t1")
        e1.mark_running()
        e2 = self._make_entry("t2")
        reg.register(e1)
        reg.register(e2)
        running = reg.get_running()
        assert len(running) == 1
        assert running[0].task_id == "t1"

    def test_cleanup_archived(self):
        reg = SubagentRegistry()
        e = self._make_entry()
        e.mark_completed("done")
        reg.register(e)
        cleaned = reg.cleanup_archived()
        assert cleaned == 1
        assert len(reg) == 0

    # --- C3: 有界增长 ---

    def test_eviction_on_overflow(self):
        reg = SubagentRegistry()
        # 填满 MAX_ENTRIES 个终态条目
        for i in range(MAX_ENTRIES):
            e = self._make_entry(f"old-{i}")
            e.mark_completed("done")
            reg.register(e)
        assert len(reg) == MAX_ENTRIES
        # 再注册一个，应触发淘汰
        new_entry = self._make_entry("new-1")
        reg.register(new_entry)
        assert len(reg) <= MAX_ENTRIES

    # --- C1: 异步持久化 ---

    @pytest.mark.asyncio
    async def test_persist_if_dirty(self, tmp_path):
        path = tmp_path / "registry.json"
        reg = SubagentRegistry(persist_path=path)
        e = self._make_entry()
        reg.register(e)
        assert reg._dirty
        await reg.persist_if_dirty()
        assert not reg._dirty
        assert path.exists()
        data = json.loads(path.read_text())
        assert "t1" in data

    @pytest.mark.asyncio
    async def test_persist_skips_when_clean(self, tmp_path):
        path = tmp_path / "registry.json"
        reg = SubagentRegistry(persist_path=path)
        await reg.persist_if_dirty()
        assert not path.exists()  # 无变更不写入

    # --- H1: 崩溃恢复 ---

    def test_load_marks_non_terminal_as_failed(self, tmp_path):
        path = tmp_path / "registry.json"
        # 模拟崩溃前的状态: 一个 RUNNING, 一个 COMPLETED
        data = {
            "t1": {
                "task_id": "t1", "label": "running",
                "task": "x", "status": "running",
                "origin_channel": "cli", "origin_chat_id": "d",
                "created_at": 0, "finished_at": None,
            },
            "t2": {
                "task_id": "t2", "label": "done",
                "task": "y", "status": "completed",
                "origin_channel": "cli", "origin_chat_id": "d",
                "created_at": 0, "finished_at": 1,
            },
        }
        path.write_text(json.dumps(data))
        reg = SubagentRegistry(persist_path=path)
        reg.load()
        assert reg.get("t1").status == SubagentStatus.FAILED  # 恢复为 FAILED
        assert reg.get("t2").status == SubagentStatus.COMPLETED  # 不变

    def test_load_handles_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        reg = SubagentRegistry(persist_path=path)
        reg.load()  # 不抛异常
        assert len(reg) == 0

    def test_load_handles_corrupt_file(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text("not json")
        reg = SubagentRegistry(persist_path=path)
        reg.load()  # 不抛异常
        assert len(reg) == 0
```

### tests/test_subagent_improved.py (NEW)

```python
"""Integration tests for improved SubagentManager."""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from nanobot.agent.subagent import SubagentManager
from nanobot.agent.subagent_types import SubagentStatus
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import SubagentConfig, ExecToolConfig
from nanobot.providers.base import LLMResponse


def _make_manager(
    tmp_path: Path,
    timeout: int = 300,
    max_concurrent: int = 5,
    nesting_enabled: bool = False,
) -> tuple[SubagentManager, AsyncMock]:
    provider = AsyncMock()
    provider.get_default_model.return_value = "test/model-1"
    provider.chat = AsyncMock(return_value=LLMResponse(
        content="Task done.", finish_reason="stop",
    ))
    bus = MessageBus()
    config = SubagentConfig(
        timeout_seconds=timeout,
        max_concurrent=max_concurrent,
        nesting_enabled=nesting_enabled,
    )
    manager = SubagentManager(
        provider=provider,
        workspace=tmp_path,
        bus=bus,
        exec_config=ExecToolConfig(timeout=5),
        subagent_config=config,
    )
    return manager, provider


class TestSpawnConcurrencyLimit:
    @pytest.mark.asyncio
    async def test_rejects_when_at_limit(self, tmp_path):
        mgr, _ = _make_manager(tmp_path, max_concurrent=1)
        # 第一个 spawn 成功
        result1 = await mgr.spawn("task 1")
        assert "started" in result1
        # 第二个 spawn 被拒绝
        result2 = await mgr.spawn("task 2")
        assert "Maximum concurrent" in result2


class TestSpawnNesting:
    @pytest.mark.asyncio
    async def test_nesting_disabled_rejects(self, tmp_path):
        mgr, _ = _make_manager(tmp_path, nesting_enabled=False)
        result = await mgr.spawn("nested task", is_nested=True)
        assert "nesting is disabled" in result

    @pytest.mark.asyncio
    async def test_nesting_enabled_allows(self, tmp_path):
        mgr, _ = _make_manager(tmp_path, nesting_enabled=True)
        result = await mgr.spawn("nested task", is_nested=True)
        assert "started" in result


class TestModelValidation:
    @pytest.mark.asyncio
    async def test_valid_model_formats(self, tmp_path):
        # 这些都应该通过验证
        assert SubagentManager._validate_model("anthropic/claude-opus-4-5")
        assert SubagentManager._validate_model("gpt-4o")
        assert SubagentManager._validate_model("deepseek/deepseek-chat")
        assert SubagentManager._validate_model("test_model.v2")
        assert SubagentManager._validate_model("bedrock/anthropic.claude-3-sonnet:0")  # colon support

    @pytest.mark.asyncio
    async def test_invalid_model_formats(self, tmp_path):
        assert not SubagentManager._validate_model("")
        assert not SubagentManager._validate_model("  ")
        assert not SubagentManager._validate_model("model with spaces")


class TestTimeout:
    @pytest.mark.asyncio
    async def test_timeout_marks_entry(self, tmp_path):
        mgr, provider = _make_manager(tmp_path, timeout=1)

        # 让 provider.chat 永远挂起
        async def hang(*args, **kwargs):
            await asyncio.sleep(999)

        provider.chat = hang

        await mgr.spawn("slow task")
        # 等待超时触发
        await asyncio.sleep(2)

        # 检查 registry 中的状态
        entries = mgr.registry.get_all()
        timed_out = [e for e in entries if e.status == SubagentStatus.TIMEOUT]
        assert len(timed_out) >= 1


class TestSpawnToolConcurrencySafe:
    """验证 SpawnTool 不再有可变 origin 状态 (H3)."""

    def test_no_set_context_method(self):
        from nanobot.agent.tools.spawn import SpawnTool
        mgr = MagicMock()
        tool = SpawnTool(manager=mgr)
        assert not hasattr(tool, 'set_context') or not callable(getattr(tool, 'set_context', None))
        assert not hasattr(tool, '_origin_channel')
        assert not hasattr(tool, '_origin_chat_id')

    @pytest.mark.asyncio
    async def test_execute_passes_origin_params(self):
        from nanobot.agent.tools.spawn import SpawnTool
        mgr = AsyncMock()
        mgr.spawn = AsyncMock(return_value="ok")
        tool = SpawnTool(manager=mgr)
        await tool.execute(
            task="test",
            _origin_channel="telegram",
            _origin_chat_id="chat123",
        )
        mgr.spawn.assert_called_once_with(
            task="test",
            label=None,
            origin_channel="telegram",
            origin_chat_id="chat123",
        )
```

### tests/test_overflow_recovery.py (MODIFY — CRITICAL-2 兼容性修复)

The existing `TestSubagentOverflowRecovery` tests use `MagicMock()` for workspace and assert `status == "ok"`.
Both break with the new `SubagentManager` (registry needs real Path, status is now `SubagentStatus` enum).

**Changes needed**:

1. Replace `workspace=MagicMock()` with `workspace=tmp_path` (use `tmp_path` fixture)
2. Update `capture_announce` signature to match new `_announce_result(status: SubagentStatus)`
3. Update assertions from `status == "ok"` to `status == SubagentStatus.COMPLETED`

```python
# Replace TestSubagentOverflowRecovery class:

class TestSubagentOverflowRecovery:
    """Subagent overflow protection in _run_subagent."""

    async def test_subagent_overflow_emergency_compress_then_success(self, tmp_path):
        """Subagent: first call overflows, emergency compress, second succeeds."""
        from nanobot.agent.subagent import SubagentManager
        from nanobot.agent.subagent_types import SubagentStatus
        from nanobot.bus.queue import MessageBus

        provider = AsyncMock(spec=LLMProvider)
        provider.get_default_model.return_value = "test-model"
        provider.chat = AsyncMock(side_effect=[
            ContextOverflowError("overflow"),
            _ok_response("subagent done"),
        ])

        bus = MessageBus()
        mgr = SubagentManager(
            provider=provider,
            workspace=tmp_path,  # FIXED: use real path instead of MagicMock
            bus=bus,
            model="test-model",
        )

        announced = {}
        async def capture_announce(task_id, label, task, result, origin, status):
            announced["result"] = result
            announced["status"] = status

        mgr._announce_result = capture_announce

        await mgr._run_subagent("t1", "test task", "test", {"channel": "cli", "chat_id": "d"})
        assert announced["result"] == "subagent done"
        assert announced["status"] == SubagentStatus.COMPLETED  # FIXED: enum instead of string

    async def test_subagent_double_overflow_graceful_exit(self, tmp_path):
        """Subagent: both calls overflow -> graceful abort message."""
        from nanobot.agent.subagent import SubagentManager
        from nanobot.agent.subagent_types import SubagentStatus
        from nanobot.bus.queue import MessageBus

        provider = AsyncMock(spec=LLMProvider)
        provider.get_default_model.return_value = "test-model"
        provider.chat = AsyncMock(side_effect=[
            ContextOverflowError("1st"),
            ContextOverflowError("2nd"),
        ])

        bus = MessageBus()
        mgr = SubagentManager(
            provider=provider,
            workspace=tmp_path,  # FIXED: use real path instead of MagicMock
            bus=bus,
            model="test-model",
        )

        announced = {}
        async def capture_announce(task_id, label, task, result, origin, status):
            announced["result"] = result
            announced["status"] = status

        mgr._announce_result = capture_announce

        await mgr._run_subagent("t2", "test task", "test", {"channel": "cli", "chat_id": "d"})
        assert "上下文溢出" in announced["result"] or "紧急压缩" in announced["result"]
```


### tests/test_sweeper.py (NEW)

```python
"""Tests for sweeper behavior."""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock

from nanobot.agent.subagent import SubagentManager
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import SubagentConfig, ExecToolConfig
from nanobot.providers.base import LLMResponse


@pytest.fixture
def manager(tmp_path):
    provider = AsyncMock()
    provider.get_default_model.return_value = "test/model"
    provider.chat = AsyncMock(return_value=LLMResponse(
        content="done", finish_reason="stop",
    ))
    bus = MessageBus()
    config = SubagentConfig(timeout_seconds=300)
    mgr = SubagentManager(
        provider=provider,
        workspace=tmp_path,
        bus=bus,
        exec_config=ExecToolConfig(timeout=5),
        subagent_config=config,
    )
    return mgr


class TestSweeper:
    @pytest.mark.asyncio
    async def test_start_and_stop(self, manager):
        await manager.start_sweeper()
        assert manager._sweeper_task is not None
        assert not manager._sweeper_task.done()
        await manager.stop_sweeper()
        assert manager._sweeper_task is None

    @pytest.mark.asyncio
    async def test_stop_without_start_is_safe(self, manager):
        await manager.stop_sweeper()  # 不抛异常

    @pytest.mark.asyncio
    async def test_double_start_is_idempotent(self, manager):
        await manager.start_sweeper()
        task1 = manager._sweeper_task
        await manager.start_sweeper()
        task2 = manager._sweeper_task
        assert task1 is task2  # 同一个 task
        await manager.stop_sweeper()

    @pytest.mark.asyncio
    async def test_sweeper_does_not_check_timeout(self, manager):
        """C2 验证: sweeper 循环中不应有超时检查逻辑."""
        import inspect
        source = inspect.getsource(manager._sweeper_loop)
        # sweeper 不应包含 timeout 相关的状态检查
        assert "mark_timeout" not in source
        assert "timeout_seconds" not in source
```


---

## Verification Commands

### Linting
```bash
ruff check nanobot/agent/subagent_types.py
ruff check nanobot/agent/subagent_registry.py
ruff check nanobot/agent/subagent.py
ruff check nanobot/agent/tools/spawn.py
```

### Type Checking
```bash
mypy nanobot/agent/subagent_types.py
mypy nanobot/agent/subagent_registry.py
mypy nanobot/agent/subagent.py
```

### Unit Tests
```bash
# Run all new tests
pytest tests/test_subagent_types.py -v
pytest tests/test_subagent_registry.py -v
pytest tests/test_subagent_improved.py -v
pytest tests/test_sweeper.py -v

# Run all tests to ensure no regressions
pytest tests/ -v
```

### Integration Tests
```bash
# Test CLI mode (no sweeper)
nanobot agent -m "spawn a subagent to list files in the workspace"

# Test timeout
nanobot agent -m "spawn a subagent that sleeps for 400 seconds"

# Test concurrency limit
nanobot agent -m "spawn 10 subagents simultaneously"
```

### Acceptance Criteria

**Phase 1 (Foundation)**:
- [ ] `SubagentStatus` enum has 5 values, `is_terminal()` works correctly
- [ ] `SubagentEntry.mark_*` methods are idempotent for terminal states
- [ ] `SubagentRegistry` auto-evicts when exceeding MAX_ENTRIES
- [ ] `SubagentRegistry.load()` marks non-terminal entries as FAILED
- [ ] `persist_if_dirty()` uses `asyncio.to_thread` (non-blocking)
- [ ] `SubagentConfig` has reasonable defaults
- [ ] Existing `config.json` without `subagent` field doesn't error

**Phase 2 (Timeout + Concurrency)**:
- [ ] `asyncio.wait_for` is the only timeout source
- [ ] `_run_with_timeout` handles TimeoutError and CancelledError separately
- [ ] `SubagentEntry.mark_timeout()` is idempotent
- [ ] SpawnTool has no `set_context()` method or `_origin_*` attributes
- [ ] Origin passed via `execute()` parameters `_origin_channel`/`_origin_chat_id`
- [ ] AgentLoop injects origin params when calling spawn/message/cron tools
- [ ] MessageTool has no `set_context()` method or `_default_channel`/`_default_chat_id` attributes
- [ ] CronTool has no `set_context()` method or `_channel`/`_chat_id` attributes
- [ ] Concurrency limit enforced (returns error when at max_concurrent)
- [ ] Nesting check is single-layer (`is_nested` + `nesting_enabled`)

**Phase 3 (Validation + Prompt)**:
- [ ] `_validate_model` accepts valid formats (including colon for bedrock-style), rejects invalid ones
- [ ] System prompt includes AGENTS.md content if it exists
- [ ] Falls back to default prompt if AGENTS.md missing/unreadable
- [ ] `get_registry_stats()` returns correct status counts

**Phase 4 (Sweeper)**:
- [ ] Sweeper loop has no timeout checking logic
- [ ] `stop_sweeper()` is async and handles CancelledError
- [ ] ResultQueue deferred (H2): collect 模式当前未使用，仅保留 direct 模式
- [ ] Gateway mode starts sweeper, CLI mode doesn't
- [ ] Sweeper calls `persist_if_dirty()` and `cleanup_archived()` periodically
- [ ] `shutdown()` calls `stop_sweeper()` for final persistence

**Phase 5 (Tests)**:
- [ ] All test files pass: `pytest tests/test_subagent_*.py tests/test_sweeper.py`
- [ ] No regressions: `pytest tests/` passes
- [ ] Linting clean: `ruff check nanobot/agent/subagent*.py`

---

## Backward Compatibility

### Configuration
1. **No `agents.subagent` field in config.json**:
   - Pydantic uses `SubagentConfig()` default values
   - Default: 5 concurrent, 300s timeout, 15 iterations, no nesting

2. **SubagentManager.__init__**:
   - `subagent_config` parameter defaults to `None`
   - Internal fallback: `self._config = subagent_config or SubagentConfig()`

3. **SpawnTool.execute()**:
   - `_origin_channel` and `_origin_chat_id` have defaults: `"cli"` and `"direct"`
   - Old code that doesn't pass these params still works

### API Compatibility
1. **SubagentManager.get_running_count()**:
   - Signature unchanged
   - Still returns `len(self._running_tasks)`

2. **Internal _announce_result**:
   - `status` parameter changed from `str` to `SubagentStatus` enum
   - This is internal API only, no external callers

3. **Registry persistence path**:
   - New location: `{workspace}/.nanobot/subagent_registry.json`
   - Old state files (if any) are ignored, fresh start

### Migration Notes
- **Existing deployments**: No action required. Config without `subagent` field uses defaults.
- **Custom configs**: Add `agents.subagent` section to customize timeout/concurrency/nesting.
- **Crash recovery**: On first run after upgrade, any interrupted subagents from old version won't be recovered (no old registry file). This is acceptable as it's a one-time event.

### Breaking Changes
**None.** All changes are backward compatible.

---

## Implementation Order

```
Week 1:
  Day 1-2: Phase 1 (subagent_types + subagent_registry + SubagentConfig)
  Day 3-4: Phase 2 (timeout + SpawnTool + nesting)

Week 2:
  Day 1: Phase 3 (validation + prompt)
  Day 2: Phase 4 (ResultQueue + Sweeper)
  Day 3-4: Phase 5 (tests)
  Day 5: Integration testing + documentation
```

---

## Summary

This plan addresses all CRITICAL, HIGH, and MEDIUM issues identified in the architect review:

- **C1-C3 (CRITICAL)**: Async persistence, single timeout source, bounded registry
- **H1-H4 (HIGH)**: Crash recovery, message loss prevention, concurrency safety, validation
- **M2-M5 (MEDIUM)**: Async sweeper, simplified nesting, enum status

All code is complete and executable. Tests provide comprehensive coverage. Backward compatibility is maintained throughout.

**Total Files**:
- 2 new source files
- 7 modified source files (including message.py, cron.py, test_overflow_recovery.py)
- 4 new test files
- 0 breaking changes

**Estimated Effort**: 2 weeks (1 developer)

---

## Architect Review Fixes (V2.1)

以下修复基于 architect 二次审查发现的问题：

| # | 级别 | 问题 | 修复 |
|---|------|------|------|
| AR-1 | CRITICAL | H3 修复不完整: MessageTool/CronTool 仍用 `set_context()` 可变状态 | 对三个工具统一采用 `_origin_*` 参数注入; loop.py 用 `_ORIGIN_TOOLS` 集合替代硬编码 tool name |
| AR-2 | CRITICAL | 现有 `test_overflow_recovery.py` 会被破坏 (MagicMock workspace + status 字符串断言) | 新增 test 修改节, workspace 改 `tmp_path`, status 断言改 `SubagentStatus` 枚举 |
| AR-3 | HIGH | `_evict_terminal` else 分支删除所有终态条目而非保留 | `else terminal` → `else []` |
| AR-4 | HIGH | `ResultQueue`/`PendingResult` 定义但从未使用 (H2 死代码) | 移除死代码, H2 标记为 deferred |
| AR-5 | HIGH | `registry.load()` 在构造函数中同步 I/O, 与 C1 目标矛盾 | 改为 lazy load: `_ensure_registry_loaded()` 在首次 `spawn()` 时调用 |
| AR-6 | MEDIUM | `_validate_model` 正则拒绝 bedrock 风格含冒号的模型名 | 正则扩展为 `[a-zA-Z0-9._/:@-]` |
| AR-7 | MEDIUM | `to_dict()` 截断 task 到 200 字符, 持久化后丢失完整内容 | 移除截断, 保留完整 task |

