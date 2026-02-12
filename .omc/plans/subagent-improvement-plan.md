# Subagent 系统优化方案

> 基于对 nanobot 现有代码的深度分析，设计 6 项渐进式改进。
> Phase 1 (1-3) 为基础设施级高优先级，Phase 2 (4-6) 为增量增强。

---

## 现状分析

### 当前架构

```
AgentLoop (loop.py)
  ├── SubagentManager (subagent.py)     # 管理子代理生命周期
  │     ├── spawn()                      # 创建 asyncio.Task
  │     ├── _run_subagent()              # 独立 LLM 循环 (max 15 轮)
  │     ├── _announce_result()           # 通过 MessageBus 注入 InboundMessage
  │     └── _running_tasks: dict[str, asyncio.Task]  # 唯一的状态追踪
  │
  └── SpawnTool (tools/spawn.py)        # LLM 调用入口
        ├── set_context()                # 设置 origin channel/chat_id
        └── execute() → manager.spawn()
```

### 关键缺陷

| 缺陷 | 位置 | 影响 |
|------|------|------|
| 无注册表 | `subagent.py:50` `_running_tasks` 仅存 asyncio.Task 引用 | 无法查询历史、无法持久化、无法审计 |
| 无嵌套防护 | `subagent.py:102-114` 子代理工具集无 spawn 但 prompt 仅口头禁止 | 当前靠不注册 SpawnTool 实现，但无显式标记 |
| 无超时 | `subagent.py:128` while 循环无时间限制 | 子代理可能无限运行 |
| 无清理 | `subagent.py:86` done_callback 仅 pop | 已完成任务无记录、无归档 |
| 结果立即注入 | `subagent.py:221-228` 直接 publish_inbound | 主代理忙时会打断当前处理 |
| 固定模型 | `subagent.py:46` 使用父代理同一模型 | 无法为简单任务使用更便宜的模型 |
| 系统提示简陋 | `subagent.py:231-260` 硬编码字符串 | 无法定制行为约束 |

---

## 架构总览

```
                          ┌─────────────────────────────────┐
                          │         Config (schema.py)       │
                          │  SubagentConfig                  │
                          │  ├── max_concurrent: 5           │
                          │  ├── timeout_seconds: 300        │
                          │  ├── archive_after_seconds: 3600 │
                          │  ├── model: str | None           │
                          │  └── system_prompt_path: str|None│
                          └──────────┬──────────────────────┘
                                     │
┌────────────────────────────────────┼────────────────────────────────┐
│                          AgentLoop │                                │
│                                    ▼                                │
│  ┌──────────────┐    ┌─────────────────────────┐                   │
│  │  SpawnTool    │───▶│   SubagentManager       │                   │
│  │  (spawn.py)   │    │   (subagent.py)          │                   │
│  └──────────────┘    │                           │                   │
│                       │  ┌─────────────────────┐ │                   │
│                       │  │  SubagentRegistry    │ │  ← Phase 1.1    │
│  ┌──────────────┐    │  │  (registry.py)       │ │                   │
│  │ ResultQueue   │◀──│  │  ├── entries: dict    │ │                   │
│  │ (Phase 2.4)   │    │  │  ├── query()         │ │                   │
│  └──────┬───────┘    │  │  ├── update_status()  │ │                   │
│         │            │  │  └── persist()/load()  │ │                   │
│         ▼            │  └─────────────────────┘ │                   │
│  ┌──────────────┐    │                           │                   │
│  │ _process_     │    │  ┌─────────────────────┐ │                   │
│  │  system_msg() │    │  │  Sweeper             │ │  ← Phase 1.3    │
│  └──────────────┘    │  │  (sweeper task)      │ │                   │
│                       │  │  ├── cancel timeout  │ │                   │
│                       │  │  └── archive done    │ │                   │
│                       │  └─────────────────────┘ │                   │
│                       └─────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: 基础设施 (高优先级)

---

### 1. SubagentRegistry — 子代理注册表

#### 目标

为每个子代理建立完整的生命周期记录，支持查询、停止、审计。

#### 数据结构

新建文件: `nanobot/agent/subagent_registry.py`

```python
"""Subagent registry for lifecycle tracking."""

import time
import json
import asyncio
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger


class SubagentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class SubagentEntry:
    """Single subagent lifecycle record."""
    run_id: str                          # 8-char UUID
    task: str                            # 原始任务描述
    label: str                           # 显示标签
    status: SubagentStatus = SubagentStatus.PENDING
    requester_session_key: str = ""      # 发起者的 session key (channel:chat_id)
    child_session_key: str = ""          # 子代理自身的 session key
    model: str = ""                      # 使用的模型
    is_subagent: bool = True             # 标记为子代理 (用于嵌套防护)

    # 时间戳 (epoch seconds)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None

    # 结果
    outcome: str | None = None           # 最终结果摘要
    error: str | None = None             # 错误信息
    iterations: int = 0                  # 实际执行轮数
    token_usage: dict[str, int] = field(default_factory=dict)

    def elapsed_seconds(self) -> float | None:
        """运行耗时。"""
        if self.started_at is None:
            return None
        end = self.completed_at or time.time()
        return end - self.started_at

    def is_terminal(self) -> bool:
        """是否已结束。"""
        return self.status in (
            SubagentStatus.COMPLETED,
            SubagentStatus.FAILED,
            SubagentStatus.TIMEOUT,
            SubagentStatus.CANCELLED,
        )


class SubagentRegistry:
    """
    In-memory registry with optional disk persistence.

    线程安全: 所有操作在同一 event loop 中执行，无需额外锁。
    """

    def __init__(self, persist_path: Path | None = None):
        self._entries: dict[str, SubagentEntry] = {}
        self._tasks: dict[str, asyncio.Task] = {}  # run_id → asyncio.Task
        self._persist_path = persist_path

        # 启动时加载持久化数据
        if persist_path and persist_path.exists():
            self._load()

    # ── 注册 / 更新 ──

    def register(self, entry: SubagentEntry) -> None:
        """注册新的子代理条目。"""
        self._entries[entry.run_id] = entry
        self._maybe_persist()

    def bind_task(self, run_id: str, task: asyncio.Task) -> None:
        """绑定 asyncio.Task 到注册条目。"""
        self._tasks[run_id] = task
        entry = self._entries.get(run_id)
        if entry:
            entry.status = SubagentStatus.RUNNING
            entry.started_at = time.time()
            self._maybe_persist()

    def mark_completed(
        self, run_id: str, outcome: str | None = None,
        iterations: int = 0, token_usage: dict | None = None,
    ) -> None:
        entry = self._entries.get(run_id)
        if not entry:
            return
        entry.status = SubagentStatus.COMPLETED
        entry.completed_at = time.time()
        entry.outcome = outcome
        entry.iterations = iterations
        if token_usage:
            entry.token_usage = token_usage
        self._tasks.pop(run_id, None)
        self._maybe_persist()

    def mark_failed(self, run_id: str, error: str) -> None:
        entry = self._entries.get(run_id)
        if not entry:
            return
        entry.status = SubagentStatus.FAILED
        entry.completed_at = time.time()
        entry.error = error
        self._tasks.pop(run_id, None)
        self._maybe_persist()

    def mark_timeout(self, run_id: str) -> None:
        entry = self._entries.get(run_id)
        if not entry:
            return
        entry.status = SubagentStatus.TIMEOUT
        entry.completed_at = time.time()
        entry.error = "Exceeded maximum runtime"
        self._tasks.pop(run_id, None)
        self._maybe_persist()

    # ── 查询 ──

    def get(self, run_id: str) -> SubagentEntry | None:
        return self._entries.get(run_id)

    def query(
        self,
        status: SubagentStatus | None = None,
        session_key: str | None = None,
    ) -> list[SubagentEntry]:
        """按条件查询子代理。"""
        results = list(self._entries.values())
        if status:
            results = [e for e in results if e.status == status]
        if session_key:
            results = [e for e in results if e.requester_session_key == session_key]
        return sorted(results, key=lambda e: e.created_at, reverse=True)

    def active_count(self) -> int:
        """当前活跃 (pending + running) 的子代理数量。"""
        return sum(
            1 for e in self._entries.values()
            if e.status in (SubagentStatus.PENDING, SubagentStatus.RUNNING)
        )

    # ── 控制 ──

    async def cancel(self, run_id: str) -> bool:
        """取消一个运行中的子代理。"""
        task = self._tasks.get(run_id)
        if task and not task.done():
            task.cancel()
            entry = self._entries.get(run_id)
            if entry:
                entry.status = SubagentStatus.CANCELLED
                entry.completed_at = time.time()
            self._tasks.pop(run_id, None)
            self._maybe_persist()
            return True
        return False

    # ── 清理 ──

    def cleanup_archived(self, max_age_seconds: float = 3600) -> int:
        """清理已归档的终态条目。返回清理数量。"""
        now = time.time()
        to_remove = []
        for run_id, entry in self._entries.items():
            if entry.is_terminal() and entry.completed_at:
                if now - entry.completed_at > max_age_seconds:
                    to_remove.append(run_id)
        for run_id in to_remove:
            del self._entries[run_id]
            self._tasks.pop(run_id, None)
        if to_remove:
            self._maybe_persist()
        return len(to_remove)

    # ── 持久化 ──

    def _maybe_persist(self) -> None:
        if not self._persist_path:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                run_id: asdict(entry)
                for run_id, entry in self._entries.items()
            }
            self._persist_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2)
            )
        except Exception as e:
            logger.warning(f"Failed to persist subagent registry: {e}")

    def _load(self) -> None:
        try:
            data = json.loads(self._persist_path.read_text())
            for run_id, raw in data.items():
                raw["status"] = SubagentStatus(raw["status"])
                self._entries[run_id] = SubagentEntry(**raw)
        except Exception as e:
            logger.warning(f"Failed to load subagent registry: {e}")
```

#### 对现有文件的修改

**`nanobot/agent/subagent.py`** — 核心改造:

```python
# 修改前 (subagent.py:50)
self._running_tasks: dict[str, asyncio.Task[None]] = {}

# 修改后
from nanobot.agent.subagent_registry import SubagentRegistry, SubagentEntry, SubagentStatus

# __init__ 新增参数
def __init__(self, ..., registry: SubagentRegistry | None = None):
    ...
    self.registry = registry or SubagentRegistry()

# spawn() 方法改造 (subagent.py:52-89)
async def spawn(self, task, label, origin_channel, origin_chat_id) -> str:
    task_id = str(uuid.uuid4())[:8]
    display_label = label or task[:30] + ("..." if len(task) > 30 else "")

    # 注册
    entry = SubagentEntry(
        run_id=task_id,
        task=task,
        label=display_label,
        requester_session_key=f"{origin_channel}:{origin_chat_id}",
        model=self.model,
    )
    self.registry.register(entry)

    origin = {"channel": origin_channel, "chat_id": origin_chat_id}
    bg_task = asyncio.create_task(
        self._run_subagent(task_id, task, display_label, origin)
    )
    self.registry.bind_task(task_id, bg_task)

    logger.info(f"Spawned subagent [{task_id}]: {display_label}")
    return f"Subagent [{display_label}] started (id: {task_id})."

# _run_subagent() 完成时更新注册表 (subagent.py:191-197)
# 成功:
self.registry.mark_completed(task_id, outcome=final_result, iterations=iteration)
# 失败:
self.registry.mark_failed(task_id, error=str(e))
```

**`nanobot/agent/loop.py`** — 注入 registry:

```python
# loop.py:81-89 修改
from nanobot.agent.subagent_registry import SubagentRegistry

# __init__ 中:
self._subagent_registry = SubagentRegistry(
    persist_path=Path.home() / ".nanobot" / "data" / "subagents.json"
)
self.subagents = SubagentManager(
    ...,
    registry=self._subagent_registry,
)
```

#### 边界情况

- 进程重启时，持久化文件中 `RUNNING` 状态的条目应标记为 `FAILED`（进程崩溃）
- `SubagentRegistry._load()` 中增加: 将非终态条目标记为 `FAILED`
- 并发上限检查: `spawn()` 前检查 `registry.active_count() < max_concurrent`

---

### 2. Nesting Prevention — 嵌套防护

#### 目标

从机制层面阻止子代理递归 spawn，而非仅靠 prompt 约束。

#### 当前状态

- `subagent.py:102-114`: 子代理的 ToolRegistry 不注册 SpawnTool — 这是隐式防护
- `subagent.py:231-260`: system prompt 中写了 "Cannot spawn other subagents" — 仅口头约束
- 无显式 `is_subagent` 标记传递

#### 实现方案

**方案: `is_subagent` 标记 + 工具集显式过滤**

修改 `nanobot/agent/subagent.py`:

```python
# _run_subagent() 中构建工具集时 (subagent.py:102-114)
# 当前已经不注册 SpawnTool，保持不变
# 新增: 在 SubagentEntry 中标记 is_subagent=True (已在 registry 数据结构中)

# spawn() 方法增加嵌套检查:
async def spawn(self, task, label, origin_channel, origin_chat_id,
                is_subagent_context: bool = False) -> str:
    """Spawn a subagent. Rejects if called from subagent context."""
    if is_subagent_context:
        return "Error: Subagents cannot spawn other subagents."

    # ... 正常 spawn 逻辑
```

修改 `nanobot/agent/tools/spawn.py`:

```python
class SpawnTool(Tool):
    def __init__(self, manager: "SubagentManager", is_subagent: bool = False):
        self._manager = manager
        self._is_subagent = is_subagent  # 新增标记
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"

    async def execute(self, task: str, label: str | None = None, **kwargs) -> str:
        if self._is_subagent:
            return "Error: Subagents cannot spawn other subagents."
        return await self._manager.spawn(
            task=task, label=label,
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
        )
```

#### 双重防护策略

| 层级 | 机制 | 位置 |
|------|------|------|
| L1: 工具集 | 子代理不注册 SpawnTool | `subagent.py:102-114` (已有) |
| L2: 标记 | `is_subagent` 标记阻止 spawn | `spawn.py` execute 检查 |
| L3: 注册表 | `SubagentEntry.is_subagent` 审计 | `subagent_registry.py` |

#### 边界情况

- 即使未来有人给子代理注册了 SpawnTool，L2 标记检查仍会阻止
- 日志记录嵌套尝试，便于调试: `logger.warning(f"Nesting attempt blocked")`

---

### 3. Timeout & Sweeper — 超时与清理

#### 目标

防止子代理无限运行，自动清理已完成的历史记录。

#### 当前状态

- `subagent.py:128`: `while iteration < max_iterations` 仅限制 LLM 调用轮数 (15)，不限制总时间
- `subagent.py:86`: done_callback 仅从 `_running_tasks` 中移除，无归档
- 单次 LLM 调用可能因网络问题 hang 住

#### 实现方案

**3a. 子代理超时**

修改 `nanobot/agent/subagent.py` 的 `_run_subagent()`:

```python
async def _run_subagent(self, task_id, task, label, origin, timeout: float = 300):
    """Execute subagent with timeout protection."""
    try:
        await asyncio.wait_for(
            self._run_subagent_inner(task_id, task, label, origin),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(f"Subagent [{task_id}] timed out after {timeout}s")
        self.registry.mark_timeout(task_id)
        await self._announce_result(
            task_id, label, task,
            f"任务超时 (超过 {timeout} 秒)，已自动终止。",
            origin, "timeout",
        )
    except asyncio.CancelledError:
        logger.info(f"Subagent [{task_id}] cancelled")
        # registry 已在 cancel() 中更新
    except Exception as e:
        logger.error(f"Subagent [{task_id}] failed: {e}")
        self.registry.mark_failed(task_id, error=str(e))
        await self._announce_result(task_id, label, task, f"Error: {e}", origin, "error")

# 原有逻辑移入 _run_subagent_inner()
async def _run_subagent_inner(self, task_id, task, label, origin):
    """Core subagent execution logic (extracted from _run_subagent)."""
    # ... 原有 _run_subagent 的 try 块内容 (subagent.py:101-197)
```

**3b. Sweeper 后台任务**

在 `SubagentManager` 中新增:

```python
async def start_sweeper(self, interval: float = 30.0, archive_after: float = 3600.0):
    """启动后台清理任务。"""
    self._sweeper_running = True
    while self._sweeper_running:
        await asyncio.sleep(interval)
        try:
            # 1. 检查超时 (备用机制，asyncio.wait_for 是主要超时)
            for entry in self.registry.query(status=SubagentStatus.RUNNING):
                elapsed = entry.elapsed_seconds()
                if elapsed and elapsed > self._timeout:
                    logger.warning(f"Sweeper: subagent [{entry.run_id}] exceeded timeout")
                    await self.registry.cancel(entry.run_id)
                    self.registry.mark_timeout(entry.run_id)

            # 2. 清理已归档条目
            cleaned = self.registry.cleanup_archived(max_age_seconds=archive_after)
            if cleaned:
                logger.debug(f"Sweeper: cleaned {cleaned} archived subagent entries")
        except Exception as e:
            logger.error(f"Sweeper error: {e}")

def stop_sweeper(self):
    self._sweeper_running = False
```

**3c. 在 AgentLoop / Gateway 中启动 sweeper**

修改 `nanobot/agent/loop.py`:

```python
# AgentLoop.run() 中 (loop.py:169-196)
async def run(self) -> None:
    self._running = True
    # 启动 sweeper
    sweeper_task = asyncio.create_task(
        self.subagents.start_sweeper(
            interval=30.0,
            archive_after=3600.0,
        )
    )
    try:
        # ... 原有消息处理循环
    finally:
        self.subagents.stop_sweeper()
        sweeper_task.cancel()
```

#### 配置项

修改 `nanobot/config/schema.py`，新增 `SubagentConfig`:

```python
class SubagentConfig(BaseModel):
    """Subagent behavior configuration."""
    max_concurrent: int = 5              # 最大并发子代理数
    timeout_seconds: int = 300           # 单个子代理最大运行时间 (秒)
    max_iterations: int = 15             # 单个子代理最大 LLM 调用轮数
    archive_after_seconds: int = 3600    # 已完成条目保留时间 (秒)
    sweeper_interval_seconds: int = 30   # sweeper 检查间隔 (秒)
    model: str | None = None             # 子代理默认模型 (None = 跟随父代理)
    system_prompt_path: str | None = None  # 自定义系统提示文件路径
    persist_registry: bool = False       # 是否持久化注册表到磁盘
```

在 `AgentsConfig` 中引用:

```python
class AgentsConfig(BaseModel):
    """Agent configuration."""
    defaults: AgentDefaults = Field(default_factory=AgentDefaults)
    subagent: SubagentConfig = Field(default_factory=SubagentConfig)  # 新增
```

#### 边界情况

- `asyncio.wait_for` 超时后，内部 task 被 cancel，需确保 LLM provider 的 HTTP 连接正确关闭
- sweeper 是备用机制: 主超时由 `asyncio.wait_for` 保证，sweeper 处理漏网之鱼
- 进程重启时 sweeper 不会运行，依赖 registry `_load()` 中的崩溃恢复逻辑

---

## Phase 2: 增量增强 (迭代优先级)

---

### 4. Result Queue — 结果队列

#### 目标

子代理完成时，如果主代理正忙，将结果排队而非立即注入 MessageBus，避免打断当前对话。

#### 当前问题

- `subagent.py:221-228`: `_announce_result()` 直接调用 `bus.publish_inbound()`
- `loop.py:174-196`: 主循环 `consume_inbound()` 会立即拿到子代理结果
- 如果主代理正在处理用户消息，子代理结果会排在队列后面等待，但处理时会创建新的 LLM 调用
- 多个子代理同时完成时，每个结果都触发独立的 LLM 调用，浪费 token

#### 实现方案

新建文件: `nanobot/agent/result_queue.py`

```python
"""Result queue for batching subagent announcements."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class PendingResult:
    """A queued subagent result waiting to be announced."""
    run_id: str
    label: str
    task: str
    result: str
    origin: dict[str, str]
    status: str  # "ok" | "error" | "timeout"
    queued_at: float = field(default_factory=time.time)


class ResultQueue:
    """
    Buffers subagent results and delivers them in batches.

    Modes:
    - immediate: 直接注入 (当前行为，向后兼容)
    - collect:   收集一段时间内的结果，批量注入
    """

    def __init__(self, mode: str = "immediate", collect_window: float = 5.0):
        self._mode = mode
        self._collect_window = collect_window
        self._pending: list[PendingResult] = []
        self._lock = asyncio.Lock()

    async def enqueue(self, result: PendingResult) -> None:
        """Add a result to the queue."""
        async with self._lock:
            self._pending.append(result)

    async def drain(self) -> list[PendingResult]:
        """Drain all pending results. Returns empty list if none."""
        async with self._lock:
            results = self._pending[:]
            self._pending.clear()
            return results

    def has_pending(self) -> bool:
        return len(self._pending) > 0

    async def drain_for_session(self, session_key: str) -> list[PendingResult]:
        """Drain results for a specific session."""
        async with self._lock:
            matching = [
                r for r in self._pending
                if f"{r.origin['channel']}:{r.origin['chat_id']}" == session_key
            ]
            self._pending = [
                r for r in self._pending
                if f"{r.origin['channel']}:{r.origin['chat_id']}" != session_key
            ]
            return matching
```

#### 对现有文件的修改

**`nanobot/agent/subagent.py`** — `_announce_result()` 改造:

```python
# 修改前 (subagent.py:199-229): 直接 publish_inbound
# 修改后: 通过 ResultQueue 缓冲

async def _announce_result(self, task_id, label, task, result, origin, status):
    pending = PendingResult(
        run_id=task_id, label=label, task=task,
        result=result, origin=origin, status=status,
    )
    if self.result_queue:
        await self.result_queue.enqueue(pending)
        logger.debug(f"Subagent [{task_id}] result queued")
    else:
        # fallback: 直接注入 (向后兼容)
        await self._inject_to_bus(pending)

async def _inject_to_bus(self, pending: PendingResult) -> None:
    """将结果注入 MessageBus (原有逻辑)。"""
    # ... 原有 _announce_result 的 InboundMessage 构建逻辑
```

**`nanobot/agent/loop.py`** — 在消息处理前检查队列:

```python
# _process_message() 开头 (loop.py:203-212)
# 在构建 messages 之前，检查是否有待处理的子代理结果
pending_results = await self.subagents.result_queue.drain_for_session(msg.session_key)
if pending_results:
    # 将结果合并为一条上下文消息
    batch_content = self._format_batch_results(pending_results)
    # 追加到 messages 中作为额外上下文
```

#### 边界情况

- `immediate` 模式保持完全向后兼容，不改变任何现有行为
- `collect` 模式下，如果用户长时间不发消息，结果会一直排队 — 需要一个最大等待时间后自动注入
- 批量结果的 token 消耗需要控制，超过阈值时分批发送

---

### 5. Dedicated System Prompt — 专用系统提示

#### 目标

为子代理注入更精细的行为约束，支持自定义提示模板。

#### 当前状态

- `subagent.py:231-260`: `_build_subagent_prompt()` 硬编码字符串
- 提示内容合理但不可配置，无法针对不同任务类型定制

#### 实现方案

**5a. 提示模板系统**

修改 `nanobot/agent/subagent.py` 的 `_build_subagent_prompt()`:

```python
def _build_subagent_prompt(self, task: str, label: str | None = None) -> str:
    """Build subagent system prompt from template or custom file."""
    # 1. 尝试加载自定义提示文件
    if self._system_prompt_path:
        custom_path = Path(self._system_prompt_path).expanduser()
        if custom_path.exists():
            template = custom_path.read_text()
            return template.format(
                task=task, label=label or "",
                workspace=self.workspace,
            )

    # 2. 使用增强版默认提示
    return f"""# Subagent

You are a focused subagent spawned to complete a specific task.

## Task
{task}

## Behavioral Constraints
1. SCOPE: Complete ONLY the assigned task. Do not explore tangential topics.
2. EPHEMERAL: You have no memory of previous interactions. Each run is independent.
3. NO SIDE EFFECTS: Do not modify files outside the task scope unless explicitly required.
4. CONCISE OUTPUT: Your final response will be relayed to the user.
   - Lead with the conclusion/answer
   - Include supporting evidence
   - Omit process narration ("First I did X, then Y...")
5. ERROR HANDLING: If you cannot complete the task, explain why clearly.
6. NO SPAWNING: You cannot create other subagents.
7. NO MESSAGING: You cannot send messages directly to users.

## Output Format
Structure your response as:
- **Result**: The main finding or action taken (1-2 sentences)
- **Details**: Supporting information (if needed)
- **Issues**: Any problems encountered (if any)

## Workspace
{self.workspace}
"""
```

**5b. 配置集成**

`SubagentConfig.system_prompt_path` (已在 Phase 1.3 配置中定义) 传递到 `SubagentManager`:

```python
# SubagentManager.__init__ 新增:
self._system_prompt_path = system_prompt_path  # 从 config 传入
```

#### 边界情况

- 自定义模板中的 `{task}` 等占位符需要文档说明
- 模板文件不存在时 fallback 到默认提示，不报错
- 模板过长会消耗子代理的上下文窗口 — 建议限制在 2000 字符以内

---

### 6. Flexible Model Config — 灵活模型配置

#### 目标

允许子代理使用不同于父代理的模型，支持全局默认和按任务覆盖。

#### 当前状态

- `subagent.py:46`: `self.model = model or provider.get_default_model()` — 使用父代理模型
- `spawn.py:42-56`: SpawnTool 参数中无 model 字段
- 所有子代理共享同一模型，无法为简单任务使用更便宜的模型

#### 实现方案

**6a. SpawnTool 增加 model 参数**

修改 `nanobot/agent/tools/spawn.py`:

```python
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
                "description": "Optional short label for the task",
            },
            "model": {
                "type": "string",
                "description": (
                    "Optional model override for this subagent. "
                    "Use a cheaper model for simple tasks (e.g. 'deepseek/deepseek-chat'). "
                    "Omit to use the default subagent model."
                ),
            },
        },
        "required": ["task"],
    }

async def execute(self, task: str, label: str | None = None,
                  model: str | None = None, **kwargs) -> str:
    return await self._manager.spawn(
        task=task, label=label, model=model,
        origin_channel=self._origin_channel,
        origin_chat_id=self._origin_chat_id,
    )
```

**6b. SubagentManager 模型解析优先级**

修改 `nanobot/agent/subagent.py`:

```python
async def spawn(self, task, label, origin_channel, origin_chat_id,
                model: str | None = None) -> str:
    # 模型优先级: per-task override > subagent config default > parent model
    resolved_model = model or self._subagent_default_model or self.model

    entry = SubagentEntry(
        ...,
        model=resolved_model,
    )
    ...

# _run_subagent_inner 中使用 entry.model:
async def _run_subagent_inner(self, task_id, task, label, origin):
    entry = self.registry.get(task_id)
    model = entry.model if entry else self.model
    # 使用 model 替代 self.model 进行 LLM 调用
```

**6c. 模型解析优先级图**

```
per-task model (SpawnTool 参数)
    │
    ▼ (如果为 None)
SubagentConfig.model (config.json)
    │
    ▼ (如果为 None)
AgentDefaults.model (父代理模型)
```

#### 边界情况

- 需要验证 model 字符串对应的 provider 有可用的 API key
- 不同模型的上下文窗口不同，`get_context_window()` 需要使用子代理实际模型
- LLM 主动选择 model 参数时可能产生幻觉模型名 — 需要在 `spawn()` 中校验

---

## 文件变更总览

### 新建文件

| 文件 | 用途 | Phase |
|------|------|-------|
| `nanobot/agent/subagent_registry.py` | SubagentRegistry + SubagentEntry + SubagentStatus | 1.1 |
| `nanobot/agent/result_queue.py` | ResultQueue + PendingResult | 2.4 |
| `tests/test_subagent_registry.py` | 注册表单元测试 | 1.1 |
| `tests/test_subagent_nesting.py` | 嵌套防护测试 | 1.2 |
| `tests/test_subagent_timeout.py` | 超时与 sweeper 测试 | 1.3 |
| `tests/test_result_queue.py` | 结果队列测试 | 2.4 |

### 修改文件

| 文件 | 变更内容 | Phase |
|------|---------|-------|
| `nanobot/agent/subagent.py` | 集成 registry, timeout, result_queue, model 解析 | 1.1-2.6 |
| `nanobot/agent/tools/spawn.py` | 增加 is_subagent 检查, model 参数 | 1.2, 2.6 |
| `nanobot/agent/loop.py` | 注入 registry, 启动 sweeper, 检查 result_queue | 1.1, 1.3, 2.4 |
| `nanobot/config/schema.py` | 新增 SubagentConfig, AgentsConfig 引用 | 1.3 |
| `nanobot/cli/commands.py` | 传递 subagent config 到 AgentLoop | 1.3 |

---

## 测试策略

### Phase 1 测试

```python
# tests/test_subagent_registry.py

import pytest
import asyncio
from nanobot.agent.subagent_registry import (
    SubagentRegistry, SubagentEntry, SubagentStatus,
)

class TestSubagentRegistry:
    def test_register_and_query(self):
        reg = SubagentRegistry()
        entry = SubagentEntry(run_id="abc", task="test", label="test")
        reg.register(entry)
        assert reg.get("abc") is not None
        assert reg.active_count() == 1

    def test_lifecycle_transitions(self):
        reg = SubagentRegistry()
        entry = SubagentEntry(run_id="abc", task="test", label="test")
        reg.register(entry)
        assert entry.status == SubagentStatus.PENDING

        # 模拟 bind_task
        loop = asyncio.new_event_loop()
        task = loop.create_task(asyncio.sleep(100))
        reg.bind_task("abc", task)
        assert entry.status == SubagentStatus.RUNNING

        reg.mark_completed("abc", outcome="done")
        assert entry.status == SubagentStatus.COMPLETED
        assert entry.is_terminal()
        loop.close()

    def test_cleanup_archived(self):
        reg = SubagentRegistry()
        entry = SubagentEntry(run_id="abc", task="test", label="test")
        reg.register(entry)
        reg.mark_completed("abc", outcome="done")
        entry.completed_at = 0  # 很久以前
        cleaned = reg.cleanup_archived(max_age_seconds=1)
        assert cleaned == 1
        assert reg.get("abc") is None

    def test_persist_and_load(self, tmp_path):
        path = tmp_path / "registry.json"
        reg = SubagentRegistry(persist_path=path)
        entry = SubagentEntry(run_id="abc", task="test", label="test")
        reg.register(entry)
        assert path.exists()

        reg2 = SubagentRegistry(persist_path=path)
        assert reg2.get("abc") is not None


# tests/test_subagent_nesting.py

class TestNestingPrevention:
    @pytest.mark.asyncio
    async def test_spawn_tool_blocks_in_subagent_context(self):
        from nanobot.agent.tools.spawn import SpawnTool
        tool = SpawnTool(manager=None, is_subagent=True)
        result = await tool.execute(task="nested task")
        assert "cannot spawn" in result.lower()

    @pytest.mark.asyncio
    async def test_spawn_tool_allows_in_main_context(self):
        # 需要 mock SubagentManager
        ...


# tests/test_subagent_timeout.py

class TestSubagentTimeout:
    @pytest.mark.asyncio
    async def test_timeout_cancels_subagent(self):
        """子代理超时后应被标记为 TIMEOUT。"""
        # 创建一个会 hang 的 mock provider
        # 验证 registry 中状态为 TIMEOUT
        ...

    @pytest.mark.asyncio
    async def test_sweeper_cleans_archived(self):
        """Sweeper 应清理过期的已完成条目。"""
        ...
```

### Phase 2 测试

```python
# tests/test_result_queue.py

class TestResultQueue:
    @pytest.mark.asyncio
    async def test_enqueue_and_drain(self):
        from nanobot.agent.result_queue import ResultQueue, PendingResult
        queue = ResultQueue()
        result = PendingResult(
            run_id="abc", label="test", task="test",
            result="done", origin={"channel": "cli", "chat_id": "direct"},
            status="ok",
        )
        await queue.enqueue(result)
        assert queue.has_pending()
        drained = await queue.drain()
        assert len(drained) == 1
        assert not queue.has_pending()

    @pytest.mark.asyncio
    async def test_drain_for_session(self):
        from nanobot.agent.result_queue import ResultQueue, PendingResult
        queue = ResultQueue()
        r1 = PendingResult(
            run_id="a", label="t1", task="t1", result="r1",
            origin={"channel": "telegram", "chat_id": "123"}, status="ok",
        )
        r2 = PendingResult(
            run_id="b", label="t2", task="t2", result="r2",
            origin={"channel": "cli", "chat_id": "direct"}, status="ok",
        )
        await queue.enqueue(r1)
        await queue.enqueue(r2)
        matched = await queue.drain_for_session("telegram:123")
        assert len(matched) == 1
        assert matched[0].run_id == "a"
        assert queue.has_pending()  # r2 still there
```

---

## 迁移与向后兼容

### 零破坏性原则

| 改动 | 兼容策略 |
|------|---------|
| SubagentRegistry | 默认 `persist_registry=False`，纯内存模式，不影响现有行为 |
| Nesting Prevention | 现有代码已不注册 SpawnTool，新增检查是额外保险 |
| Timeout | 默认 300s，远大于当前 15 轮迭代的典型耗时 |
| Sweeper | 仅在 `run()` 中启动，CLI 单次模式不受影响 |
| ResultQueue | 默认 `immediate` 模式，行为与当前完全一致 |
| System Prompt | 增强版默认提示兼容现有行为，自定义路径可选 |
| Model Config | `SubagentConfig.model = None` 时跟随父代理，零变化 |

### 配置迁移

现有 `~/.nanobot/config.json` 无需修改。新增的 `agents.subagent` 字段全部有默认值:

```json
{
  "agents": {
    "defaults": { "model": "anthropic/claude-opus-4-5", "..." : "..." },
    "subagent": {}
  }
}
```

Pydantic 的 `Field(default_factory=SubagentConfig)` 确保缺失字段自动填充默认值。

### 实施顺序建议

```
Week 1: Phase 1.1 (Registry) + Phase 1.2 (Nesting)
         → 这两个改动互相独立，可并行开发
         → Registry 是后续所有功能的基础

Week 2: Phase 1.3 (Timeout & Sweeper)
         → 依赖 Registry
         → 包含 config schema 变更

Week 3: Phase 2.5 (System Prompt) + Phase 2.6 (Model Config)
         → 独立改动，可并行
         → Model Config 依赖 config schema (已在 Week 2 完成)

Week 4: Phase 2.4 (Result Queue)
         → 最复杂的增强，涉及 loop.py 消息处理流程
         → 建议最后实施，充分测试
```

---

## 参考文件索引

| 文件 | 行号 | 说明 |
|------|------|------|
| `nanobot/agent/subagent.py:50` | `_running_tasks` 定义 | 当前唯一的状态追踪 |
| `nanobot/agent/subagent.py:52-89` | `spawn()` 方法 | 子代理创建入口 |
| `nanobot/agent/subagent.py:91-197` | `_run_subagent()` | 核心执行循环 |
| `nanobot/agent/subagent.py:199-229` | `_announce_result()` | 结果注入 MessageBus |
| `nanobot/agent/subagent.py:231-260` | `_build_subagent_prompt()` | 系统提示构建 |
| `nanobot/agent/tools/spawn.py:11-65` | `SpawnTool` 完整实现 | LLM 调用入口 |
| `nanobot/agent/loop.py:81-89` | SubagentManager 初始化 | AgentLoop 中的集成点 |
| `nanobot/agent/loop.py:169-196` | `run()` 主循环 | sweeper 启动位置 |
| `nanobot/agent/loop.py:203-334` | `_process_message()` | result_queue 检查位置 |
| `nanobot/agent/loop.py:336-440` | `_process_system_message()` | 子代理结果处理 |
| `nanobot/config/schema.py:157-170` | `AgentsConfig` | 配置扩展点 |
| `nanobot/bus/queue.py:25-27` | `publish_inbound()` | 子代理结果注入点 |
| `nanobot/bus/events.py:9-23` | `InboundMessage` | 子代理结果消息格式 |
