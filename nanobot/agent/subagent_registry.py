"""Subagent registry with bounded entries and async persistence."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

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
        """清理最旧终态条目，保留 ARCHIVE_KEEP 个，返回清理数量."""
        terminal = sorted(
            [(tid, e) for tid, e in self._entries.items() if e.status.is_terminal()],
            key=lambda x: x[1].created_at,
        )
        to_remove = terminal[:-ARCHIVE_KEEP] if len(terminal) > ARCHIVE_KEEP else []
        for tid, _ in to_remove:
            del self._entries[tid]
        if to_remove:
            self._dirty = True
        return len(to_remove)

    # --- 持久化 (异步, 解决 C1) ---

    async def persist_if_dirty(self) -> None:
        """仅在有变更时异步写入磁盘."""
        if not self._dirty or not self._persist_path:
            return
        # 先清除 dirty 标记，避免序列化与写入之间的新变更丢失标记 (H2)
        self._dirty = False
        data = {
            tid: e.to_dict() for tid, e in self._entries.items()
        }
        content = json.dumps(data, ensure_ascii=False, indent=2)
        try:
            await asyncio.to_thread(self._sync_write, content)
        except Exception as exc:
            self._dirty = True  # 写入失败，恢复 dirty 标记
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
                result=data.get("result"),
                error=data.get("error"),
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
