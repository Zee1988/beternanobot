"""Subagent manager for background task execution."""

import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.compressor import (
    compress_messages,
    emergency_compress,
    get_context_window,
    trim_tool_result,
)
from nanobot.agent.subagent_registry import ARCHIVE_KEEP, PERSIST_INTERVAL, SubagentRegistry
from nanobot.agent.subagent_types import SubagentEntry, SubagentStatus
from nanobot.agent.tools.filesystem import ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import ContextOverflowError, LLMProvider


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
        exec_config: "ExecToolConfig | None" = None,  # noqa: F821
        restrict_to_workspace: bool = False,
        subagent_config: "SubagentConfig | None" = None,  # noqa: F821
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
                if len(self.registry) > ARCHIVE_KEEP * 2:
                    cleaned = self.registry.cleanup_archived()
                    if cleaned:
                        logger.debug(f"Sweeper cleaned {cleaned} archived entries")

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Sweeper error: {exc}")
                await asyncio.sleep(5)  # 错误后短暂等待再重试
