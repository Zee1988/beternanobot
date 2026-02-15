"""Agent loop: the core processing engine."""

import asyncio
import json
import re
from pathlib import Path

from loguru import logger

from nanobot.agent.compressor import (
    compress_messages,
    emergency_compress,
    get_context_window,
    trim_tool_result,
)
from nanobot.agent.context import ContextBuilder
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import ContextOverflowError, LLMProvider, LLMResponse
from nanobot.session.manager import SessionManager
from nanobot.session.tool_result_guard import ToolResultGuard
from nanobot.storage.retrieval import MEMORY_CONTEXT_PREFIX

# H3: 需要 origin 上下文的工具集合
_ORIGIN_TOOLS = {"spawn", "message", "cron"}

# 检测 LLM 输出代码块但未调用工具的模式 (tool-call 退化)
_CODE_BLOCK_RE = re.compile(
    r"```(?:bash|sh|shell|python|cmd|powershell|json|javascript|js)\b",
    re.IGNORECASE,
)
_MAX_NUDGES = 3

_TOOL_NUDGE = (
    "You have tools available. Do NOT output shell commands or code in markdown "
    "code blocks. Instead, call the appropriate tool (e.g. exec for shell commands, "
    "read_file for reading files). Execute the action now using a tool call."
)

_TOOL_NUDGE_HARD = (
    "STOP outputting code blocks. You MUST use function calling / tool_use to "
    "execute actions. For shell commands, call the 'exec' tool with a 'command' "
    "parameter. Do NOT wrap it in ```json or ```bash. Make an actual tool call NOW."
)


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        exec_config: "ExecToolConfig | None" = None,  # noqa: F821
        cron_service: "CronService | None" = None,  # noqa: F821
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        memory_config: dict | None = None,
        context_compression: bool = True,
        context_window_override: int | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        subagent_config: "SubagentConfig | None" = None,  # noqa: F821
        llm_call_timeout: int = 120,
    ):
        from nanobot.config.schema import ExecToolConfig
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self.memory_config = memory_config
        self._compression_enabled = context_compression
        self._context_window = get_context_window(
            self.model, context_window_override
        )
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._llm_timeout = llm_call_timeout

        # 并发消息处理: 每 session 一把锁，防止同用户消息竞争
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._active_tasks: set[asyncio.Task] = set()

        self.context = ContextBuilder(workspace, memory_config=memory_config)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
            subagent_config=subagent_config,
        )

        self._running = False
        self._register_default_tools()

    async def initialize_memory(self):
        """Initialize smart memory system."""
        await self.context.memory.initialize()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools (restrict to workspace if configured)
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))

        # Shell tool
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))

        # Web tools: 只保留 fetch, 搜索用 tavily skill
        self.tools.register(WebFetchTool())

        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)

        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)

        # Cron tool (for scheduling)
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

    async def _call_llm_with_timeout(
        self,
        messages: list[dict],
        tools_defs: list[dict] | None,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Single LLM call wrapped with asyncio.wait_for timeout."""
        return await asyncio.wait_for(
            self.provider.chat(
                messages=messages,
                tools=tools_defs,
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
            ),
            timeout=self._llm_timeout,
        )

    async def _call_llm_with_recovery(
        self,
        messages: list[dict],
        tools_defs: list[dict] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Shared LLM call with timeout + compression + overflow recovery."""
        _max_tokens = max_tokens if max_tokens is not None else self._max_tokens
        _temperature = temperature if temperature is not None else self._temperature

        if self._compression_enabled:
            messages = compress_messages(messages, self._context_window, self.model)

        try:
            return await self._call_llm_with_timeout(
                messages, tools_defs, _max_tokens, _temperature,
            )
        except asyncio.TimeoutError:
            logger.warning(f"LLM call timed out after {self._llm_timeout}s, retrying...")
            try:
                return await self._call_llm_with_timeout(
                    messages, tools_defs, _max_tokens, _temperature,
                )
            except asyncio.TimeoutError:
                logger.error("LLM call timed out twice, giving up")
                return LLMResponse(content="请求超时，请稍后重试。", finish_reason="error")
        except ContextOverflowError:
            logger.warning("Context overflow, applying emergency compression")
            messages = emergency_compress(messages, self._context_window, self.model)
            try:
                return await self._call_llm_with_timeout(
                    messages, tools_defs, _max_tokens, _temperature,
                )
            except ContextOverflowError:
                logger.error("Emergency compression failed, context still too large")
                msg = "对话历史过长，自动压缩后仍然超出模型限制。" \
                    "请发送 /clear 清空会话后重试。"
                return LLMResponse(content=msg, finish_reason="error")

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus concurrently."""
        self._running = True
        logger.info("Agent loop started (concurrent mode)")

        # 启动子代理 sweeper (gateway 模式)
        await self.subagents.start_sweeper()

        try:
            while self._running:
                try:
                    msg = await asyncio.wait_for(
                        self.bus.consume_inbound(),
                        timeout=1.0
                    )
                    # 并发派发: 每条消息一个 task，不阻塞主循环
                    task = asyncio.create_task(self._handle_message(msg))
                    self._active_tasks.add(task)
                    task.add_done_callback(self._active_tasks.discard)
                except asyncio.TimeoutError:
                    continue
        finally:
            await self.subagents.stop_sweeper()
            # 等待所有活跃任务完成 (带超时)
            if self._active_tasks:
                logger.info(f"Waiting for {len(self._active_tasks)} active tasks to finish...")
                done, pending = await asyncio.wait(
                    self._active_tasks, timeout=30.0
                )
                for t in pending:
                    t.cancel()
                if pending:
                    logger.warning(f"Force-cancelled {len(pending)} tasks on shutdown")

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    def _get_session_lock(self, session_key: str) -> asyncio.Lock:
        """获取 session 对应的锁 (懒创建)."""
        if session_key not in self._session_locks:
            self._session_locks[session_key] = asyncio.Lock()
        return self._session_locks[session_key]

    async def _heartbeat(self, channel: str, chat_id: str, delay: float = 30.0):
        """处理超时后自动发送进度提示."""
        await asyncio.sleep(delay)
        await self.bus.publish_outbound(OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content="仍在处理中，请稍候...",
        ))

    async def _handle_message(self, msg: InboundMessage) -> None:
        """并发安全的消息处理入口: session 加锁 + 心跳 + 异常兜底."""
        # 确定 session key (system 消息用 origin session)
        if msg.channel == "system" and ":" in msg.chat_id:
            session_key = msg.chat_id  # "origin_channel:origin_chat_id"
        else:
            session_key = msg.session_key

        lock = self._get_session_lock(session_key)

        async with lock:
            # 非 system 消息启动心跳
            heartbeat_task = None
            if msg.channel != "system":
                heartbeat_task = asyncio.create_task(
                    self._heartbeat(msg.channel, msg.chat_id)
                )

            try:
                response = await self._process_message(msg)
                if response:
                    await self.bus.publish_outbound(response)
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                # 确定回复目标
                if msg.channel == "system" and ":" in msg.chat_id:
                    parts = msg.chat_id.split(":", 1)
                    ch, cid = parts[0], parts[1]
                else:
                    ch, cid = msg.channel, msg.chat_id
                await self.bus.publish_outbound(OutboundMessage(
                    channel=ch,
                    chat_id=cid,
                    content=f"Sorry, I encountered an error: {str(e)}"
                ))
            finally:
                if heartbeat_task and not heartbeat_task.done():
                    heartbeat_task.cancel()

    async def _process_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a single inbound message.

        Args:
            msg: The inbound message to process.

        Returns:
            The response message, or None if no response needed.
        """
        # Handle system messages (subagent announces)
        # The chat_id contains the original "channel:chat_id" to route back to
        if msg.channel == "system":
            return await self._process_system_message(msg)

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}: {preview}")

        # Get or create session
        session = self.sessions.get_or_create(msg.session_key)

        # Build initial messages (use get_history for LLM-formatted messages)
        if self._compression_enabled:
            history_budget = self._context_window - self._max_tokens - 4096
            history_budget = max(history_budget, 4096)
            history = session.get_history(max_tokens=history_budget)
        else:
            history = session.get_history()

        messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )

        # Smart memory: async recall with timeout
        memory_context = await self.context.memory.recall(msg.content, timeout=0.15)
        if memory_context:
            # Safety check: ensure prefix exists (#25)
            if not memory_context.startswith(MEMORY_CONTEXT_PREFIX):
                memory_context = f"{MEMORY_CONTEXT_PREFIX}\n{memory_context}\n[记忆参考结束]"
            messages.insert(-1, {"role": "user", "content": memory_context})

        # Feed message to SummaryTimer
        self.context.memory.feed_message("user", msg.content)

        # Agent loop
        iteration = 0
        nudge_count = 0
        final_content = None

        while iteration < self.max_iterations:
            iteration += 1

            # Call LLM
            response = await self._call_llm_with_recovery(
                messages, self.tools.get_definitions()
            )

            # Handle tool calls
            if response.has_tool_calls:
                # Add assistant message with tool calls
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)  # Must be JSON string
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )

                # Execute tools
                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                    # H3: 对需要 origin 的工具统一注入上下文
                    call_params = dict(tool_call.arguments)
                    if tool_call.name in _ORIGIN_TOOLS:
                        call_params["_origin_channel"] = msg.channel
                        call_params["_origin_chat_id"] = msg.chat_id
                    result = await self.tools.execute(tool_call.name, call_params)
                    if self._compression_enabled:
                        result = trim_tool_result(result)
                    # Smart memory: record tool observation
                    await self.context.memory.on_tool_executed(
                        tool_call.name, tool_call.arguments, result
                    )
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                # No tool calls — check for code-block-without-tool-call pattern
                content = response.content or ""
                if (
                    nudge_count < _MAX_NUDGES
                    and _CODE_BLOCK_RE.search(content)
                ):
                    nudge_count += 1
                    nudge = _TOOL_NUDGE if nudge_count == 1 else _TOOL_NUDGE_HARD
                    logger.warning(
                        f"Detected code-block output without tool call, "
                        f"nudging model (attempt {nudge_count}/{_MAX_NUDGES})"
                    )
                    messages = self.context.add_assistant_message(
                        messages, content,
                        reasoning_content=response.reasoning_content,
                    )
                    messages.append({"role": "user", "content": nudge})
                    continue

                final_content = content
                # Feed assistant response to SummaryTimer
                if final_content:
                    self.context.memory.feed_message("assistant", final_content)
                break

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        # Log response preview
        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info(f"Response to {msg.channel}:{msg.sender_id}: {preview}")

        # Save to session — preserve full tool call structure for positive reinforcement
        session.add_message("user", msg.content)
        self._save_tool_history(session, messages)
        session.add_message("assistant", final_content)
        self.sessions.save(session)

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},  # Pass through for channel-specific needs (e.g. Slack thread_ts)
        )

    @staticmethod
    def _save_tool_history(session, messages: list[dict]) -> None:
        """Extract tool call exchanges from LLM messages and save to session.

        Uses ToolResultGuard to ensure every tool call has a matching result
        before persisting. This gives the model positive reinforcement by
        seeing structured tool calls in future history.
        """
        # Collect new tool exchanges (walk backwards from end to last user msg)
        new_tool_msgs = []
        for msg in reversed(messages):
            role = msg.get("role")
            if role == "tool":
                new_tool_msgs.append(msg)
            elif role == "assistant" and msg.get("tool_calls"):
                new_tool_msgs.append(msg)
            elif role == "user":
                break
            elif role == "assistant":
                continue

        new_tool_msgs.reverse()

        # Pass through the guard to ensure pairing integrity
        guard = ToolResultGuard()
        for msg in new_tool_msgs:
            guarded = guard.process(msg)
            for m in guarded:
                role = m["role"]
                content = m.get("content", "") or ""
                if role == "assistant" and m.get("tool_calls"):
                    session.add_message(
                        "assistant", content,
                        tool_calls=m["tool_calls"],
                    )
                elif role == "tool":
                    session.add_message(
                        "tool", content,
                        tool_call_id=m.get("tool_call_id", ""),
                        name=m.get("name", ""),
                    )

        # Flush any remaining unmatched tool calls
        for m in guard.flush_pending():
            session.add_message(
                "tool", m.get("content", ""),
                tool_call_id=m.get("tool_call_id", ""),
                name=m.get("name", ""),
            )

    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).

        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")

        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id

        # Use the origin session for context
        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)

        # Build messages with the announce content
        if self._compression_enabled:
            history_budget = self._context_window - self._max_tokens - 4096
            history_budget = max(history_budget, 4096)
            history = session.get_history(max_tokens=history_budget)
        else:
            history = session.get_history()

        messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            channel=origin_channel,
            chat_id=origin_chat_id,
        )

        # Agent loop (limited for announce handling)
        iteration = 0
        nudge_count = 0
        final_content = None

        while iteration < self.max_iterations:
            iteration += 1

            response = await self._call_llm_with_recovery(
                messages, self.tools.get_definitions()
            )

            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )

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
            else:
                content = response.content or ""
                if (
                    nudge_count < _MAX_NUDGES
                    and _CODE_BLOCK_RE.search(content)
                ):
                    nudge_count += 1
                    nudge = _TOOL_NUDGE if nudge_count == 1 else _TOOL_NUDGE_HARD
                    logger.warning(
                        f"Detected code-block output without tool call in system handler, "
                        f"nudging (attempt {nudge_count}/{_MAX_NUDGES})"
                    )
                    messages = self.context.add_assistant_message(
                        messages, content,
                        reasoning_content=response.reasoning_content,
                    )
                    messages.append({"role": "user", "content": nudge})
                    continue

                final_content = content
                break

        if final_content is None:
            final_content = "Background task completed."

        # Save to session — preserve full tool call structure for positive reinforcement
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        self._save_tool_history(session, messages)
        session.add_message("assistant", final_content)
        self.sessions.save(session)

        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content
        )

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
    ) -> str:
        """
        Process a message directly (for CLI or cron usage).

        Args:
            content: The message content.
            session_key: Session identifier.
            channel: Source channel (for context).
            chat_id: Source chat ID (for context).

        Returns:
            The agent's response.
        """
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content
        )

        response = await self._process_message(msg)
        return response.content if response else ""

    async def shutdown(self):
        """Shutdown agent loop and smart memory."""
        self.stop()
        await self.subagents.stop_sweeper()
        await self.context.memory.shutdown()
