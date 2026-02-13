"""Tests for the anti-hang improvements.

Component 1: Concurrent message processing with per-session locking
Component 2: Progress heartbeat for long-running requests
Component 3: LLM call timeout safety net
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse


# ── Helpers ──────────────────────────────────────────────────


def _make_bus() -> MessageBus:
    return MessageBus()


def _make_provider(chat_fn=None):
    provider = MagicMock()
    provider.get_default_model.return_value = "test/model"
    if chat_fn:
        provider.chat = AsyncMock(side_effect=chat_fn)
    else:
        provider.chat = AsyncMock(
            return_value=LLMResponse(content="ok", finish_reason="stop")
        )
    return provider


def _make_loop(bus, provider, tmp_path, **kwargs):
    defaults = dict(
        bus=bus, provider=provider, workspace=tmp_path,
        model="test/model", llm_call_timeout=2,
    )
    defaults.update(kwargs)
    return AgentLoop(**defaults)


# ═══════════════════════════════════════════════════════════════
# Component 1: Concurrent message processing
# ═══════════════════════════════════════════════════════════════


class TestConcurrentProcessing:
    """Messages from different users process concurrently."""

    @pytest.mark.asyncio
    async def test_msg2_processed_while_msg1_hangs(self, tmp_path):
        """Core fix: user B gets a response even when user A's request is slow."""
        hang_event = asyncio.Event()
        outbound: list[OutboundMessage] = []

        async def slow_for_u1(**kwargs):
            msgs = kwargs.get("messages", [])
            user_msgs = [m for m in msgs if m.get("role") == "user"]
            content = user_msgs[-1]["content"] if user_msgs else ""
            if "msg1" in content:
                await hang_event.wait()
                return LLMResponse(content="msg1 done", finish_reason="stop")
            return LLMResponse(content="msg2 done", finish_reason="stop")

        bus = _make_bus()
        provider = _make_provider(slow_for_u1)
        loop = _make_loop(bus, provider, tmp_path)

        # Capture outbound messages
        original_publish = bus.publish_outbound

        async def capture_outbound(msg):
            outbound.append(msg)
            await original_publish(msg)

        bus.publish_outbound = capture_outbound

        loop_task = asyncio.create_task(loop.run())

        # Send msg1 (will hang)
        await bus.publish_inbound(InboundMessage(
            channel="test", sender_id="u1", chat_id="c1", content="msg1",
        ))
        await asyncio.sleep(0.3)

        # Send msg2 from different user
        await bus.publish_inbound(InboundMessage(
            channel="test", sender_id="u2", chat_id="c2", content="msg2",
        ))
        await asyncio.sleep(1.0)

        # msg2 should be done while msg1 is still hanging
        msg2_responses = [m for m in outbound if m.chat_id == "c2" and "msg2 done" in m.content]
        assert len(msg2_responses) >= 1, "msg2 should be processed while msg1 hangs"

        msg1_responses = [m for m in outbound if m.chat_id == "c1" and "msg1 done" in m.content]
        assert len(msg1_responses) == 0, "msg1 should still be hanging"

        # Release msg1
        hang_event.set()
        await asyncio.sleep(1.0)

        msg1_responses = [m for m in outbound if m.chat_id == "c1" and "msg1 done" in m.content]
        assert len(msg1_responses) >= 1, "msg1 should complete after release"

        loop.stop()
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_same_user_messages_serialized(self, tmp_path):
        """Messages from the same user are serialized via session lock."""
        order: list[str] = []

        async def track_order(**kwargs):
            msgs = kwargs.get("messages", [])
            user_msgs = [m for m in msgs if m.get("role") == "user"]
            content = user_msgs[-1]["content"] if user_msgs else ""
            order.append(f"start:{content}")
            if "first" in content:
                await asyncio.sleep(0.5)
            order.append(f"end:{content}")
            return LLMResponse(content=f"re:{content}", finish_reason="stop")

        bus = _make_bus()
        provider = _make_provider(track_order)
        loop = _make_loop(bus, provider, tmp_path)

        loop_task = asyncio.create_task(loop.run())

        # Send two messages from the SAME user
        await bus.publish_inbound(InboundMessage(
            channel="test", sender_id="u1", chat_id="c1", content="first",
        ))
        await asyncio.sleep(0.05)
        await bus.publish_inbound(InboundMessage(
            channel="test", sender_id="u1", chat_id="c1", content="second",
        ))

        await asyncio.sleep(2.0)

        # "first" must complete before "second" starts
        assert "start:first" in order
        assert "end:first" in order
        assert "start:second" in order
        first_end = order.index("end:first")
        second_start = order.index("start:second")
        assert first_end < second_start, "Same-user messages must be serialized"

        loop.stop()
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass


# ═══════════════════════════════════════════════════════════════
# Component 2: Progress heartbeat
# ═══════════════════════════════════════════════════════════════


class TestProgressHeartbeat:
    """Long-running requests send a progress message to the user."""

    @pytest.mark.asyncio
    async def test_heartbeat_fires_after_delay(self, tmp_path):
        """If processing takes longer than heartbeat delay, user gets a progress msg."""
        outbound: list[OutboundMessage] = []

        async def slow_response(**kwargs):
            await asyncio.sleep(1.5)  # Longer than heartbeat delay
            return LLMResponse(content="done", finish_reason="stop")

        bus = _make_bus()
        provider = _make_provider(slow_response)
        loop = _make_loop(bus, provider, tmp_path)

        original_publish = bus.publish_outbound

        async def capture(msg):
            outbound.append(msg)
            await original_publish(msg)

        bus.publish_outbound = capture

        # Patch heartbeat delay to 0.5s for fast test
        original_heartbeat = loop._heartbeat

        async def fast_heartbeat(channel, chat_id, delay=0.5):
            await original_heartbeat(channel, chat_id, delay=delay)

        loop._heartbeat = fast_heartbeat

        loop_task = asyncio.create_task(loop.run())

        await bus.publish_inbound(InboundMessage(
            channel="test", sender_id="u1", chat_id="c1", content="hello",
        ))

        await asyncio.sleep(2.5)

        heartbeats = [m for m in outbound if "仍在处理中" in m.content]
        assert len(heartbeats) >= 1, "Should have sent a heartbeat message"
        assert heartbeats[0].channel == "test"
        assert heartbeats[0].chat_id == "c1"

        loop.stop()
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_heartbeat_cancelled_on_fast_response(self, tmp_path):
        """If processing finishes quickly, no heartbeat is sent."""
        outbound: list[OutboundMessage] = []

        bus = _make_bus()
        provider = _make_provider()  # Instant response
        loop = _make_loop(bus, provider, tmp_path)

        original_publish = bus.publish_outbound

        async def capture(msg):
            outbound.append(msg)
            await original_publish(msg)

        bus.publish_outbound = capture

        loop_task = asyncio.create_task(loop.run())

        await bus.publish_inbound(InboundMessage(
            channel="test", sender_id="u1", chat_id="c1", content="quick",
        ))

        await asyncio.sleep(1.0)

        heartbeats = [m for m in outbound if "仍在处理中" in m.content]
        assert len(heartbeats) == 0, "No heartbeat for fast responses"

        loop.stop()
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_no_heartbeat_for_system_messages(self, tmp_path):
        """System messages (subagent announces) should not trigger heartbeat."""
        outbound: list[OutboundMessage] = []

        async def slow_response(**kwargs):
            await asyncio.sleep(1.5)
            return LLMResponse(content="done", finish_reason="stop")

        bus = _make_bus()
        provider = _make_provider(slow_response)
        loop = _make_loop(bus, provider, tmp_path)

        original_publish = bus.publish_outbound

        async def capture(msg):
            outbound.append(msg)
            await original_publish(msg)

        bus.publish_outbound = capture

        # Patch heartbeat delay
        original_heartbeat = loop._heartbeat

        async def fast_heartbeat(channel, chat_id, delay=0.5):
            await original_heartbeat(channel, chat_id, delay=delay)

        loop._heartbeat = fast_heartbeat

        loop_task = asyncio.create_task(loop.run())

        # Send a system message
        await bus.publish_inbound(InboundMessage(
            channel="system", sender_id="subagent-1",
            chat_id="test:c1", content="task result",
        ))

        await asyncio.sleep(2.5)

        heartbeats = [m for m in outbound if "仍在处理中" in m.content]
        assert len(heartbeats) == 0, "System messages should not trigger heartbeat"

        loop.stop()
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass


# ═══════════════════════════════════════════════════════════════
# Component 3: LLM call timeout safety net
# ═══════════════════════════════════════════════════════════════


class TestLLMCallTimeout:
    """LLM calls are wrapped with asyncio.wait_for to prevent indefinite hangs."""

    @pytest.mark.asyncio
    async def test_single_timeout_retries_and_succeeds(self, tmp_path):
        """First call times out, retry succeeds."""
        call_count = 0

        async def hang_then_ok(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(999)  # Hang
            return LLMResponse(content="recovered", finish_reason="stop")

        bus = _make_bus()
        provider = _make_provider(hang_then_ok)
        loop = _make_loop(bus, provider, tmp_path, llm_call_timeout=1)

        result = await loop._call_llm_with_recovery(
            [{"role": "user", "content": "test"}]
        )
        assert result.content == "recovered"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_double_timeout_returns_error(self, tmp_path):
        """Both calls time out → returns error LLMResponse."""
        async def always_hang(**kwargs):
            await asyncio.sleep(999)
            return LLMResponse(content="never", finish_reason="stop")

        bus = _make_bus()
        provider = _make_provider(always_hang)
        loop = _make_loop(bus, provider, tmp_path, llm_call_timeout=1)

        result = await loop._call_llm_with_recovery(
            [{"role": "user", "content": "test"}]
        )
        assert "超时" in result.content
        assert result.finish_reason == "error"

    @pytest.mark.asyncio
    async def test_normal_call_not_affected_by_timeout(self, tmp_path):
        """Fast LLM calls work normally with timeout wrapper."""
        bus = _make_bus()
        provider = _make_provider()  # Instant response
        loop = _make_loop(bus, provider, tmp_path, llm_call_timeout=10)

        result = await loop._call_llm_with_recovery(
            [{"role": "user", "content": "hello"}]
        )
        assert result.content == "ok"
        assert provider.chat.call_count == 1


# ═══════════════════════════════════════════════════════════════
# Graceful shutdown
# ═══════════════════════════════════════════════════════════════


class TestGracefulShutdown:
    """stop() waits for active tasks to complete."""

    @pytest.mark.asyncio
    async def test_stop_waits_for_active_tasks(self, tmp_path):
        """Active tasks finish before run() returns."""
        completed = []

        async def slow_response(**kwargs):
            await asyncio.sleep(0.5)
            completed.append("done")
            return LLMResponse(content="ok", finish_reason="stop")

        bus = _make_bus()
        provider = _make_provider(slow_response)
        loop = _make_loop(bus, provider, tmp_path)

        loop_task = asyncio.create_task(loop.run())

        await bus.publish_inbound(InboundMessage(
            channel="test", sender_id="u1", chat_id="c1", content="hello",
        ))
        await asyncio.sleep(0.1)

        # Stop while message is being processed
        loop.stop()
        await asyncio.sleep(2.0)

        assert "done" in completed, "Active task should complete before shutdown"

        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_active_tasks_tracked(self, tmp_path):
        """_active_tasks set tracks in-flight message processing."""
        hang_event = asyncio.Event()

        async def hang(**kwargs):
            await hang_event.wait()
            return LLMResponse(content="ok", finish_reason="stop")

        bus = _make_bus()
        provider = _make_provider(hang)
        loop = _make_loop(bus, provider, tmp_path)

        loop_task = asyncio.create_task(loop.run())

        await bus.publish_inbound(InboundMessage(
            channel="test", sender_id="u1", chat_id="c1", content="hello",
        ))
        await asyncio.sleep(0.3)

        assert len(loop._active_tasks) >= 1, "Should track active task"

        hang_event.set()
        await asyncio.sleep(0.5)

        assert len(loop._active_tasks) == 0, "Completed task should be removed"

        loop.stop()
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass
