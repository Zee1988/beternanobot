"""Diagnostic tests to verify hang hypotheses.

Hypothesis 1: LLM single-request TCP hang (connection alive, response never comes)
Hypothesis 2: Large context causes extreme LLM latency
Hypothesis 3: Agent loop is serial — one hung message blocks all subsequent messages

These tests simulate each scenario and measure behavior.
"""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.subagent import SubagentManager
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import ExecToolConfig, SubagentConfig
from nanobot.providers.base import LLMResponse, ToolCallRequest


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


# ═══════════════════════════════════════════════════════════════
# Hypothesis 1: Single LLM request hangs (TCP half-open)
#
# Simulate: first chat() call hangs forever, second call works.
# This proves the provider is fine but one request got stuck.
# ═══════════════════════════════════════════════════════════════


class TestHypothesis1_SingleRequestHang:
    """One LLM request hangs while subsequent requests work fine."""

    @pytest.mark.asyncio
    async def test_litellm_has_no_timeout_protection(self):
        """Prove: provider.chat() has no asyncio.wait_for — a hung
        acompletion blocks the caller indefinitely."""
        import inspect
        from nanobot.providers.litellm_provider import LiteLLMProvider

        source = inspect.getsource(LiteLLMProvider.chat)
        # No wait_for or timeout in the chat method
        assert "wait_for" not in source, "chat() already has timeout (hypothesis invalid)"
        assert "timeout" not in source.lower(), "chat() already has timeout config"

    @pytest.mark.asyncio
    async def test_hung_request_blocks_process_message(self, tmp_path):
        """If provider.chat() hangs, _process_message never returns."""
        hang_started = asyncio.Event()
        hang_released = asyncio.Event()

        async def hang_once(**kwargs):
            hang_started.set()
            await hang_released.wait()  # Simulate TCP hang
            return LLMResponse(content="finally", finish_reason="stop")

        bus = _make_bus()
        provider = _make_provider(hang_once)

        loop = AgentLoop(
            bus=bus, provider=provider, workspace=tmp_path,
            model="test/model",
        )

        msg = InboundMessage(
            channel="test", sender_id="u1", chat_id="c1",
            content="hello",
        )

        # Start processing in background
        task = asyncio.create_task(loop._process_message(msg))

        # Wait for the hang to start
        await asyncio.wait_for(hang_started.wait(), timeout=5.0)

        # Verify: task is NOT done (it's stuck)
        await asyncio.sleep(0.5)
        assert not task.done(), "Expected _process_message to be stuck"

        # Release the hang
        hang_released.set()
        result = await asyncio.wait_for(task, timeout=5.0)
        assert result is not None

    @pytest.mark.asyncio
    async def test_second_request_works_after_first_hangs(self, tmp_path):
        """Provider itself is fine — only the specific request is stuck."""
        call_count = 0

        async def first_hangs_second_ok(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(999)  # First request hangs
                return LLMResponse(content="never", finish_reason="stop")
            return LLMResponse(content="instant reply", finish_reason="stop")

        provider = _make_provider(first_hangs_second_ok)

        # First call: hangs
        task1 = asyncio.create_task(
            provider.chat(messages=[], model="test/model")
        )
        await asyncio.sleep(0.1)
        assert not task1.done()

        # Second call: works immediately
        result2 = await asyncio.wait_for(
            provider.chat(messages=[], model="test/model"),
            timeout=2.0,
        )
        assert result2.content == "instant reply"

        # Cleanup
        task1.cancel()
        try:
            await task1
        except asyncio.CancelledError:
            pass


# ═══════════════════════════════════════════════════════════════
# Hypothesis 2: Agent loop is serial — one hung message blocks all
#
# The run() loop does: msg = consume(); response = _process_message(msg)
# This is sequential. If _process_message hangs, no new messages
# can be consumed.
# ═══════════════════════════════════════════════════════════════


class TestHypothesis2_SerialLoopBlocking:
    """Agent loop now processes messages concurrently (fix applied)."""

    @pytest.mark.asyncio
    async def test_agent_loop_is_concurrent(self):
        """Verify: run() uses create_task for concurrent message dispatch."""
        import inspect
        source = inspect.getsource(AgentLoop.run)
        assert "create_task" in source, \
            "Agent loop should use create_task for concurrent processing"
        assert "_handle_message" in source, \
            "Agent loop should dispatch via _handle_message"

    @pytest.mark.asyncio
    async def test_hung_message_does_not_block_others(self, tmp_path):
        """Message 1 hangs → Message 2 from different user still processed."""
        hang_event = asyncio.Event()
        processed_messages = []

        async def hang_first(**kwargs):
            msgs = kwargs.get("messages", [])
            user_msgs = [m for m in msgs if m.get("role") == "user"]
            content = user_msgs[-1]["content"] if user_msgs else ""

            if "msg1" in content:
                await hang_event.wait()
                return LLMResponse(content="msg1 done", finish_reason="stop")
            else:
                processed_messages.append(content)
                return LLMResponse(content="msg2 done", finish_reason="stop")

        bus = _make_bus()
        provider = _make_provider(hang_first)

        loop = AgentLoop(
            bus=bus, provider=provider, workspace=tmp_path,
            model="test/model",
        )

        loop_task = asyncio.create_task(loop.run())

        # Send message 1 (will hang)
        await bus.publish_inbound(InboundMessage(
            channel="test", sender_id="u1", chat_id="c1", content="msg1",
        ))
        await asyncio.sleep(0.5)

        # Send message 2 from different user
        await bus.publish_inbound(InboundMessage(
            channel="test", sender_id="u2", chat_id="c2", content="msg2",
        ))

        # msg2 SHOULD be processed even while msg1 hangs (concurrent)
        await asyncio.sleep(1.0)
        assert "msg2" in processed_messages, \
            "msg2 should be processed concurrently while msg1 hangs"

        hang_event.set()
        await asyncio.sleep(0.5)

        loop.stop()
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass


# ═══════════════════════════════════════════════════════════════
# Hypothesis 3: No per-call timing instrumentation
#
# There's no logging of how long each LLM call or tool execution
# takes, making it impossible to diagnose which step hung.
# ═══════════════════════════════════════════════════════════════


class TestHypothesis3_NoTimingInstrumentation:
    """No duration logging for LLM calls or tool executions."""

    def test_no_duration_logging_in_llm_call(self):
        """_call_llm_with_recovery has no timing/duration log."""
        import inspect
        source = inspect.getsource(AgentLoop._call_llm_with_recovery)
        # Check for any timing patterns
        has_timing = any(kw in source for kw in [
            "time.time", "time.monotonic", "perf_counter",
            "duration", "elapsed", "took",
        ])
        assert not has_timing, \
            "LLM call already has timing (hypothesis invalid)"

    def test_no_duration_logging_in_process_message(self):
        """_process_message has no per-iteration timing."""
        import inspect
        source = inspect.getsource(AgentLoop._process_message)
        has_timing = any(kw in source for kw in [
            "time.time", "time.monotonic", "perf_counter",
            "duration", "elapsed", "took",
        ])
        assert not has_timing, \
            "Message processing already has timing (hypothesis invalid)"


# ═══════════════════════════════════════════════════════════════
# Proof of concept: asyncio.wait_for would fix the hang
# ═══════════════════════════════════════════════════════════════


class TestProofOfConcept:
    """Demonstrate that wrapping provider.chat in wait_for prevents hang."""

    @pytest.mark.asyncio
    async def test_wait_for_cancels_hung_request(self):
        """asyncio.wait_for correctly cancels a hung coroutine."""
        async def hung_chat(**kwargs):
            await asyncio.sleep(999)
            return LLMResponse(content="never", finish_reason="stop")

        provider = _make_provider(hung_chat)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                provider.chat(messages=[], model="test/model"),
                timeout=0.5,
            )

    @pytest.mark.asyncio
    async def test_retry_after_timeout_succeeds(self):
        """After a timed-out request, the next request works fine."""
        call_count = 0

        async def hang_then_ok(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(999)
            return LLMResponse(content="recovered", finish_reason="stop")

        provider = _make_provider(hang_then_ok)

        # First call: times out
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                provider.chat(messages=[], model="test/model"),
                timeout=0.5,
            )

        # Second call: works
        result = await asyncio.wait_for(
            provider.chat(messages=[], model="test/model"),
            timeout=2.0,
        )
        assert result.content == "recovered"

    @pytest.mark.asyncio
    async def test_concurrent_processing_unblocks_queue(self, tmp_path):
        """If messages were processed concurrently, msg2 wouldn't wait for msg1."""
        results = []

        async def slow_then_fast(**kwargs):
            msgs = kwargs.get("messages", [])
            user_msgs = [m for m in msgs if m.get("role") == "user"]
            content = user_msgs[-1]["content"] if user_msgs else ""

            if "slow" in content:
                await asyncio.sleep(2.0)
                return LLMResponse(content="slow done", finish_reason="stop")
            results.append(content)
            return LLMResponse(content="fast done", finish_reason="stop")

        bus = _make_bus()
        provider = _make_provider(slow_then_fast)

        loop = AgentLoop(
            bus=bus, provider=provider, workspace=tmp_path,
            model="test/model",
        )

        # Process two messages concurrently (simulating the fix)
        msg1 = InboundMessage(
            channel="test", sender_id="u1", chat_id="c1", content="slow msg",
        )
        msg2 = InboundMessage(
            channel="test", sender_id="u2", chat_id="c2", content="fast msg",
        )

        t1 = asyncio.create_task(loop._process_message(msg1))
        t2 = asyncio.create_task(loop._process_message(msg2))

        # msg2 should finish before msg1
        done, _ = await asyncio.wait([t2], timeout=1.0)
        assert len(done) == 1, "fast msg should complete quickly"
        assert "fast msg" in results

        # msg1 still running
        assert not t1.done()

        # Wait for msg1
        await asyncio.wait_for(t1, timeout=5.0)
