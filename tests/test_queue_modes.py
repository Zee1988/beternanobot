import asyncio

import pytest

from nanobot.agent.queue_manager import QueueManager
from nanobot.agent.run_manager import RunManager
from nanobot.bus.events import InboundMessage
from nanobot.config.schema import QueueConfig


@pytest.mark.asyncio
async def test_followup_preserves_order():
    run_manager = RunManager()
    cfg = QueueConfig()
    seen: list[str] = []

    async def process(msg, ctx):
        seen.append(msg.content)
        return msg.content

    queue = QueueManager(process_message=process, run_manager=run_manager, config=cfg)

    await queue.enqueue(InboundMessage(
        channel="test", sender_id="u1", chat_id="c1", content="one",
    ), mode="followup")
    await queue.enqueue(InboundMessage(
        channel="test", sender_id="u1", chat_id="c1", content="two",
    ), mode="followup")

    await queue.wait_idle("test:c1")

    assert seen == ["one", "two"]


@pytest.mark.asyncio
async def test_collect_merges_messages():
    run_manager = RunManager()
    cfg = QueueConfig(debounce_ms=10)
    seen: list[str] = []

    async def process(msg, ctx):
        seen.append(msg.content)
        return msg.content

    queue = QueueManager(process_message=process, run_manager=run_manager, config=cfg)

    await queue.enqueue(InboundMessage(
        channel="test", sender_id="u1", chat_id="c1", content="alpha",
    ), mode="collect")
    await queue.enqueue(InboundMessage(
        channel="test", sender_id="u1", chat_id="c1", content="beta",
    ), mode="collect")

    await queue.wait_idle("test:c1")

    assert seen == ["alpha\nbeta"]


@pytest.mark.asyncio
async def test_steer_cancels_active_run():
    run_manager = RunManager()
    cfg = QueueConfig()
    seen: list[str] = []

    async def process(msg, ctx):
        if msg.content == "first":
            await ctx.cancel_event.wait()
            seen.append("first_cancelled")
            return "cancelled"
        seen.append(msg.content)
        return msg.content

    queue = QueueManager(process_message=process, run_manager=run_manager, config=cfg)

    await queue.enqueue(InboundMessage(
        channel="test", sender_id="u1", chat_id="c1", content="first",
    ), mode="followup")
    await asyncio.sleep(0.05)
    await queue.enqueue(InboundMessage(
        channel="test", sender_id="u1", chat_id="c1", content="second",
    ), mode="steer")

    await queue.wait_idle("test:c1")

    assert seen == ["first_cancelled", "second"]
