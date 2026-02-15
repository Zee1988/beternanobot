import asyncio

import pytest

from nanobot.agent.run_manager import RunManager


@pytest.mark.asyncio
async def test_run_wait_resolves():
    rm = RunManager()
    run_id = rm.create_run("test:1")
    waiter = asyncio.create_task(rm.wait(run_id, timeout=1))
    rm.mark_completed(run_id, "done")
    assert await waiter == "done"
