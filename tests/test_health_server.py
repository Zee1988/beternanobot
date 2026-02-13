"""Tests for nanobot.gateway.health — HealthServer."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from nanobot.gateway.health import HealthServer


# ---------------------------------------------------------------------------
# Heartbeat & health state
# ---------------------------------------------------------------------------

class TestHeartbeatState:
    def test_initial_state_is_healthy(self):
        hs = HealthServer()
        assert hs.is_healthy is True

    def test_heartbeat_keeps_healthy(self):
        hs = HealthServer(stale_after=120)
        hs.heartbeat()
        assert hs.is_healthy is True

    def test_stale_heartbeat_unhealthy(self, monkeypatch):
        hs = HealthServer(stale_after=60)
        # Freeze _last_heartbeat in the past
        base = time.monotonic()
        monkeypatch.setattr(time, "monotonic", lambda: base + 120)
        assert hs.is_healthy is False

    def test_uptime_returns_seconds(self, monkeypatch):
        base = time.monotonic()
        hs = HealthServer()
        monkeypatch.setattr(time, "monotonic", lambda: base + 42)
        # uptime = monotonic() - _start_time; _start_time was set at base
        # so uptime ≈ 42 (may differ slightly due to init timing)
        assert hs.uptime >= 40


# ---------------------------------------------------------------------------
# HTTP protocol — real TCP connections
# ---------------------------------------------------------------------------

async def _raw_request(port: int, data: bytes) -> bytes:
    """Send raw bytes to localhost:port and return the response."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(data)
    await writer.drain()
    resp = await asyncio.wait_for(reader.read(8192), timeout=5)
    writer.close()
    return resp


class TestHTTPProtocol:
    async def test_get_200_ok(self):
        hs = HealthServer(port=0)  # port=0 won't work; pick a random port
        # Use a random high port
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()

        hs = HealthServer(port=port, stale_after=300)
        await hs.start()
        try:
            resp = await _raw_request(port, b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
            assert b"200 OK" in resp
            # Parse JSON body
            body_str = resp.split(b"\r\n\r\n", 1)[1].decode()
            body = json.loads(body_str)
            assert body["status"] == "ok"
            assert "uptime" in body
        finally:
            await hs.stop()

    async def test_get_503_unhealthy(self, monkeypatch):
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()

        hs = HealthServer(port=port, stale_after=1)
        await hs.start()
        try:
            # Force stale by moving monotonic forward
            original_monotonic = time.monotonic
            base = original_monotonic()
            monkeypatch.setattr(time, "monotonic", lambda: base + 999)
            resp = await _raw_request(port, b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
            assert b"503" in resp
            body_str = resp.split(b"\r\n\r\n", 1)[1].decode()
            body = json.loads(body_str)
            assert body["status"] == "unhealthy"
        finally:
            monkeypatch.setattr(time, "monotonic", original_monotonic)
            await hs.stop()

    async def test_non_get_closes_connection(self):
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()

        hs = HealthServer(port=port)
        await hs.start()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"POST / HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()
            resp = await asyncio.wait_for(reader.read(8192), timeout=3)
            # Should get empty or connection closed (no HTTP response)
            assert resp == b"" or b"HTTP" not in resp
            writer.close()
        finally:
            await hs.stop()

    async def test_empty_data_closes_connection(self):
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()

        hs = HealthServer(port=port)
        await hs.start()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            await asyncio.sleep(0.1)
        finally:
            await hs.stop()

    async def test_content_length_matches_body(self):
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()

        hs = HealthServer(port=port, stale_after=300)
        await hs.start()
        try:
            resp = await _raw_request(port, b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
            headers_part, body_part = resp.split(b"\r\n\r\n", 1)
            for line in headers_part.split(b"\r\n"):
                if line.lower().startswith(b"content-length:"):
                    cl = int(line.split(b":")[1].strip())
                    assert cl == len(body_part)
                    break
            else:
                pytest.fail("Content-Length header not found")
        finally:
            await hs.stop()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    async def test_start_and_stop(self):
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()

        hs = HealthServer(port=port)
        await hs.start()
        assert hs._server is not None
        await hs.stop()
        assert hs._server is None

    async def test_stop_without_start(self):
        hs = HealthServer()
        await hs.stop()  # should not raise
        assert hs._server is None
