"""Minimal async HTTP health endpoint for gateway liveness detection.

Uses raw ``asyncio.start_server`` — zero extra dependencies.
"""

import asyncio
import json
import time

from loguru import logger


class HealthServer:
    """Lightweight TCP server that responds to ``GET /`` with JSON health status.

    The gateway event-loop calls :meth:`heartbeat` periodically.  If no
    heartbeat arrives within *stale_after* seconds the endpoint returns 503,
    signalling a stuck event-loop (deadlock / infinite await).
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 18791, stale_after: int = 120):
        self.host = host
        self.port = port
        self.stale_after = stale_after
        self._start_time = time.monotonic()
        self._last_heartbeat = time.monotonic()
        self._server: asyncio.AbstractServer | None = None

    # -- public API ----------------------------------------------------------

    def heartbeat(self) -> None:
        """Record that the event-loop is alive."""
        self._last_heartbeat = time.monotonic()

    @property
    def is_healthy(self) -> bool:
        return (time.monotonic() - self._last_heartbeat) < self.stale_after

    @property
    def uptime(self) -> int:
        return int(time.monotonic() - self._start_time)

    async def start(self) -> None:
        self._start_time = time.monotonic()
        self._last_heartbeat = time.monotonic()
        self._server = await asyncio.start_server(self._handle, self.host, self.port)
        logger.info(f"Health endpoint listening on {self.host}:{self.port}")

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    # -- internals -----------------------------------------------------------

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            data = await asyncio.wait_for(reader.read(4096), timeout=5.0)
        except (asyncio.TimeoutError, ConnectionError):
            writer.close()
            return

        if not data or not data.startswith(b"GET"):
            writer.close()
            return

        healthy = self.is_healthy
        status_code = 200 if healthy else 503
        body = json.dumps({
            "status": "ok" if healthy else "unhealthy",
            "uptime": self.uptime,
        })
        body_bytes = body.encode()
        response = (
            f"HTTP/1.1 {status_code} {'OK' if healthy else 'Service Unavailable'}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body_bytes)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        ).encode() + body_bytes
        try:
            writer.write(response)
            await writer.drain()
        except ConnectionError:
            pass
        finally:
            writer.close()
