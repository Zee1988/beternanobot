"""External watchdog service — monitors and restarts the gateway process.

Runs as a separate ``nanobot watchdog`` process.  Detects gateway death via
process polling **and** HTTP health-check failures, then restarts and notifies.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable

import httpx
from loguru import logger


class WatchdogService:
    """Monitor a gateway subprocess, restart on failure."""

    def __init__(
        self,
        gateway_port: int = 18790,
        health_port: int | None = None,
        check_interval: int = 30,
        max_failures: int = 3,
        max_restarts: int = 20,
        gateway_args: list[str] | None = None,
    ):
        self.gateway_port = gateway_port
        self.health_port = health_port or (gateway_port + 1)
        self.check_interval = check_interval
        self.max_failures = max_failures
        self.max_restarts = max_restarts
        self.gateway_args = gateway_args or []

        self._process: subprocess.Popen | None = None
        self._consecutive_failures = 0
        self._restart_count = 0
        self._running = False

    @property
    def restart_count(self) -> int:
        """Number of restarts performed so far."""
        return self._restart_count

    # -- process management --------------------------------------------------

    def start_gateway(self) -> None:
        """Launch the gateway as a child process."""
        cmd = [
            sys.executable, "-m", "nanobot", "gateway",
            "--port", str(self.gateway_port),
            *self.gateway_args,
        ]
        logger.info(f"Starting gateway: {' '.join(cmd)}")
        self._process = subprocess.Popen(cmd)
        self._consecutive_failures = 0

    def stop_gateway(self) -> None:
        """Gracefully stop the gateway child process."""
        if self._process is None:
            return
        logger.info("Stopping gateway process...")
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Gateway did not stop gracefully, killing")
            self._process.kill()
            self._process.wait(timeout=5)
        self._process = None

    def is_process_alive(self) -> bool:
        """Return True if the child process is still running."""
        if self._process is None:
            return False
        return self._process.poll() is None

    def check_health(self) -> bool:
        """HTTP GET the health endpoint.  Returns True if 200."""
        try:
            with httpx.Client(timeout=10) as client:
                r = client.get(f"http://127.0.0.1:{self.health_port}/")
                return r.status_code == 200
        except Exception:
            return False

    def restart_gateway(self, reason: str) -> None:
        """Kill the old process and start a fresh one."""
        logger.warning(f"Restarting gateway — reason: {reason}")
        self.stop_gateway()
        time.sleep(2)
        self.start_gateway()
        self._restart_count += 1

    # -- main loop -----------------------------------------------------------

    def run(self, on_restart: Callable[[str, int], None] | None = None) -> None:
        """Blocking main loop.  *on_restart(reason, count)* is called after
        each restart so the CLI layer can fire notifications."""

        self._running = True
        self.start_gateway()
        # Give gateway a moment to boot before first health check
        time.sleep(min(self.check_interval, 10))

        while self._running:
            try:
                alive = self.is_process_alive()
                healthy = self.check_health() if alive else False

                if not alive:
                    reason = "process exited"
                    logger.error(f"Gateway process exited (code={self._process.returncode if self._process else '?'})")
                    if self._restart_count >= self.max_restarts:
                        logger.critical("Max restarts reached — giving up")
                        break
                    self.restart_gateway(reason)
                    if on_restart:
                        on_restart(reason, self._restart_count)
                    time.sleep(min(self.check_interval, 10))
                    continue

                if not healthy:
                    self._consecutive_failures += 1
                    logger.warning(
                        f"Health check failed ({self._consecutive_failures}/{self.max_failures})"
                    )
                    if self._consecutive_failures >= self.max_failures:
                        reason = f"{self.max_failures} consecutive health-check failures"
                        if self._restart_count >= self.max_restarts:
                            logger.critical("Max restarts reached — giving up")
                            break
                        self.restart_gateway(reason)
                        if on_restart:
                            on_restart(reason, self._restart_count)
                        time.sleep(min(self.check_interval, 10))
                        continue
                else:
                    if self._consecutive_failures > 0:
                        logger.info("Health check recovered")
                    self._consecutive_failures = 0

                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                logger.info("Watchdog interrupted")
                break

        self.stop_gateway()
        self._running = False

    def stop(self) -> None:
        """Signal the run-loop to exit."""
        self._running = False
