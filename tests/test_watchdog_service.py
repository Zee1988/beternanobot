"""Tests for nanobot.watchdog.service — WatchdogService."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import httpx
import pytest

from nanobot.watchdog.service import WatchdogService


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_health_port(self):
        svc = WatchdogService(gateway_port=18790)
        assert svc.health_port == 18791

    def test_custom_params(self):
        svc = WatchdogService(
            gateway_port=9000,
            health_port=9999,
            check_interval=10,
            max_failures=5,
            max_restarts=50,
            gateway_args=["--verbose"],
        )
        assert svc.gateway_port == 9000
        assert svc.health_port == 9999
        assert svc.check_interval == 10
        assert svc.max_failures == 5
        assert svc.max_restarts == 50
        assert svc.gateway_args == ["--verbose"]


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

class TestProcessManagement:
    @patch("nanobot.watchdog.service.subprocess.Popen")
    def test_start_gateway_calls_popen(self, mock_popen):
        svc = WatchdogService(gateway_port=18790)
        svc.start_gateway()
        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]
        assert "gateway" in cmd
        assert "--port" in cmd
        assert "18790" in cmd

    @patch("nanobot.watchdog.service.subprocess.Popen")
    def test_start_gateway_resets_consecutive_failures(self, mock_popen):
        svc = WatchdogService()
        svc._consecutive_failures = 5
        svc.start_gateway()
        assert svc._consecutive_failures == 0

    def test_stop_gateway_terminate_success(self):
        proc = MagicMock()
        proc.wait.return_value = 0
        svc = WatchdogService()
        svc._process = proc
        svc.stop_gateway()
        proc.terminate.assert_called_once()
        proc.kill.assert_not_called()
        assert svc._process is None

    def test_stop_gateway_terminate_timeout_escalates_to_kill(self):
        proc = MagicMock()
        proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 10), 0]
        svc = WatchdogService()
        svc._process = proc
        svc.stop_gateway()
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()
        assert svc._process is None

    def test_stop_gateway_none_process(self):
        svc = WatchdogService()
        svc._process = None
        svc.stop_gateway()  # should not raise

    def test_is_process_alive_running(self):
        proc = MagicMock()
        proc.poll.return_value = None  # still running
        svc = WatchdogService()
        svc._process = proc
        assert svc.is_process_alive() is True

    def test_is_process_alive_exited(self):
        proc = MagicMock()
        proc.poll.return_value = 1
        svc = WatchdogService()
        svc._process = proc
        assert svc.is_process_alive() is False

    def test_is_process_alive_none(self):
        svc = WatchdogService()
        svc._process = None
        assert svc.is_process_alive() is False


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    @patch("nanobot.watchdog.service.httpx.Client")
    def test_check_health_200(self, mock_client_cls):
        resp = MagicMock()
        resp.status_code = 200
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock(get=MagicMock(return_value=resp)))
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        svc = WatchdogService()
        assert svc.check_health() is True

    @patch("nanobot.watchdog.service.httpx.Client")
    def test_check_health_503(self, mock_client_cls):
        resp = MagicMock()
        resp.status_code = 503
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock(get=MagicMock(return_value=resp)))
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        svc = WatchdogService()
        assert svc.check_health() is False

    @patch("nanobot.watchdog.service.httpx.Client")
    def test_check_health_connection_error(self, mock_client_cls):
        mock_client_cls.return_value.__enter__ = MagicMock(
            return_value=MagicMock(get=MagicMock(side_effect=httpx.ConnectError("refused")))
        )
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        svc = WatchdogService()
        assert svc.check_health() is False


# ---------------------------------------------------------------------------
# Restart logic
# ---------------------------------------------------------------------------

class TestRestart:
    @patch("nanobot.watchdog.service.time.sleep")
    @patch.object(WatchdogService, "start_gateway")
    @patch.object(WatchdogService, "stop_gateway")
    def test_restart_gateway_calls_stop_sleep_start(self, mock_stop, mock_start, mock_sleep):
        svc = WatchdogService()
        svc.restart_gateway("test reason")
        mock_stop.assert_called_once()
        mock_sleep.assert_called_once_with(2)
        mock_start.assert_called_once()
        assert svc.restart_count == 1

    @patch("nanobot.watchdog.service.time.sleep")
    @patch.object(WatchdogService, "start_gateway")
    @patch.object(WatchdogService, "stop_gateway")
    def test_restart_count_increments(self, mock_stop, mock_start, mock_sleep):
        svc = WatchdogService()
        svc.restart_gateway("r1")
        svc.restart_gateway("r2")
        assert svc.restart_count == 2


# ---------------------------------------------------------------------------
# Main loop — run()
# ---------------------------------------------------------------------------

PLACEHOLDER_CONTINUE = "<!-- more run tests -->"


class TestRun:
    @patch("nanobot.watchdog.service.time.sleep")
    @patch.object(WatchdogService, "check_health", return_value=True)
    @patch.object(WatchdogService, "start_gateway")
    @patch.object(WatchdogService, "stop_gateway")
    def test_process_exit_triggers_restart(self, mock_stop, mock_start, mock_health, mock_sleep):
        svc = WatchdogService(max_restarts=5)
        proc = MagicMock()
        proc.poll.return_value = 1
        proc.returncode = 1

        call_count = 0
        orig_is_alive = svc.is_process_alive

        def fake_is_alive():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                svc._process = proc
                return False  # first check: dead
            svc._running = False  # stop after restart
            return True

        callback = MagicMock()
        with patch.object(svc, "is_process_alive", side_effect=fake_is_alive):
            with patch.object(svc, "restart_gateway", wraps=svc.restart_gateway) as mock_restart:
                # Prevent real restart from calling real stop/start
                mock_restart.side_effect = lambda reason: (
                    setattr(svc, '_restart_count', svc._restart_count + 1)
                )
                svc.run(on_restart=callback)

        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == "process exited"
        assert args[1] == 1

    @patch("nanobot.watchdog.service.time.sleep")
    @patch.object(WatchdogService, "start_gateway")
    @patch.object(WatchdogService, "stop_gateway")
    def test_consecutive_health_failures_trigger_restart(self, mock_stop, mock_start, mock_sleep):
        svc = WatchdogService(max_failures=2, max_restarts=5)
        callback = MagicMock()

        health_results = iter([False, False])
        alive_results = iter([True, True])

        call_count = 0

        def fake_is_alive():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return True
            svc._running = False
            return True

        def fake_check_health():
            try:
                return next(health_results)
            except StopIteration:
                return True

        with patch.object(svc, "is_process_alive", side_effect=fake_is_alive):
            with patch.object(svc, "check_health", side_effect=fake_check_health):
                with patch.object(svc, "restart_gateway", wraps=svc.restart_gateway) as mock_restart:
                    mock_restart.side_effect = lambda reason: (
                        setattr(svc, '_restart_count', svc._restart_count + 1),
                        setattr(svc, '_consecutive_failures', 0),
                    )
                    svc.run(on_restart=callback)

        callback.assert_called_once()
        assert "health-check" in callback.call_args[0][0]

    @patch("nanobot.watchdog.service.time.sleep")
    @patch.object(WatchdogService, "start_gateway")
    @patch.object(WatchdogService, "stop_gateway")
    def test_health_recovery_resets_failures(self, mock_stop, mock_start, mock_sleep):
        svc = WatchdogService(max_failures=3)

        call_count = 0

        def fake_is_alive():
            nonlocal call_count
            call_count += 1
            if call_count > 3:
                svc._running = False
            return True

        health_seq = iter([False, False, True])

        with patch.object(svc, "is_process_alive", side_effect=fake_is_alive):
            with patch.object(svc, "check_health", side_effect=lambda: next(health_seq, True)):
                svc.run()

        assert svc._consecutive_failures == 0

    @patch("nanobot.watchdog.service.time.sleep")
    @patch.object(WatchdogService, "start_gateway")
    @patch.object(WatchdogService, "stop_gateway")
    def test_max_restarts_exits_loop(self, mock_stop, mock_start, mock_sleep):
        svc = WatchdogService(max_restarts=0)
        proc = MagicMock()
        proc.poll.return_value = 1
        proc.returncode = 1
        svc._process = proc

        with patch.object(svc, "is_process_alive", return_value=False):
            svc.run()

        # Should have exited without restarting
        assert svc._restart_count == 0

    @patch("nanobot.watchdog.service.time.sleep")
    @patch.object(WatchdogService, "start_gateway")
    @patch.object(WatchdogService, "stop_gateway")
    def test_keyboard_interrupt_graceful_exit(self, mock_stop, mock_start, mock_sleep):
        svc = WatchdogService()

        with patch.object(svc, "is_process_alive", side_effect=KeyboardInterrupt):
            svc.run()

        assert svc._running is False
        mock_stop.assert_called()  # stop_gateway called on exit

    @patch("nanobot.watchdog.service.time.sleep")
    @patch.object(WatchdogService, "start_gateway")
    @patch.object(WatchdogService, "stop_gateway")
    def test_on_restart_receives_correct_args(self, mock_stop, mock_start, mock_sleep):
        svc = WatchdogService(max_failures=1, max_restarts=5)
        callback = MagicMock()

        def fake_is_alive():
            return True

        def do_restart(reason):
            svc._restart_count += 1
            svc._consecutive_failures = 0
            svc._running = False  # stop after first restart

        with patch.object(svc, "is_process_alive", side_effect=fake_is_alive):
            with patch.object(svc, "check_health", return_value=False):
                with patch.object(svc, "restart_gateway", side_effect=do_restart):
                    svc.run(on_restart=callback)

        reason, count = callback.call_args[0]
        assert isinstance(reason, str)
        assert count == 1
