"""Tests for nanobot.watchdog.notify — multi-platform notification."""

from __future__ import annotations

import json
import smtplib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from nanobot.watchdog.notify import (
    _send_dingtalk,
    _send_discord,
    _send_email_sync,
    _send_feishu,
    _send_slack,
    _send_telegram,
    notify_all,
    notify_all_sync,
)


# ---------------------------------------------------------------------------
# Helpers — fake config objects
# ---------------------------------------------------------------------------

def _make_channels(**overrides):
    """Build a minimal channels namespace for notify_all."""
    defaults = dict(
        telegram=SimpleNamespace(enabled=False, token="", allow_from=[]),
        discord=SimpleNamespace(enabled=False, token="", allow_from=[]),
        feishu=SimpleNamespace(enabled=False, app_id="", app_secret="", allow_from=[]),
        dingtalk=SimpleNamespace(enabled=False, client_id="", client_secret="", allow_from=[]),
        slack=SimpleNamespace(enabled=False, bot_token="", dm=SimpleNamespace(allow_from=[])),
        email=SimpleNamespace(
            enabled=False, smtp_host="", allow_from=[],
            smtp_port=587, smtp_username="u", smtp_password="p",
            smtp_use_ssl=False, smtp_use_tls=False, from_address="a@b.c",
        ),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_config(**ch_overrides):
    return SimpleNamespace(channels=_make_channels(**ch_overrides))


# ---------------------------------------------------------------------------
# Per-platform senders
# ---------------------------------------------------------------------------

class TestSendTelegram:
    async def test_correct_url_and_payload(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp

        with patch("nanobot.watchdog.notify.httpx.AsyncClient") as cls:
            cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            cls.return_value.__aexit__ = AsyncMock(return_value=False)
            await _send_telegram("tok123", "chat1", "hello")

        mock_client.post.assert_called_once()
        url, kwargs = mock_client.post.call_args[0][0], mock_client.post.call_args[1]
        assert "bot" + "tok123" in url
        assert kwargs["json"]["chat_id"] == "chat1"
        assert kwargs["json"]["text"] == "hello"


class TestSendDiscord:
    async def test_creates_dm_then_sends(self):
        dm_resp = MagicMock()
        dm_resp.raise_for_status = MagicMock()
        dm_resp.json.return_value = {"id": "dm_chan_42"}
        msg_resp = MagicMock()
        msg_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.side_effect = [dm_resp, msg_resp]

        with patch("nanobot.watchdog.notify.httpx.AsyncClient") as cls:
            cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            cls.return_value.__aexit__ = AsyncMock(return_value=False)
            await _send_discord("bot_tok", "user1", "hi")

        assert mock_client.post.call_count == 2
        # First call: create DM
        first_url = mock_client.post.call_args_list[0][0][0]
        assert "users/@me/channels" in first_url
        # Second call: send message to DM channel
        second_url = mock_client.post.call_args_list[1][0][0]
        assert "dm_chan_42" in second_url


class TestSendFeishu:
    async def test_gets_token_then_sends(self):
        token_resp = MagicMock()
        token_resp.raise_for_status = MagicMock()
        token_resp.json.return_value = {"tenant_access_token": "t_abc"}
        msg_resp = MagicMock()
        msg_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.side_effect = [token_resp, msg_resp]

        with patch("nanobot.watchdog.notify.httpx.AsyncClient") as cls:
            cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            cls.return_value.__aexit__ = AsyncMock(return_value=False)
            await _send_feishu("aid", "asec", "uid", "msg")

        # Verify content is json.dumps'd
        send_call = mock_client.post.call_args_list[1]
        content_field = send_call[1]["json"]["content"]
        parsed = json.loads(content_field)
        assert parsed["text"] == "msg"


class TestSendDingtalk:
    async def test_gets_token_then_sends(self):
        token_resp = MagicMock()
        token_resp.raise_for_status = MagicMock()
        token_resp.json.return_value = {"accessToken": "dt_tok"}
        msg_resp = MagicMock()
        msg_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.side_effect = [token_resp, msg_resp]

        with patch("nanobot.watchdog.notify.httpx.AsyncClient") as cls:
            cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            cls.return_value.__aexit__ = AsyncMock(return_value=False)
            await _send_dingtalk("cid", "csec", "staff1", "alert")

        send_call = mock_client.post.call_args_list[1]
        payload = send_call[1]["json"]
        parsed = json.loads(payload["msgParam"])
        assert parsed["content"] == "alert"
        assert payload["userIds"] == ["staff1"]


class TestSendSlack:
    async def test_correct_header_and_payload(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp

        with patch("nanobot.watchdog.notify.httpx.AsyncClient") as cls:
            cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            cls.return_value.__aexit__ = AsyncMock(return_value=False)
            await _send_slack("xoxb-tok", "U123", "hey")

        call_kwargs = mock_client.post.call_args[1]
        assert "Bearer xoxb-tok" in call_kwargs["headers"]["Authorization"]
        assert call_kwargs["json"]["channel"] == "U123"
        assert call_kwargs["json"]["text"] == "hey"


class TestSendEmail:
    def test_smtp_connect_login_send_quit(self):
        mock_server = MagicMock()
        cfg = SimpleNamespace(
            smtp_host="smtp.test.com", smtp_port=587,
            smtp_username="user", smtp_password="pass",
            smtp_use_ssl=False, smtp_use_tls=True,
            from_address="bot@test.com",
        )
        with patch("nanobot.watchdog.notify.smtplib.SMTP", return_value=mock_server) as mock_cls:
            _send_email_sync(cfg, "admin@test.com", "alert text")

        mock_cls.assert_called_once_with("smtp.test.com", 587, timeout=15)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("user", "pass")
        mock_server.send_message.assert_called_once()
        mock_server.quit.assert_called_once()

    def test_smtp_ssl_mode(self):
        mock_server = MagicMock()
        cfg = SimpleNamespace(
            smtp_host="smtp.test.com", smtp_port=465,
            smtp_username="user", smtp_password="pass",
            smtp_use_ssl=True, smtp_use_tls=False,
            from_address="",
        )
        with patch("nanobot.watchdog.notify.smtplib.SMTP_SSL", return_value=mock_server) as mock_cls:
            _send_email_sync(cfg, "admin@test.com", "alert")

        mock_cls.assert_called_once_with("smtp.test.com", 465, timeout=15)
        mock_server.starttls.assert_not_called()


# ---------------------------------------------------------------------------
# notify_all integration
# ---------------------------------------------------------------------------

class TestNotifyAll:
    async def test_sends_to_enabled_channels_only(self):
        config = _make_config(
            telegram=SimpleNamespace(enabled=True, token="t", allow_from=["u1"]),
            discord=SimpleNamespace(enabled=False, token="t", allow_from=["u2"]),
        )
        with patch("nanobot.watchdog.notify._send_telegram", new_callable=AsyncMock) as mock_tg:
            with patch("nanobot.watchdog.notify._send_discord", new_callable=AsyncMock) as mock_dc:
                await notify_all(config, "test")

        mock_tg.assert_called_once_with("t", "u1", "test")
        mock_dc.assert_not_called()

    async def test_single_channel_failure_does_not_block_others(self):
        config = _make_config(
            telegram=SimpleNamespace(enabled=True, token="t", allow_from=["u1"]),
            slack=SimpleNamespace(enabled=True, bot_token="xoxb", dm=SimpleNamespace(allow_from=["U1"])),
        )
        with patch("nanobot.watchdog.notify._send_telegram", new_callable=AsyncMock, side_effect=Exception("boom")):
            with patch("nanobot.watchdog.notify._send_slack", new_callable=AsyncMock) as mock_slack:
                await notify_all(config, "msg")

        mock_slack.assert_called_once()

    async def test_all_disabled_no_error(self):
        config = _make_config()  # all disabled by default
        await notify_all(config, "nothing")  # should not raise

    async def test_special_chars_serialized_correctly(self):
        """Verify text with quotes/backslashes passes through without JSON injection."""
        config = _make_config(
            telegram=SimpleNamespace(enabled=True, token="t", allow_from=["u1"]),
        )
        text = 'He said "hello\\world"'
        with patch("nanobot.watchdog.notify._send_telegram", new_callable=AsyncMock) as mock_tg:
            await notify_all(config, text)

        assert mock_tg.call_args[0][2] == text


# ---------------------------------------------------------------------------
# notify_all_sync boundary
# ---------------------------------------------------------------------------

class TestNotifyAllSync:
    def test_no_running_loop_uses_asyncio_run(self):
        config = _make_config()
        with patch("nanobot.watchdog.notify.notify_all", new_callable=AsyncMock) as mock_na:
            notify_all_sync(config, "sync msg")
        mock_na.assert_called_once_with(config, "sync msg")

    def test_with_running_loop_uses_thread_pool(self):
        """When called from inside a running loop, should use ThreadPoolExecutor."""
        import asyncio

        config = _make_config()
        called = False

        async def _inner():
            nonlocal called
            with patch("nanobot.watchdog.notify.notify_all", new_callable=AsyncMock) as mock_na:
                notify_all_sync(config, "from loop")
                called = True
                mock_na.assert_called_once()

        asyncio.run(_inner())
        assert called
