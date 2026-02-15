"""Tests for Telegram channel retry and timeout functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nanobot.channels.telegram import (
    _is_recoverable_error,
    _send_with_retry,
    RECOVERABLE_ERRORS,
)


class TestIsRecoverableError:
    """Test _is_recoverable_error function."""

    def test_telegram_error_429_recoverable(self):
        """Rate limit error should be recoverable."""
        from telegram.error import TelegramError
        err = TelegramError("429: Rate limit exceeded")
        assert _is_recoverable_error(err) is True

    def test_telegram_error_500_recoverable(self):
        """Internal server error should be recoverable."""
        from telegram.error import TelegramError
        err = TelegramError("500: Internal server error")
        assert _is_recoverable_error(err) is True

    def test_telegram_error_502_recoverable(self):
        """Bad gateway should be recoverable."""
        from telegram.error import TelegramError
        err = TelegramError("502: Bad gateway")
        assert _is_recoverable_error(err) is True

    def test_telegram_error_503_recoverable(self):
        """Service unavailable should be recoverable."""
        from telegram.error import TelegramError
        err = TelegramError("503: Service unavailable")
        assert _is_recoverable_error(err) is True

    def test_telegram_error_504_recoverable(self):
        """Gateway timeout should be recoverable."""
        from telegram.error import TelegramError
        err = TelegramError("504: Gateway timeout")
        assert _is_recoverable_error(err) is True

    def test_telegram_error_400_not_recoverable(self):
        """Bad request should not be recoverable."""
        from telegram.error import TelegramError
        err = TelegramError("400: Bad request")
        assert _is_recoverable_error(err) is False

    def test_telegram_error_401_not_recoverable(self):
        """Unauthorized should not be recoverable."""
        from telegram.error import TelegramError
        err = TelegramError("401: Unauthorized")
        assert _is_recoverable_error(err) is False

    def test_timeout_error_recoverable(self):
        """Timed out error should be recoverable."""
        err = Exception("Timed out")
        assert _is_recoverable_error(err) is True

    def test_connection_reset_recoverable(self):
        """Connection reset should be recoverable."""
        err = Exception("Connection reset by peer")
        assert _is_recoverable_error(err) is True

    def test_connection_refused_recoverable(self):
        """Connection refused should be recoverable."""
        err = Exception("Connection refused")
        assert _is_recoverable_error(err) is True

    def test_network_unreachable_recoverable(self):
        """Network unreachable should be recoverable."""
        err = Exception("Network is unreachable")
        assert _is_recoverable_error(err) is True

    def test_invalid_chat_id_not_recoverable(self):
        """Invalid chat ID should not be recoverable."""
        err = Exception("Bad Request: chat not found")
        assert _is_recoverable_error(err) is False


class TestSendWithRetry:
    """Test _send_with_retry function."""

    @pytest.mark.asyncio
    async def test_send_success_first_attempt(self):
        """Should succeed on first attempt without retries."""
        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock()

        await _send_with_retry(mock_bot, chat_id=123, text="Hello", timeout=30)

        mock_bot.send_message.assert_called_once()
        # Verify timeout was passed
        call_kwargs = mock_bot.send_message.call_args.kwargs
        assert call_kwargs.get("timeout") == 30

    @pytest.mark.asyncio
    async def test_send_retry_on_recoverable_error(self):
        """Should retry on recoverable error."""
        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock(
            side_effect=[Exception("Timed out"), None]
        )

        await _send_with_retry(mock_bot, chat_id=123, text="Hello")

        assert mock_bot.send_message.call_count == 2

    @pytest.mark.asyncio
    async def test_send_retry_exponential_backoff(self):
        """Should use exponential backoff between retries."""
        mock_bot = MagicMock()
        call_times = []

        async def mock_send(*args, **kwargs):
            import asyncio
            call_times.append(asyncio.get_event_loop().time())
            if len(call_times) < 3:
                raise Exception("Timed out")

        mock_bot.send_message = mock_send

        with patch("nanobot.channels.telegram.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await _send_with_retry(mock_bot, chat_id=123, text="Hello")

            # Should have slept twice (between attempts 1-2 and 2-3)
            assert mock_sleep.call_count == 2
            # First delay = 1s * 2^0 = 1s, second = 1s * 2^1 = 2s
            assert mock_sleep.call_args_list[0][0][0] == 1.0
            assert mock_sleep.call_args_list[1][0][0] == 2.0

    @pytest.mark.asyncio
    async def test_send_max_retries_exceeded(self):
        """Should raise after max retries exceeded."""
        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock(side_effect=Exception("Timed out"))

        with pytest.raises(Exception, match="Timed out"):
            await _send_with_retry(mock_bot, chat_id=123, text="Hello")

        # Should have tried 3 times
        assert mock_bot.send_message.call_count == 3

    @pytest.mark.asyncio
    async def test_send_no_retry_on_non_recoverable_error(self):
        """Should not retry on non-recoverable error."""
        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock(side_effect=Exception("Invalid chat ID"))

        with pytest.raises(Exception, match="Invalid chat ID"):
            await _send_with_retry(mock_bot, chat_id=123, text="Hello")

        # Should only try once
        assert mock_bot.send_message.call_count == 1


class TestRecoverableErrorsSet:
    """Test RECOVERABLE_ERRORS constant."""

    def test_contains_timeout_patterns(self):
        """Should contain common timeout patterns."""
        error_strings = {e.lower() for e in RECOVERABLE_ERRORS}
        assert "timed out" in error_strings
        assert "connect timeout" in error_strings
        assert "socket timeout" in error_strings

    def test_contains_connection_patterns(self):
        """Should contain common connection error patterns."""
        error_strings = {e.lower() for e in RECOVERABLE_ERRORS}
        assert "connection reset" in error_strings
        assert "connection refused" in error_strings
        assert "connection aborted" in error_strings
