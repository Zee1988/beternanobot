import inspect

from nanobot.channels.telegram import TelegramChannel


def test_telegram_typing_uses_public_methods():
    on_message_src = inspect.getsource(TelegramChannel._on_message)
    assert "._start_typing" not in on_message_src
    assert ".start_typing" in on_message_src

    send_src = inspect.getsource(TelegramChannel.send)
    assert "._stop_typing" not in send_src
    assert ".stop_typing" in send_src

    stop_src = inspect.getsource(TelegramChannel.stop)
    assert "._stop_typing" not in stop_src
    assert ".stop_typing" in stop_src
