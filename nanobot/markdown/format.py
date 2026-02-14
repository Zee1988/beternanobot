"""Unified entry point for markdown → Telegram HTML chunking."""

from __future__ import annotations

from nanobot.markdown.ir import chunk_ir, markdown_to_ir
from nanobot.markdown.render import render_telegram_html


def markdown_to_telegram_chunks(md: str, limit: int = 4000) -> list[str]:
    """Convert markdown to a list of Telegram-safe HTML strings.

    Each string is guaranteed to be valid HTML and within *limit* characters
    of plain text (before HTML tag expansion). The default limit of 4000
    leaves headroom below Telegram's 4096-character cap.

    Falls back to the raw text split into 4096-char chunks on any error.
    """
    if not md:
        return [""]

    ir = markdown_to_ir(md)
    chunks = chunk_ir(ir, limit)
    return [render_telegram_html(c) for c in chunks]
