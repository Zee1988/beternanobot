"""Plain text intelligent splitting with fence-awareness."""

from __future__ import annotations

import re


def find_fences(text: str) -> list[tuple[int, int]]:
    """Detect ``` fence regions in text, returning (start, end) positions."""
    fences: list[tuple[int, int]] = []
    for m in re.finditer(r'```[^\n]*\n[\s\S]*?```', text):
        fences.append((m.start(), m.end()))
    return fences


def _in_fence(pos: int, fences: list[tuple[int, int]]) -> bool:
    """Check if a position falls inside any fence region."""
    for start, end in fences:
        if start <= pos < end:
            return True
    return False


def chunk_text(text: str, limit: int) -> list[str]:
    """Split text into chunks of at most *limit* characters.

    Split priority: paragraph boundary (\\n\\n) > newline (\\n) > space > hard cut.
    Never splits inside a fenced code block when a better break point exists.
    """
    if len(text) <= limit:
        return [text]

    fences = find_fences(text)
    chunks: list[str] = []
    remaining = text

    while len(remaining) > limit:
        window = remaining[:limit]

        # Try split points in priority order
        split_pos = _find_split(window, remaining, fences, len(text) - len(remaining), limit)

        chunk = remaining[:split_pos].rstrip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_pos:].lstrip('\n')

    if remaining.strip():
        chunks.append(remaining.strip())

    return chunks if chunks else [text]


def _find_split(
    window: str,
    remaining: str,
    fences: list[tuple[int, int]],
    offset: int,
    limit: int,
) -> int:
    """Find the best split position within the window."""
    # 1. Paragraph boundary
    pos = window.rfind('\n\n')
    if pos > limit // 4 and not _in_fence(offset + pos, fences):
        return pos + 1  # keep one newline with the chunk

    # 2. Newline
    pos = window.rfind('\n')
    if pos > limit // 4 and not _in_fence(offset + pos, fences):
        return pos + 1

    # 3. Space
    pos = window.rfind(' ')
    if pos > limit // 4 and not _in_fence(offset + pos, fences):
        return pos + 1

    # 4. Hard cut
    return limit
