"""IR to Telegram HTML renderer.

Converts a MarkdownIR into Telegram-compatible HTML by inserting tags at span
boundaries with correct nesting order.
"""

from __future__ import annotations

from nanobot.markdown.ir import LinkSpan, MarkdownIR, StyleSpan

_TAG_MAP = {
    "bold": "b",
    "italic": "i",
    "strikethrough": "s",
    "code": "code",
}


def render_telegram_html(ir: MarkdownIR) -> str:
    """Render a MarkdownIR to Telegram-safe HTML."""
    if not ir.text:
        return ""

    # Build events: (position, priority, "open"/"close", tag_html_open, tag_html_close)
    events: list[tuple[int, int, str, str, str]] = []

    all_spans: list[StyleSpan | LinkSpan] = list(ir.styles) + list(ir.links)

    for span in all_spans:
        if isinstance(span, LinkSpan):
            href = _escape_attr(span.href)
            open_tag = f'<a href="{href}">'
            close_tag = "</a>"
            # Links open before styles, close after
            events.append((span.start, 0, "open", open_tag, close_tag))
            events.append((span.end, 2, "close", open_tag, close_tag))
        elif isinstance(span, StyleSpan):
            if span.style == "pre":
                open_tag = "<pre><code>"
                close_tag = "</code></pre>"
            else:
                tag = _TAG_MAP.get(span.style, "b")
                open_tag = f"<{tag}>"
                close_tag = f"</{tag}>"
            events.append((span.start, 1, "open", open_tag, close_tag))
            events.append((span.end, 1, "close", open_tag, close_tag))

    # Sort: by position, then closes before opens (correct for adjacent spans), then by priority
    events.sort(key=lambda e: (e[0], 1 if e[2] == "open" else 0, e[1]))

    # Build output
    parts: list[str] = []
    last_pos = 0
    # Track which spans are "pre" or "code" for escaping decisions
    in_code = _build_code_ranges(ir)

    for pos, _prio, action, open_tag, close_tag in events:
        if pos > last_pos:
            segment = ir.text[last_pos:pos]
            parts.append(_escape_segment(segment, last_pos, in_code))
            last_pos = pos
        if action == "open":
            parts.append(open_tag)
        else:
            parts.append(close_tag)

    # Remaining text
    if last_pos < len(ir.text):
        segment = ir.text[last_pos:]
        parts.append(_escape_segment(segment, last_pos, in_code))

    return "".join(parts)


def _build_code_ranges(ir: MarkdownIR) -> list[tuple[int, int]]:
    """Return ranges that are inside code/pre spans (no double-escaping)."""
    ranges = []
    for span in ir.styles:
        if span.style in ("code", "pre"):
            ranges.append((span.start, span.end))
    return ranges


def _in_code(pos: int, ranges: list[tuple[int, int]]) -> bool:
    for start, end in ranges:
        if start <= pos < end:
            return True
    return False


def _escape_segment(text: str, start_pos: int, code_ranges: list[tuple[int, int]]) -> str:
    """Escape HTML entities. Always escape in all regions for Telegram."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _escape_attr(value: str) -> str:
    """Escape an HTML attribute value."""
    return value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
