"""Markdown to IR (Intermediate Representation) parser.

Converts markdown text into a plain-text string plus style/link span metadata,
enabling safe text-level splitting before HTML rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from markdown_it import MarkdownIt


@dataclass
class StyleSpan:
    start: int
    end: int
    style: str  # "bold", "italic", "strikethrough", "code", "pre"


@dataclass
class LinkSpan:
    start: int
    end: int
    href: str


@dataclass
class MarkdownIR:
    text: str = ""
    styles: list[StyleSpan] = field(default_factory=list)
    links: list[LinkSpan] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_STYLE_MAP = {
    "strong": "bold",
    "em": "italic",
    "s": "strikethrough",
}


def markdown_to_ir(md: str) -> MarkdownIR:
    """Parse markdown into an intermediate representation."""
    parser = MarkdownIt("commonmark", {"typographer": False})
    # Enable strikethrough
    parser.enable("strikethrough")
    tokens = parser.parse(md)

    ir = MarkdownIR()
    _walk_tokens(tokens, ir, set())
    return ir


def _walk_tokens(tokens: list, ir: MarkdownIR, active_styles: set[str]) -> None:
    """Recursively walk token tree, building IR text and spans."""
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        ttype = tok.type

        # --- Block-level containers: recurse into children ---
        if ttype in (
            "bullet_list_open", "ordered_list_open",
            "blockquote_open",
        ):
            # just recurse; closing token handled by matching _close
            i += 1
            continue

        if ttype in (
            "bullet_list_close", "ordered_list_close",
            "blockquote_close",
        ):
            i += 1
            continue

        # --- List item ---
        if ttype == "list_item_open":
            # Add bullet prefix
            ir.text += "• "
            i += 1
            continue

        if ttype == "list_item_close":
            i += 1
            continue

        # --- Paragraph ---
        if ttype == "paragraph_open":
            i += 1
            continue

        if ttype == "paragraph_close":
            if ir.text and not ir.text.endswith("\n"):
                ir.text += "\n"
            i += 1
            continue

        # --- Heading (render as bold text) ---
        if ttype == "heading_open":
            active_styles.add("bold")
            start = len(ir.text)
            # Process inline children until heading_close
            i += 1
            while i < len(tokens) and tokens[i].type != "heading_close":
                _walk_tokens([tokens[i]], ir, active_styles)
                i += 1
            active_styles.discard("bold")
            ir.styles.append(StyleSpan(start=start, end=len(ir.text), style="bold"))
            if ir.text and not ir.text.endswith("\n"):
                ir.text += "\n"
            i += 1  # skip heading_close
            continue

        # --- Inline container ---
        if ttype == "inline" and tok.children:
            _walk_inline(tok.children, ir, active_styles)
            i += 1
            continue

        # --- Fenced code block ---
        if ttype == "fence":
            start = len(ir.text)
            content = tok.content
            if content.endswith("\n"):
                content = content[:-1]
            ir.text += content
            ir.styles.append(StyleSpan(start=start, end=len(ir.text), style="pre"))
            ir.text += "\n"
            i += 1
            continue

        # --- Code block (indented) ---
        if ttype == "code_block":
            start = len(ir.text)
            content = tok.content
            if content.endswith("\n"):
                content = content[:-1]
            ir.text += content
            ir.styles.append(StyleSpan(start=start, end=len(ir.text), style="pre"))
            ir.text += "\n"
            i += 1
            continue

        # --- HR ---
        if ttype == "hr":
            ir.text += "---\n"
            i += 1
            continue

        i += 1


def _walk_inline(children: list, ir: MarkdownIR, active_styles: set[str]) -> None:
    """Process inline-level tokens (bold, italic, code, links, text)."""
    i = 0
    while i < len(children):
        tok = children[i]
        ttype = tok.type

        # --- Plain text ---
        if ttype == "text":
            ir.text += tok.content
            i += 1
            continue

        # --- Soft break ---
        if ttype == "softbreak":
            ir.text += "\n"
            i += 1
            continue

        # --- Hard break ---
        if ttype == "hardbreak":
            ir.text += "\n"
            i += 1
            continue

        # --- Inline code ---
        if ttype == "code_inline":
            start = len(ir.text)
            ir.text += tok.content
            ir.styles.append(StyleSpan(start=start, end=len(ir.text), style="code"))
            i += 1
            continue

        # --- Link (must be checked before generic _open handler) ---
        if ttype == "link_open":
            href = tok.attrs.get("href", "") if tok.attrs else ""
            start = len(ir.text)
            i += 1
            while i < len(children) and children[i].type != "link_close":
                _walk_inline([children[i]], ir, active_styles)
                i += 1
            ir.links.append(LinkSpan(start=start, end=len(ir.text), href=href))
            i += 1  # skip link_close
            continue

        # --- Style open (bold, italic, strikethrough) ---
        if ttype.endswith("_open"):
            base = ttype[:-5]  # e.g. "strong_open" -> "strong"
            style = _STYLE_MAP.get(base) or _STYLE_MAP.get(tok.tag, base)
            start = len(ir.text)
            # Collect children until matching close
            i += 1
            while i < len(children) and children[i].type != f"{base}_close":
                _walk_inline([children[i]], ir, active_styles | {style})
                i += 1
            ir.styles.append(StyleSpan(start=start, end=len(ir.text), style=style))
            i += 1  # skip close token
            continue

        # --- Image (render alt text) ---
        if ttype == "image":
            alt = tok.content or "image"
            ir.text += alt
            i += 1
            continue

        # --- HTML inline (pass through) ---
        if ttype == "html_inline":
            ir.text += tok.content
            i += 1
            continue

        i += 1


# ---------------------------------------------------------------------------
# Span slicing
# ---------------------------------------------------------------------------

def slice_spans(
    spans: list[StyleSpan | LinkSpan],
    start: int,
    end: int,
) -> list[StyleSpan | LinkSpan]:
    """Slice spans to fit within [start, end), adjusting offsets to 0-based."""
    result = []
    for span in spans:
        # Skip spans entirely outside the range
        if span.end <= start or span.start >= end:
            continue
        new_start = max(span.start, start) - start
        new_end = min(span.end, end) - start
        if new_start >= new_end:
            continue
        if isinstance(span, StyleSpan):
            result.append(StyleSpan(start=new_start, end=new_end, style=span.style))
        elif isinstance(span, LinkSpan):
            result.append(LinkSpan(start=new_start, end=new_end, href=span.href))
    return result


def chunk_ir(ir: MarkdownIR, limit: int) -> list[MarkdownIR]:
    """Split an IR into multiple IRs, each with text no longer than *limit*."""
    from nanobot.markdown.chunk import chunk_text

    text_chunks = chunk_text(ir.text, limit)
    if len(text_chunks) <= 1:
        return [ir]

    result: list[MarkdownIR] = []
    offset = 0
    for chunk in text_chunks:
        # Find where this chunk starts in the original text
        chunk_start = ir.text.index(chunk, offset) if chunk in ir.text[offset:] else offset
        chunk_end = chunk_start + len(chunk)

        styles = [s for s in slice_spans(ir.styles, chunk_start, chunk_end)
                  if isinstance(s, StyleSpan)]
        links = [s for s in slice_spans(ir.links, chunk_start, chunk_end)
                 if isinstance(s, LinkSpan)]

        result.append(MarkdownIR(text=chunk, styles=styles, links=links))
        offset = chunk_end

    return result
