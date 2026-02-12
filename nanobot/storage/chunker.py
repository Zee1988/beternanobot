"""Token-aware text chunking for memory indexing."""

import hashlib
import re


def estimate_tokens(text: str) -> int:
    """
    Simple token estimator.

    English: split by whitespace.
    CJK: each character ≈ 0.7 tokens (adjusted from 0.6).
    CJK punctuation: each ≈ 1 token.
    """
    ascii_tokens = len(re.findall(r'[a-zA-Z]+', text))
    cjk_chars = len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', text))
    cjk_punct = len(re.findall(r'[\u3000-\u303f\uff00-\uffef]', text))
    other = len(re.findall(r'[0-9]+', text))
    return ascii_tokens + int(cjk_chars * 0.7) + cjk_punct + other


def chunk_text(
    text: str,
    source_type: str,
    source_id: str = "",
    max_tokens: int = 256,
    overlap_chars: int = 80,
) -> list[dict]:
    """
    Split text into chunks, each <= max_tokens (token-aware).

    Returns list of dicts: [{id, text, chunk_index, content_hash}, ...]
    """
    paragraphs = re.split(r'\n{2,}', text.strip())
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        candidate = current + "\n\n" + para if current else para
        if estimate_tokens(candidate) > max_tokens and current:
            chunks.append(current)
            # Overlap: keep tail for context continuity
            current = (
                current[-overlap_chars:] + "\n\n" + para if overlap_chars else para
            )
        else:
            current = candidate

    if current:
        chunks.append(current)

    results = []
    for i, chunk in enumerate(chunks):
        content_hash = hashlib.sha256(chunk.encode()).hexdigest()[:16]
        doc_id = (
            f"{source_type}:{source_id}:{i}"
            if source_id
            else f"{source_type}:{content_hash}:{i}"
        )
        results.append({
            "id": doc_id,
            "text": chunk,
            "chunk_index": i,
            "content_hash": content_hash,
        })
    return results
