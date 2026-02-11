"""Unified redaction module for sensitive data sanitization."""

import re
import json
from typing import Any

# Blacklist field names (JSON key-level redaction)
SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "api-key",
    "token",
    "secret",
    "password",
    "authorization",
    "cookie",
    "session_id",
    "private_key",
    "access_token",
    "refresh_token",
    "bearer",
    "credential",
}

# Regex patterns (text-level redaction)
REDACT_PATTERNS = [
    # API keys / tokens
    (re.compile(r"sk-[a-zA-Z0-9]{20,}"), "[API_KEY]"),
    (re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.I), "[BEARER_TOKEN]"),
    (re.compile(r"(?:Authorization|Cookie):\s*\S+", re.I), "[AUTH_HEADER]"),
    # PII
    (re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"), "[EMAIL]"),
    (
        re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}"),
        "[PHONE]",
    ),
    # Secrets patterns
    (
        re.compile(r"(?:api[_-]?key|token|secret|password|passwd)\s*[=:]\s*\S+", re.I),
        "[REDACTED]",
    ),
    (
        re.compile(r"-----BEGIN\s+\w+\s+PRIVATE\s+KEY-----[\s\S]*?-----END", re.I),
        "[PRIVATE_KEY]",
    ),
    # AWS / cloud keys
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[AWS_KEY]"),
    (re.compile(r"ghp_[A-Za-z0-9]{36}"), "[GITHUB_TOKEN]"),
]


def redact_text(text: str) -> str:
    """Apply regex-based redaction to plain text."""
    for pattern, replacement in REDACT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Apply structural redaction to a dict (blacklist keys + value regex)."""
    result = {}
    for k, v in data.items():
        if k.lower().replace("-", "_") in SENSITIVE_KEYS:
            result[k] = "[REDACTED]"
        elif isinstance(v, str):
            result[k] = redact_text(v)
        elif isinstance(v, dict):
            result[k] = redact_dict(v)
        else:
            result[k] = v
    return result


def safe_preview(value: Any, max_chars: int = 500) -> str:
    """Safely serialize + truncate any value type for preview."""
    if value is None:
        return "(none)"
    if isinstance(value, bytes):
        return f"(bytes, {len(value)} bytes)"
    if isinstance(value, dict):
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = str(value)
    elif not isinstance(value, str):
        text = str(value)
    else:
        text = value
    text = redact_text(text)
    if len(text) > max_chars:
        text = text[:max_chars] + f"... (+{len(text) - max_chars} chars)"
    return text
