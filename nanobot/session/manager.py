"""Session management for conversation history."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.session.tool_call_id import sanitize_tool_call_ids
from nanobot.session.turns import validate_anthropic_turns
from nanobot.utils.helpers import ensure_dir, safe_filename


def _repair_tool_pairing(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Repair tool call / result pairing in message history.

    Inspired by openclaw's session-transcript-repair: ensures every assistant
    tool_call has a matching tool result, and drops orphan tool results.
    This prevents LLM API rejections from providers that enforce strict pairing.
    """
    if not messages:
        return messages

    result: list[dict[str, Any]] = []
    # Track which tool call IDs have been seen from assistant messages
    pending_tool_ids: set[str] = set()

    for msg in messages:
        role = msg.get("role")

        if role == "assistant" and msg.get("tool_calls"):
            result.append(msg)
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id") or tc.get("function", {}).get("name", "")
                if tc_id:
                    pending_tool_ids.add(tc_id)

        elif role == "tool":
            tc_id = msg.get("tool_call_id", "")
            if tc_id in pending_tool_ids:
                result.append(msg)
                pending_tool_ids.discard(tc_id)
            else:
                # Orphan tool result — drop it
                logger.debug(f"Dropping orphan tool result: {tc_id}")

        else:
            # Before appending a non-tool message, insert synthetic results
            # for any pending (unmatched) tool calls
            if pending_tool_ids:
                for tc_id in list(pending_tool_ids):
                    result.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": "unknown",
                        "content": "[Result unavailable]",
                    })
                    logger.debug(f"Inserted synthetic tool result for: {tc_id}")
                pending_tool_ids.clear()
            result.append(msg)

    # Flush any remaining pending tool calls at the end
    for tc_id in pending_tool_ids:
        result.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "name": "unknown",
            "content": "[Result unavailable]",
        })

    return result


@dataclass
class Session:
    """
    A conversation session.
    
    Stores messages in JSONL format for easy reading and persistence.
    """
    
    key: str  # channel:chat_id
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the session."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()
    
    def get_history(
        self,
        max_messages: int = 50,
        max_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get message history for LLM context.

        Args:
            max_messages: Maximum messages to return (hard cap).
            max_tokens: Optional token budget. When set, returns as many
                        recent messages as fit within this budget.

        Returns:
            List of messages in LLM format.
        """
        # Get recent messages
        recent = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages

        # Convert to LLM format, preserving tool call structure
        history = []
        for m in recent:
            content = m["content"]
            # Normalize whitespace-only content on assistant+tool_calls messages.
            # Some providers (e.g. MiniMax) return "\n\n\n" as content alongside
            # tool_calls, then reject that same whitespace in history with error
            # 2013 "tool call result does not follow tool call".
            if m.get("tool_calls") and isinstance(content, str) and not content.strip():
                content = ""
            entry: dict[str, Any] = {"role": m["role"], "content": content}
            # Preserve tool call fields for positive reinforcement
            if "tool_calls" in m:
                entry["tool_calls"] = m["tool_calls"]
            if "tool_call_id" in m:
                entry["tool_call_id"] = m["tool_call_id"]
            if "name" in m:
                entry["name"] = m["name"]
            history.append(entry)

        # Repair tool call / result pairing before sending to LLM
        history = _repair_tool_pairing(history)

        # Validate turn ordering (Anthropic/MiniMax require strict alternation)
        history = validate_anthropic_turns(history)

        # Note: sanitize_tool_call_ids() is available but NOT applied by
        # default — most providers accept their own IDs. Only enable when
        # switching providers (e.g. replaying Anthropic history on Mistral).

        if max_tokens is not None and max_tokens > 0:
            from nanobot.storage.chunker import estimate_tokens

            # Collect atomic "rounds" from history.  A round is either:
            #   - a single message, or
            #   - a tool-call sequence: [user, assistant(tc), tool*, assistant]
            # We must keep these together so tool results always follow their
            # tool calls (MiniMax error 2013 if broken).
            rounds: list[list[dict[str, Any]]] = []
            idx = 0
            while idx < len(history):
                msg = history[idx]
                # Detect start of a tool-call sequence:
                # user → assistant(tool_calls) → tool+ → assistant
                if (
                    msg.get("role") == "user"
                    and idx + 1 < len(history)
                    and history[idx + 1].get("role") == "assistant"
                    and history[idx + 1].get("tool_calls")
                ):
                    group = [msg, history[idx + 1]]
                    j = idx + 2
                    while j < len(history) and history[j].get("role") == "tool":
                        group.append(history[j])
                        j += 1
                    # Include trailing assistant (the final text response)
                    if j < len(history) and history[j].get("role") == "assistant":
                        group.append(history[j])
                        j += 1
                    rounds.append(group)
                    idx = j
                else:
                    rounds.append([msg])
                    idx += 1

            # Walk rounds from newest to oldest, fitting within budget
            result: list[dict[str, Any]] = []
            total = 0
            for rnd in reversed(rounds):
                rnd_tokens = sum(
                    estimate_tokens(m.get("content", "") or "")
                    for m in rnd
                )
                if total + rnd_tokens > max_tokens and result:
                    break
                # Prepend round (we're walking backwards)
                result = rnd + result
                total += rnd_tokens
            return result

        return history
    
    def clear(self) -> None:
        """Clear all messages in the session."""
        self.messages = []
        self.updated_at = datetime.now()


class SessionManager:
    """
    Manages conversation sessions.
    
    Sessions are stored as JSONL files in the sessions directory.
    """
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.sessions_dir = ensure_dir(Path.home() / ".nanobot" / "sessions")
        self._cache: dict[str, Session] = {}
    
    def _get_session_path(self, key: str) -> Path:
        """Get the file path for a session."""
        safe_key = safe_filename(key.replace(":", "_"))
        return self.sessions_dir / f"{safe_key}.jsonl"
    
    def get_or_create(self, key: str) -> Session:
        """
        Get an existing session or create a new one.
        
        Args:
            key: Session key (usually channel:chat_id).
        
        Returns:
            The session.
        """
        # Check cache
        if key in self._cache:
            return self._cache[key]
        
        # Try to load from disk
        session = self._load(key)
        if session is None:
            session = Session(key=key)
        
        self._cache[key] = session
        return session
    
    def _load(self, key: str) -> Session | None:
        """Load a session from disk."""
        path = self._get_session_path(key)
        
        if not path.exists():
            return None
        
        try:
            messages = []
            metadata = {}
            created_at = None
            
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    data = json.loads(line)
                    
                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
                    else:
                        messages.append(data)
            
            return Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                metadata=metadata
            )
        except Exception as e:
            logger.warning(f"Failed to load session {key}: {e}")
            return None
    
    def save(self, session: Session) -> None:
        """Save a session to disk."""
        path = self._get_session_path(session.key)
        
        with open(path, "w") as f:
            # Write metadata first
            metadata_line = {
                "_type": "metadata",
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metadata": session.metadata
            }
            f.write(json.dumps(metadata_line) + "\n")
            
            # Write messages
            for msg in session.messages:
                f.write(json.dumps(msg) + "\n")
        
        self._cache[session.key] = session
    
    def delete(self, key: str) -> bool:
        """
        Delete a session.
        
        Args:
            key: Session key.
        
        Returns:
            True if deleted, False if not found.
        """
        # Remove from cache
        self._cache.pop(key, None)
        
        # Remove file
        path = self._get_session_path(key)
        if path.exists():
            path.unlink()
            return True
        return False
    
    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all sessions.
        
        Returns:
            List of session info dicts.
        """
        sessions = []
        
        for path in self.sessions_dir.glob("*.jsonl"):
            try:
                # Read just the metadata line
                with open(path) as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        if data.get("_type") == "metadata":
                            sessions.append({
                                "key": path.stem.replace("_", ":"),
                                "created_at": data.get("created_at"),
                                "updated_at": data.get("updated_at"),
                                "path": str(path)
                            })
            except Exception:
                continue
        
        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)
