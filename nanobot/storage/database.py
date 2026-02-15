"""SQLite database for smart memory storage."""

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class DocRecord:
    id: str
    source_type: str
    text: str
    title: str = ""
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    chunk_index: int = 0


@dataclass
class SearchResult:
    id: str
    text: str
    score: float
    source_type: str
    title: str = ""
    snippet: str | None = None


class MemoryDB:
    """
    Single SQLite database with WAL mode.

    - Write connection: IndexWorker exclusive, serialized via _write_lock
    - Read connection: recall() uses, read-only mode
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()

        # Write connection (IndexWorker)
        self._write_conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._init_schema(self._write_conn)

        # Read connection (recall)
        self._read_conn = sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True, check_same_thread=False
        )

    def _init_schema(self, conn: sqlite3.Connection):
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")

        # Main docs table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS docs (
                id TEXT PRIMARY KEY,
                source_type TEXT NOT NULL,
                text TEXT NOT NULL,
                title TEXT DEFAULT '',
                embedding BLOB,
                metadata TEXT DEFAULT '{}',
                content_hash TEXT NOT NULL,
                chunk_index INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # FTS5 content table mode
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts
            USING fts5(text, title, content=docs, content_rowid=rowid)
        """)

        # Corrected triggers: INSERT only inserts, DELETE only deletes, UPDATE does both
        conn.executescript("""
            CREATE TRIGGER IF NOT EXISTS docs_fts_insert
            AFTER INSERT ON docs BEGIN
                INSERT INTO docs_fts(rowid, text, title)
                VALUES (new.rowid, new.text, new.title);
            END;

            CREATE TRIGGER IF NOT EXISTS docs_fts_delete
            AFTER DELETE ON docs BEGIN
                INSERT INTO docs_fts(docs_fts, rowid, text, title)
                VALUES('delete', old.rowid, old.text, old.title);
            END;

            CREATE TRIGGER IF NOT EXISTS docs_fts_update
            AFTER UPDATE ON docs BEGIN
                INSERT INTO docs_fts(docs_fts, rowid, text, title)
                VALUES('delete', old.rowid, old.text, old.title);
                INSERT INTO docs_fts(rowid, text, title)
                VALUES (new.rowid, new.text, new.title);
            END;
        """)

        # Manifest table (file indexing state)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS manifest (
                source_path TEXT PRIMARY KEY,
                mtime REAL,
                content_hash TEXT,
                doc_ids TEXT DEFAULT '[]',
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # Meta table (embedding model version)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        conn.commit()

    def check_embedding_compat(self, model_name: str, dimension: int) -> bool:
        """Check if embedding model is compatible with index. Returns False if rebuild needed."""
        row = self._write_conn.execute(
            "SELECT value FROM meta WHERE key = 'embedding_model'"
        ).fetchone()
        if not row:
            self._write_conn.execute(
                "INSERT OR REPLACE INTO meta VALUES ('embedding_model', ?)",
                (json.dumps({"model": model_name, "dimension": dimension}),),
            )
            self._write_conn.commit()
            return True
        stored = json.loads(row[0])
        if stored.get("dimension") != dimension or stored.get("model") != model_name:
            logger.warning(
                f"Embedding model changed: {stored} -> {model_name}/{dimension}. "
                f"Rebuilding index."
            )
            return False
        return True

    def rebuild_index(self, model_name: str, dimension: int):
        """Clear all docs and manifest, re-record model info."""
        with self._write_lock:
            self._write_conn.executescript("""
                DELETE FROM docs;
                DELETE FROM manifest;
                DELETE FROM docs_fts;
            """)
            self._write_conn.execute(
                "INSERT OR REPLACE INTO meta VALUES ('embedding_model', ?)",
                (json.dumps({"model": model_name, "dimension": dimension}),),
            )
            self._write_conn.commit()

    def upsert_doc(self, doc: DocRecord) -> None:
        """Atomic write (write-lock protected)."""
        embedding_blob = _encode_vec(doc.embedding) if doc.embedding else None
        with self._write_lock:
            self._write_conn.execute(
                """
                INSERT OR REPLACE INTO docs
                    (id, source_type, text, title, embedding, metadata,
                     content_hash, chunk_index, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    doc.id, doc.source_type, doc.text, doc.title,
                    embedding_blob, json.dumps(doc.metadata),
                    doc.content_hash, doc.chunk_index,
                ),
            )
            self._write_conn.commit()

    def upsert_batch(self, docs: list[DocRecord]) -> None:
        """Batch write (single transaction)."""
        with self._write_lock:
            for doc in docs:
                emb_blob = _encode_vec(doc.embedding) if doc.embedding else None
                self._write_conn.execute(
                    """
                    INSERT OR REPLACE INTO docs
                        (id, source_type, text, title, embedding, metadata,
                         content_hash, chunk_index, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                    """,
                    (
                        doc.id, doc.source_type, doc.text, doc.title,
                        emb_blob, json.dumps(doc.metadata),
                        doc.content_hash, doc.chunk_index,
                    ),
                )
            self._write_conn.commit()

    def delete_by_source(self, source_path: str) -> None:
        with self._write_lock:
            row = self._write_conn.execute(
                "SELECT doc_ids FROM manifest WHERE source_path = ?",
                (source_path,),
            ).fetchone()
            if row:
                for did in json.loads(row[0]):
                    self._write_conn.execute("DELETE FROM docs WHERE id = ?", (did,))
                self._write_conn.execute(
                    "DELETE FROM manifest WHERE source_path = ?", (source_path,)
                )
                self._write_conn.commit()

    def cleanup_expired(self, max_age_days: int = 30) -> int:
        """Clean up expired observations."""
        with self._write_lock:
            cursor = self._write_conn.execute(
                """
                DELETE FROM docs
                WHERE source_type = 'observation'
                AND created_at < datetime('now', ?)
                """,
                (f"-{max_age_days} days",),
            )
            count = cursor.rowcount
            self._write_conn.commit()
            return count

    def wal_checkpoint(self) -> None:
        """WAL maintenance."""
        with self._write_lock:
            self._write_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def search_fts(self, query: str, limit: int = 20) -> list[SearchResult]:
        """FTS5 full-text search (read connection)."""
        safe_query = _sanitize_fts_query(query)
        if not safe_query:
            return []
        rows = self._read_conn.execute(
            """
            SELECT d.id, d.text, d.source_type, d.title,
                   rank * -1 AS score,
                   snippet(docs_fts, 0, '<b>', '</b>', '...', 32) AS snippet
            FROM docs_fts
            JOIN docs d ON d.rowid = docs_fts.rowid
            WHERE docs_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (safe_query, limit),
        ).fetchall()
        return [
            SearchResult(
                id=r[0], text=r[1], score=r[4],
                source_type=r[2], title=r[3], snippet=r[5],
            )
            for r in rows
        ]

    def get_all_embeddings(self) -> tuple[list[str], list[bytes]]:
        """Get all doc IDs and embedding blobs (read connection)."""
        rows = self._read_conn.execute(
            "SELECT id, embedding FROM docs WHERE embedding IS NOT NULL"
        ).fetchall()
        return [r[0] for r in rows], [r[1] for r in rows]

    def get_docs_by_ids(self, ids: list[str]) -> list[DocRecord]:
        """Batch fetch docs by ID."""
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self._read_conn.execute(
            f"SELECT id, source_type, text, title, metadata, content_hash, chunk_index "
            f"FROM docs WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        return [
            DocRecord(
                id=r[0], source_type=r[1], text=r[2], title=r[3],
                metadata=json.loads(r[4]), content_hash=r[5], chunk_index=r[6],
            )
            for r in rows
        ]

    def is_file_indexed(self, source_path: str, current_mtime: float) -> bool:
        """Check if file is already indexed and unmodified (manifest check)."""
        row = self._read_conn.execute(
            "SELECT mtime FROM manifest WHERE source_path = ?",
            (source_path,),
        ).fetchone()
        if not row:
            return False
        return abs(row[0] - current_mtime) < 0.001

    def update_manifest(
        self, source_path: str, mtime: float, content_hash: str, doc_ids: list[str]
    ) -> None:
        """Update manifest record."""
        with self._write_lock:
            self._write_conn.execute(
                """
                INSERT OR REPLACE INTO manifest
                    (source_path, mtime, content_hash, doc_ids, updated_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                """,
                (source_path, mtime, content_hash, json.dumps(doc_ids)),
            )
            self._write_conn.commit()

    def close(self):
        self._read_conn.close()
        self._write_conn.close()


def _encode_vec(vec: list[float]) -> bytes:
    """float list -> bytes (numpy format, compatible with VectorIndex)."""
    import numpy as np
    return np.array(vec, dtype=np.float32).tobytes()


def _decode_vec(blob: bytes) -> list[float]:
    """bytes -> float list."""
    import numpy as np
    return np.frombuffer(blob, dtype=np.float32).tolist()


def _sanitize_fts_query(query: str) -> str:
    """Clean FTS5 query, remove special chars to prevent syntax errors."""
    import re
    cleaned = re.sub(r'["\*\(\)\-\+\^\.\:\;\!\?\@\#\$\%\&\=\[\]\{\}\|\\\/\<\>\~\`,\']', ' ', query)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if not cleaned or len(cleaned) < 2:
        return ""
    return cleaned
