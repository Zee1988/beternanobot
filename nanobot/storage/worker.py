"""Background index worker with bounded queue and backpressure."""

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Any

from loguru import logger

from .chunker import chunk_text
from .database import DocRecord, MemoryDB
from .embedding import EmbeddingService
from .vector import VectorIndex


@dataclass
class IndexTask:
    source_type: str  # "file" | "note" | "memory" | "observation" | "summary"
    source_path: str
    text: str
    title: str = ""
    metadata: dict[str, Any] | None = None


class IndexWorker:
    """
    Background indexing worker.

    - BoundedQueue(maxsize=1000): drops oldest on overflow
    - Write ops serialized via MemoryDB._write_lock
    - Idle maintenance: WAL checkpoint + expired observation cleanup
    """

    QUEUE_MAX = 1000
    IDLE_CLEANUP_INTERVAL = 300  # 5 min idle triggers maintenance

    def __init__(
        self,
        db: MemoryDB,
        embedding: EmbeddingService,
        vector_index: VectorIndex | None,
    ):
        self.db = db
        self.embedding = embedding
        self.vector_index = vector_index
        self._queue: asyncio.Queue[IndexTask | None] = asyncio.Queue(
            maxsize=self.QUEUE_MAX
        )
        self._task: asyncio.Task | None = None

    def start(self):
        self._task = asyncio.create_task(self._run())

    async def submit(self, task: IndexTask) -> None:
        """Submit indexing task. Drops oldest if queue is full."""
        if self._queue.full():
            try:
                self._queue.get_nowait()  # Drop oldest
                logger.warning("IndexWorker queue full, dropped oldest task")
            except asyncio.QueueEmpty:
                pass
        await self._queue.put(task)

    async def stop(self):
        await self._queue.put(None)  # Sentinel
        if self._task:
            await self._task

    async def _run(self):
        idle_counter = 0
        while True:
            try:
                task = await asyncio.wait_for(
                    self._queue.get(), timeout=60.0
                )
            except asyncio.TimeoutError:
                idle_counter += 1
                if idle_counter >= self.IDLE_CLEANUP_INTERVAL // 60:
                    await self._maintenance()
                    idle_counter = 0
                continue

            if task is None:
                break
            idle_counter = 0
            try:
                await self._process(task)
            except Exception as e:
                logger.error(f"IndexWorker process error: {e}")

    async def _process(self, task: IndexTask):
        """Process a single indexing task."""
        # v5 fix: pass source_type and source_id to chunk_text (#19)
        chunk_results = chunk_text(
            task.text,
            source_type=task.source_type,
            source_id=task.source_path,
            max_tokens=256,
        )

        docs: list[DocRecord] = []
        for chunk_info in chunk_results:
            embedding = []
            if self.embedding.available:
                try:
                    # v5 fix: encode accepts str, returns list[list[float]]
                    # Take [0] for single chunk result (#19)
                    result = await self.embedding.encode_async(chunk_info["text"])
                    embedding = result[0] if result else []
                except Exception as e:
                    logger.debug(f"Embedding failed for chunk: {e}")

            docs.append(DocRecord(
                id=chunk_info["id"],
                source_type=task.source_type,
                text=chunk_info["text"],
                title=task.title,
                embedding=embedding,
                metadata=task.metadata or {},
                content_hash=chunk_info["content_hash"],
                chunk_index=chunk_info["chunk_index"],
            ))

        if docs:
            await asyncio.to_thread(self.db.upsert_batch, docs)
            if self.vector_index is not None:
                self.vector_index.mark_dirty()
            # Update manifest for file-based sources
            if task.source_type in ("file", "note", "memory"):
                content_hash = hashlib.sha256(task.text.encode()).hexdigest()[:16]
                doc_ids = [d.id for d in docs]
                await asyncio.to_thread(
                    self.db.update_manifest,
                    task.source_path, 0.0, content_hash, doc_ids,
                )

    async def _maintenance(self):
        """Idle maintenance: WAL checkpoint + expired cleanup."""
        try:
            await asyncio.to_thread(self.db.wal_checkpoint)
            count = await asyncio.to_thread(self.db.cleanup_expired, 30)
            if count > 0:
                logger.info(f"Cleaned up {count} expired observations")
                if self.vector_index is not None:
                    self.vector_index.mark_dirty()
        except Exception as e:
            logger.error(f"Maintenance error: {e}")
