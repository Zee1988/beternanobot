"""Smart memory system with hybrid retrieval and progressive degradation."""

import asyncio
import shutil
from pathlib import Path

from loguru import logger

from nanobot.storage.database import MemoryDB
from nanobot.storage.embedding import EmbeddingService
from nanobot.storage.hooks import ObservationHook
from nanobot.storage.retrieval import HybridRetrieval
from nanobot.storage.summary import SummaryTimer
from nanobot.storage.vector import VectorIndex
from nanobot.storage.worker import IndexTask, IndexWorker


class MemoryStore:
    """
    Smart memory system entry point.

    v5 features:
    - Hybrid retrieval (vector + FTS5) with 3-level degradation
    - Async recall with 150ms timeout
    - Background indexing via IndexWorker
    - Tool observation recording
    - Periodic conversation summarization
    - Startup indexing of existing Markdown files
    - Legacy directory migration
    - Conditional VectorIndex (skipped when embedding unavailable)
    """

    _LEGACY_DIRS = ["memory"]

    def __init__(self, data_dir: Path, config: dict | None = None):
        self.data_dir = data_dir
        self.config = config or {}
        # Support both legacy (data_dir/memory/MEMORY.md) and new (data_dir/MEMORY.md) paths
        legacy_memory = data_dir / "memory" / "MEMORY.md"
        self._memory_file = legacy_memory if legacy_memory.exists() else data_dir / "MEMORY.md"
        # Notes dir: use existing memory/ dir if present, otherwise notes/
        legacy_notes = data_dir / "memory"
        self._notes_dir = legacy_notes if legacy_notes.is_dir() else data_dir / "notes"

        # Smart memory components (lazy init)
        self._db: MemoryDB | None = None
        self._embedding: EmbeddingService | None = None
        self._vector_index: VectorIndex | None = None
        self._retrieval: HybridRetrieval | None = None
        self._worker: IndexWorker | None = None
        self._observation: ObservationHook | None = None
        self._summary: SummaryTimer | None = None

    async def initialize(self):
        """Async initialization of all smart memory components."""
        try:
            self._migrate_legacy_dirs()

            db_path = self.data_dir / "memory.db"
            self._db = MemoryDB(db_path)

            model_name = self.config.get("embedding_model", "BAAI/bge-small-zh-v1.5")
            self._embedding = EmbeddingService(model_name=model_name)

            # Conditional VectorIndex (#26)
            if self._embedding.available:
                dimension = self._embedding.dimension
                if not self._db.check_embedding_compat(model_name, dimension):
                    self._db.rebuild_index(model_name, dimension)
                self._vector_index = VectorIndex(self._db, dimension)
            else:
                self._vector_index = None
                logger.info("Embedding unavailable, VectorIndex disabled (FTS-only mode)")

            self._retrieval = HybridRetrieval(
                self._db, self._embedding, self._vector_index
            )
            self._worker = IndexWorker(
                self._db, self._embedding, self._vector_index
            )
            self._worker.start()

            self._observation = ObservationHook(self._worker)

            summary_mode = self.config.get("summary_mode", "concat")
            self._summary = SummaryTimer(
                self._worker,
                mode=summary_mode,
                llm_provider=self.config.get("summary_llm_provider"),
            )
            self._summary.start()

            # Startup indexing of existing files (#21)
            await self._index_existing_files()

            logger.info("Smart memory initialized successfully")
        except Exception as e:
            logger.warning(f"Smart memory init failed, using markdown fallback: {e}")
            self._retrieval = None

    def _migrate_legacy_dirs(self):
        """Detect and migrate legacy memory directories (#22)."""
        for legacy_name in self._LEGACY_DIRS:
            legacy_dir = self.data_dir.parent / legacy_name
            if legacy_dir.exists() and legacy_dir.is_dir():
                marker = legacy_dir / ".migrated_to_v5"
                if marker.exists():
                    continue
                logger.info(f"Migrating legacy dir: {legacy_dir} -> {self._notes_dir}")
                self._notes_dir.mkdir(parents=True, exist_ok=True)
                for f in legacy_dir.iterdir():
                    if f.name.startswith("."):
                        continue
                    dest = self._notes_dir / f.name
                    if not dest.exists():
                        shutil.copy2(str(f), str(dest))
                marker.write_text(str(self._notes_dir))
                logger.info("Migration complete. Legacy dir preserved with marker.")

    async def _index_existing_files(self):
        """Scan MEMORY.md + notes dir for startup indexing (#21)."""
        files_to_index: list[tuple[Path, str]] = []

        if self._memory_file.exists():
            files_to_index.append((self._memory_file, "memory"))

        if self._notes_dir.exists():
            for f in sorted(self._notes_dir.iterdir()):
                if f.is_file() and f.suffix in (".md", ".txt"):
                    files_to_index.append((f, "note"))

        indexed = 0
        for file_path, source_type in files_to_index:
            mtime = file_path.stat().st_mtime
            if self._db and self._db.is_file_indexed(str(file_path), mtime):
                continue
            text = file_path.read_text(encoding="utf-8")
            if text.strip() and self._worker:
                await self._worker.submit(IndexTask(
                    source_type=source_type,
                    source_path=str(file_path),
                    text=text,
                    title=file_path.stem,
                ))
                indexed += 1
        if indexed:
            logger.debug(f"Queued {indexed} files for startup indexing")

    async def recall(self, query: str, timeout: float = 0.15) -> str:
        """
        Async memory retrieval with timeout.

        Returns formatted context string or empty string on timeout/failure.
        Does NOT cache result (v5 fix #20).
        """
        if not self._retrieval:
            return ""
        try:
            result = await asyncio.wait_for(
                self._retrieval.progressive_retrieve(query, top_k=5),
                timeout=timeout,
            )
            return HybridRetrieval.format_context(result, max_tokens=800)
        except asyncio.TimeoutError:
            logger.debug(f"recall() timed out ({timeout}s), skipping memory injection")
            return ""
        except Exception as e:
            logger.warning(f"recall() failed: {e}")
            return ""

    def get_memory_context(self) -> str:
        """
        Sync interface for ContextBuilder compatibility.

        v5: Only returns Markdown base memory, no recall cache (#20).
        """
        parts = []
        if self._memory_file.exists():
            try:
                parts.append(self._memory_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        # Also include today's notes if they exist
        today_file = self._notes_dir / f"{__import__('datetime').datetime.now().strftime('%Y-%m-%d')}.md"
        if today_file.exists():
            try:
                parts.append(today_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        return "\n\n".join(parts) if parts else ""

    async def on_tool_executed(self, tool_name: str, arguments: dict, result: object):
        """Forward to ObservationHook."""
        if self._observation:
            await self._observation.on_tool_executed(tool_name, arguments, result)

    def feed_message(self, role: str, content: str):
        """Forward to SummaryTimer."""
        if self._summary:
            self._summary.feed_message(role, content)

    async def shutdown(self):
        """Graceful shutdown of all components."""
        if self._summary:
            await self._summary.stop()
        if self._worker:
            await self._worker.stop()
        if self._db:
            self._db.close()
        logger.info("Smart memory shut down")
