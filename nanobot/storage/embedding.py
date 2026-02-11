"""Embedding service with fastembed backend and graceful degradation."""

import asyncio
from typing import Protocol

from loguru import logger


class EmbeddingProvider(Protocol):
    dimension: int
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedProvider:
    MODELS = {
        "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "chinese": "BAAI/bge-small-zh-v1.5",
        "english": "sentence-transformers/all-MiniLM-L6-v2",
    }

    def __init__(self, model_name: str = "multilingual"):
        from fastembed import TextEmbedding

        actual = self.MODELS.get(model_name, model_name)
        self._model = TextEmbedding(model_name=actual)
        self.dimension = self._get_dimension()

    def _get_dimension(self) -> int:
        sample = list(self._model.embed(["test"]))[0]
        return len(sample)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [e.tolist() for e in self._model.embed(texts)]


class EmbeddingService:
    """
    Embedding service with wide exception handling.

    Falls back to FTS-only mode if fastembed is unavailable or fails.
    encode() accepts str | list[str] for convenience.
    """

    def __init__(self, model_name: str = "multilingual"):
        self._provider: EmbeddingProvider | None = None
        self.model_name = model_name
        self.dimension: int = 0
        self._init_provider()

    def _init_provider(self):
        try:
            self._provider = FastEmbedProvider(self.model_name)
            self.dimension = self._provider.dimension
        except Exception as e:
            # Catch all: ImportError, RuntimeError (download fail),
            # ValueError (bad model), OSError (disk full), etc.
            logger.warning(
                f"Embedding init failed ({type(e).__name__}: {e}), "
                f"falling back to FTS-only mode"
            )
            self._provider = None
            self.dimension = 0

    @property
    def available(self) -> bool:
        return self._provider is not None

    def encode(self, texts: str | list[str]) -> list[list[float]]:
        """Encode texts. Accepts single str or list[str]."""
        if not self._provider:
            return []
        if isinstance(texts, str):
            texts = [texts]
        try:
            return self._provider.encode(texts)
        except Exception as e:
            logger.warning(f"Embedding encode failed: {e}")
            return []

    async def encode_async(self, texts: str | list[str]) -> list[list[float]]:
        """Async wrapper using to_thread."""
        if not self._provider:
            return []
        return await asyncio.to_thread(self.encode, texts)
