"""Numpy-based vector index with cached matrix and batch cosine similarity."""

import time

from loguru import logger

from .database import MemoryDB

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]


class VectorIndex:
    """
    Pure numpy vector index.

    - Caches id_list + normalized matrix to avoid per-query rebuilds
    - Batch cosine similarity via matrix multiplication
    - Dirty flag + TTL: refresh on next search after write or after 30s
    """

    REFRESH_TTL = 30.0  # Max seconds to use cached vectors

    def __init__(self, db: MemoryDB, dimension: int):
        if np is None:
            raise ImportError("numpy is required for VectorIndex. Install with: pip install nanobot[memory]")
        self.db = db
        self.dimension = dimension
        self._id_list: list[str] = []
        self._matrix: "np.ndarray | None" = None  # shape: (N, dim)
        self._dirty = True
        self._last_refresh: float = 0.0

    def mark_dirty(self):
        self._dirty = True

    def _refresh(self):
        """Load all embeddings from DB and build normalized matrix."""
        now = time.monotonic()
        # Skip if clean and within TTL
        if not self._dirty and self._matrix is not None:
            if now - self._last_refresh < self.REFRESH_TTL:
                return

        ids, blobs = self.db.get_all_embeddings()
        if not ids:
            self._id_list = []
            self._matrix = None
            self._dirty = False
            self._last_refresh = now
            return

        vectors = []
        valid_ids = []
        for doc_id, blob in zip(ids, blobs):
            try:
                vec = np.frombuffer(blob, dtype=np.float32)
                if vec.shape[0] == self.dimension:
                    vectors.append(vec)
                    valid_ids.append(doc_id)
            except Exception:
                continue

        if vectors:
            self._matrix = np.stack(vectors)  # (N, dim)
            # L2 normalize (precompute for fast cosine)
            norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self._matrix = self._matrix / norms
        else:
            self._matrix = None

        self._id_list = valid_ids
        self._dirty = False
        self._last_refresh = now
        logger.debug(f"VectorIndex refreshed: {len(valid_ids)} vectors cached")

    def search(self, query_vec: list[float], top_k: int = 10) -> list[tuple[str, float]]:
        """
        Batch cosine similarity search.

        Returns [(doc_id, score), ...] sorted by score descending.
        """
        self._refresh()
        if self._matrix is None or len(self._id_list) == 0:
            return []

        q = np.array(query_vec, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        # Matrix multiply: (N, dim) @ (dim,) -> (N,)
        scores = self._matrix @ q

        # Top-k indices
        k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(self._id_list[i], float(scores[i])) for i in top_indices]
