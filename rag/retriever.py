"""
Retriever: embeds chunks and answers top-k queries.

Currently a stub backed by exact cosine similarity over numpy arrays.
Swap ``_embed`` for a real sentence-encoder and ``_build_index`` for
FAISS / Annoy once the model backbone is decided.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from utils.logger import get_logger
from .preprocessing import Chunk

logger = get_logger(__name__)


class Retriever:
    """
    Stores chunk embeddings and retrieves the top-k most relevant chunks
    for a query string.

    Parameters
    ----------
    top_k:
        Number of chunks to return per query.
    """

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
        self._chunks: List[Chunk] = []
        self._embeddings: np.ndarray | None = None  # shape (N, D)

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, chunks: List[Chunk]) -> None:
        """Embed all chunks and store for retrieval."""
        self._chunks = chunks
        self._embeddings = np.stack([self._embed(c.text) for c in chunks])
        logger.info("Built retrieval index: %d chunks, dim=%d", len(chunks), self._embeddings.shape[1])

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> List[Tuple[Chunk, float]]:
        """
        Return up to ``top_k`` (chunk, score) pairs ranked by cosine similarity.

        Parameters
        ----------
        query:
            Natural-language question from the user.

        Returns
        -------
        List of (Chunk, float) sorted descending by relevance score.
        """
        if self._embeddings is None:
            raise RuntimeError("Call build_index() before retrieve()")

        q_emb = self._embed(query)
        scores = self._cosine(q_emb, self._embeddings)
        top_idx = np.argsort(scores)[::-1][: self.top_k]

        return [(self._chunks[i], float(scores[i])) for i in top_idx]

    # ------------------------------------------------------------------
    # Embedding stub  –  replace with real encoder
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        """
        Placeholder: bag-of-character-trigrams hashed to a 128-d vector.

        Replace with a sentence-transformer / API embedding call once the
        backbone is chosen.
        """
        dim = 128
        vec = np.zeros(dim, dtype=np.float32)
        for i in range(len(text) - 2):
            h = hash(text[i : i + 3]) % dim
            vec[h] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    @staticmethod
    def _cosine(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        safe = np.where(norms == 0, 1.0, norms)
        normed = matrix / safe
        return normed @ query
