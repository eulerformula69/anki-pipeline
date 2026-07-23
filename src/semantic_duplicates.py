from __future__ import annotations

import logging
from functools import lru_cache
from typing import Protocol

import numpy as np

LOGGER = logging.getLogger("anki_pipeline.semantic_duplicates")

DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


class StatementEmbedder(Protocol):
    def encode(self, statements: list[str]) -> np.ndarray:
        """Return normalized embeddings, one row per statement."""


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        device: str = "cpu",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, statements: list[str]) -> np.ndarray:
        if not statements:
            return np.empty((0, 0), dtype=np.float32)
        return self._model.encode(
            statements,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


@lru_cache(maxsize=1)
def default_statement_embedder() -> StatementEmbedder | None:
    try:
        return SentenceTransformerEmbedder()
    except ImportError:
        LOGGER.warning(
            "sentence-transformers is unavailable; semantic duplicate detection "
            "falls back to lexical rules."
        )
        return None


def cosine_similarity_matrix(
    statements: list[str],
    embedder: StatementEmbedder,
) -> np.ndarray:
    embeddings = np.asarray(embedder.encode(statements), dtype=np.float32)
    if embeddings.shape[0] != len(statements):
        raise ValueError("Embedder returned an unexpected number of vectors.")
    if not len(statements):
        return np.empty((0, 0), dtype=np.float32)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-12)
    return normalized @ normalized.T


def semantic_duplicate_pairs(
    statements: list[str],
    embedder: StatementEmbedder,
    *,
    threshold: float = 0.86,
) -> set[tuple[int, int]]:
    similarities = cosine_similarity_matrix(statements, embedder)
    return {
        (first, second)
        for first in range(len(statements))
        for second in range(first + 1, len(statements))
        if float(similarities[first, second]) >= threshold
    }
