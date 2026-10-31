"""Embedding backends.

Two implementations share a common :class:`Embedder` protocol:

* :class:`SentenceTransformerEmbedder` — the real all-MiniLM-L6-v2 model.
* :class:`HashingEmbedder` — a deterministic, dependency-free fallback used in
  tests and CI so the suite never downloads a model or needs a GPU. It is NOT a
  good semantic embedder; it only provides stable, L2-normalized vectors with
  mild token-overlap signal so plumbing can be exercised.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    """Minimal embedding interface used by the vector store."""

    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - protocol
        """Return an ``(len(texts), dim)`` float32 array of unit-norm vectors."""
        ...


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class HashingEmbedder:
    """Deterministic bag-of-hashed-tokens embedder for tests/CI."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _hash_feature(self, vec: np.ndarray, feature: str, weight: float) -> None:
        h = int(hashlib.md5(feature.encode()).hexdigest(), 16)
        idx = h % self.dim
        sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
        vec[idx] += sign * weight

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        toks = _TOKEN_RE.findall(text.lower())
        for tok in toks:
            self._hash_feature(vec, tok, 1.0)
        # Add token bigrams so multi-word terms (e.g. "primary endpoint",
        # "eligibility criteria") contribute a stronger, more specific signal.
        for a, b in zip(toks, toks[1:], strict=False):
            self._hash_feature(vec, f"{a}_{b}", 1.5)
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        mat = np.vstack([self._embed_one(t) for t in texts])
        return _normalize(mat)


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model (default all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str) -> None:
        # Imported lazily so the heavy dependency is only needed for real runs.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)


def get_embedder(model_name: str, *, fake: bool = False) -> Embedder:
    """Factory: return a hashing embedder when ``fake`` else the real model."""
    if fake:
        return HashingEmbedder()
    return SentenceTransformerEmbedder(model_name)
