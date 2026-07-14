"""
embedder.py

Turns document / query text into dense vectors, L2-normalized so that
inner product search (FAISS IndexFlatIP) is mathematically equivalent
to cosine similarity:

    cos(a, b) = (a . b) / (||a|| * ||b||)

If a and b are both unit-length (||a|| = ||b|| = 1), that collapses to
    cos(a, b) = a . b
which is exactly what IndexFlatIP computes. So we normalize once here
at embedding time instead of doing it at query time.

Two backends are provided behind the same interface:

  - OpenAIEmbedder   -> production path, uses LangChain's OpenAIEmbeddings
                        (text-embedding-3-small by default). Requires
                        OPENAI_API_KEY in the environment.

  - LocalHashEmbedder -> deterministic, dependency-free fallback used for
                        local development/demo/testing without network
                        access or an API key. Same interface, same
                        normalization, swap-in compatible.

Both implement `embed_documents(list[str]) -> list[list[float]]` and
`embed_query(str) -> list[float]`, matching LangChain's Embeddings
protocol, so either can be passed anywhere the app expects an embedder.
"""

import hashlib
import os
from typing import List

import numpy as np


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization. Guards against divide-by-zero on
    an all-zero vector (shouldn't happen with real embeddings, but
    cheap to guard)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return vectors / norms


class OpenAIEmbedder:
    """Production embedder backed by OpenAI via LangChain."""

    def __init__(self, model: str = "text-embedding-3-small"):
        from langchain_openai import OpenAIEmbeddings

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY not found in environment. "
                "Set it, or use LocalHashEmbedder for offline development."
            )
        self._client = OpenAIEmbeddings(model=model)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raw = np.array(self._client.embed_documents(texts))
        return l2_normalize(raw).tolist()

    def embed_query(self, text: str) -> List[float]:
        raw = np.array([self._client.embed_query(text)])
        return l2_normalize(raw)[0].tolist()


class LocalHashEmbedder:
    """
    Deterministic offline embedder for local development and demos.

    Not semantically meaningful the way a trained model's embeddings are,
    but it's stable (same text -> same vector), dimensionally consistent,
    and lets the whole ingestion -> index -> retrieve pipeline be
    exercised end to end without network access. Swap for
    OpenAIEmbedder in production by changing one line in app.py.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _vectorize(self, text: str) -> np.ndarray:
        # Bag-of-tokens hashing: each token deterministically seeds a
        # pseudo-random vector; the document vector is their sum. This
        # gives text that shares vocabulary a nonzero dot product,
        # which is enough to demo ranking behavior end-to-end.
        vec = np.zeros(self.dim, dtype=np.float64)
        tokens = text.lower().split()
        if not tokens:
            return vec
        for token in tokens:
            h = hashlib.sha256(token.encode("utf-8")).digest()
            seed = int.from_bytes(h[:8], "little")
            rng = np.random.default_rng(seed)
            vec += rng.normal(size=self.dim)
        return vec / len(tokens)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raw = np.array([self._vectorize(t) for t in texts])
        return l2_normalize(raw).tolist()

    def embed_query(self, text: str) -> List[float]:
        raw = np.array([self._vectorize(text)])
        return l2_normalize(raw)[0].tolist()


def get_embedder():
    """
    Factory: use OpenAI if a key is present, otherwise fall back to the
    local embedder. This is the single switch point referenced in the
    README/interview notes.
    """
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIEmbedder()
    return LocalHashEmbedder()
