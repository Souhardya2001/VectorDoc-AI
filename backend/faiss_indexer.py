"""
faiss_indexer.py

Builds and maintains a FAISS IndexFlatIP vector index over document
embeddings, plus a position -> document-metadata mapping (FAISS itself
only knows about vector positions, not which document each one is).

IndexFlatIP does an exact (brute-force) inner-product search — no
approximation, no training step required. Combined with L2-normalized
vectors (see embedder.py), inner product == cosine similarity, so the
index's raw scores are directly usable as similarity scores in [-1, 1].

For the corpus sizes typical of an internal technical-docs search
(thousands, not billions, of documents), exact search is fast enough
that trading accuracy for an approximate index (IVF/HNSW) isn't
justified — this is a deliberate choice, not a shortcut.
"""

import json
import pickle
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from langchain_core.documents import Document


class FaissIndexer:
    def __init__(self, dim: Optional[int] = None):
        self.dim = dim
        self.index: Optional[faiss.IndexFlatIP] = None
        # position (row in the FAISS index) -> Document
        self.id_map: dict[int, Document] = {}

    def build(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        """Build a fresh index from scratch."""
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must be the same length")
        if not embeddings:
            raise ValueError("Cannot build an index with zero documents")

        self.dim = len(embeddings[0])
        self.index = faiss.IndexFlatIP(self.dim)

        vectors = np.array(embeddings, dtype=np.float32)
        self.index.add(vectors)

        self.id_map = {i: doc for i, doc in enumerate(documents)}

    def add(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        """Append new documents to an existing index (incremental update)."""
        if self.index is None:
            self.build(documents, embeddings)
            return

        vectors = np.array(embeddings, dtype=np.float32)
        start = self.index.ntotal
        self.index.add(vectors)
        for offset, doc in enumerate(documents):
            self.id_map[start + offset] = doc

    def search(self, query_vector: List[float], k: int = 3):
        """
        Returns a list of (Document, score) tuples, best match first.
        score is the raw inner-product value, which is cosine similarity
        since both index and query vectors are L2-normalized.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        q = np.array([query_vector], dtype=np.float32)
        k = min(k, self.index.ntotal)
        scores, positions = self.index.search(q, k)

        results = []
        for score, pos in zip(scores[0], positions[0]):
            if pos == -1:
                continue
            doc = self.id_map.get(int(pos))
            if doc is not None:
                results.append((doc, float(score)))
        return results

    # --- persistence -----------------------------------------------------

    def save(self, dir_path: str) -> None:
        out = Path(dir_path)
        out.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out / "index.faiss"))
        with (out / "id_map.pkl").open("wb") as f:
            pickle.dump(self.id_map, f)
        with (out / "meta.json").open("w") as f:
            json.dump({"dim": self.dim}, f)

    @classmethod
    def load(cls, dir_path: str) -> "FaissIndexer":
        src = Path(dir_path)
        with (src / "meta.json").open() as f:
            meta = json.load(f)
        obj = cls(dim=meta["dim"])
        obj.index = faiss.read_index(str(src / "index.faiss"))
        with (src / "id_map.pkl").open("rb") as f:
            obj.id_map = pickle.load(f)
        return obj
