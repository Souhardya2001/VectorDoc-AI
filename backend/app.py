"""
app.py

Thin FastAPI layer over the retrieval pipeline:
  document_loader -> embedder -> faiss_indexer -> retriever

Endpoints:
  GET  /api/health         -> status + corpus size + embedder backend in use
  GET  /api/documents       -> list of indexed documents (id, title, tags)
  POST /api/search          -> { query, k } -> ranked results with scores

The index is built once at startup (in-memory) from data/sample_docs.json.
"""

import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import document_loader
from embedder import get_embedder
from faiss_indexer import FaissIndexer
from retriever import Retriever

app = FastAPI(title="VectorDoc AI", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_state = {}


@app.on_event("startup")
def build_index():
    data_path = os.path.join(os.path.dirname(__file__), "data", "sample_docs.json")
    docs = document_loader.load_documents(data_path)

    embedder = get_embedder()
    t0 = time.time()
    vectors = embedder.embed_documents([d.page_content for d in docs])
    embed_ms = (time.time() - t0) * 1000

    indexer = FaissIndexer()
    indexer.build(docs, vectors)

    _state["docs"] = docs
    _state["retriever"] = Retriever(indexer, embedder)
    _state["embedder_backend"] = type(embedder).__name__
    _state["embed_ms"] = round(embed_ms, 1)
    _state["dim"] = indexer.dim


class SearchRequest(BaseModel):
    query: str
    k: int = 3


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "corpus_size": len(_state["docs"]),
        "embedder_backend": _state["embedder_backend"],
        "vector_dim": _state["dim"],
        "index_type": "IndexFlatIP"
    }


@app.get("/api/documents")
def list_documents():
    return [
        {
            "id": d.metadata.get("id"),
            "title": d.metadata.get("title"),
            "tags": d.metadata.get("tags", [])
        }
        for d in _state["docs"]
    ]


@app.post("/api/search")
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    t0 = time.time()
    results = _state["retriever"].search(req.query, k=req.k)
    latency_ms = round((time.time() - t0) * 1000, 1)

    return {
        "query": req.query,
        "latency_ms": latency_ms,
        "embedder_backend": _state["embedder_backend"],
        "results": results,
    }
