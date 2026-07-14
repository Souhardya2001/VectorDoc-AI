# VectorDoc AI — FAISS-Powered Knowledge Retrieval System

A semantic search backend for internal technical documentation (incident
postmortems, changelogs, architecture notes, integration guides). Built
around LangChain document abstractions, OpenAI embeddings, and a FAISS
`IndexFlatIP` vector index, with a FastAPI service layer and a small
console-style frontend for demoing queries.

## Architecture

```
document_loader.py  →  embedder.py  →  faiss_indexer.py  →  retriever.py  →  app.py (FastAPI)  →  frontend
     (ingest)            (embed)          (index)             (search)          (API)              (UI)
```

- **`document_loader.py`** — Reads JSON records (`{id, text, metadata}`) and
  converts each into a LangChain `Document`, with `page_content = text` and
  `metadata` carrying `id`, `title`, `tags`.
- **`embedder.py`** — Wraps `OpenAIEmbeddings` (`text-embedding-3-small`) and
  L2-normalizes every vector before it's indexed or queried.
- **`faiss_indexer.py`** — Builds a `faiss.IndexFlatIP` (exact inner-product
  search) and maintains a `position -> Document` map, since FAISS itself
  only knows vector rows, not document identity. Supports incremental
  `add()` as well as `save()`/`load()` for persistence.
- **`retriever.py`** — The single call an app actually wants: embed the
  query, search the index, return top-k results as `{id, title, tags,
  score, text}`.
- **`app.py`** — FastAPI wrapper exposing `/api/health`, `/api/documents`,
  and `/api/search`, with the index built once at startup.

## Why these specific choices

**Cosine similarity via `IndexFlatIP` + L2 normalization.**
Cosine similarity is `(a·b) / (||a|| ||b||)`. If every vector is
normalized to unit length first, that formula collapses to `a·b` — exactly
what `IndexFlatIP` computes. Normalizing once at embedding time is cheaper
than computing cosine similarity per comparison at query time, and it lets
the index itself do the ranking with zero post-processing.

**Flat (exact) index over IVF/HNSW.**
For an internal docs corpus (thousands, not billions, of vectors), brute
force inner product is fast enough that trading it for an approximate
index isn't worth the accuracy loss or the added complexity of training
an index. This is called out explicitly rather than treated as a
missed optimization — it's a deliberate scale-appropriate choice, and a
natural place to discuss trade-offs in an interview: at 10M+ documents
you'd switch to `IndexIVFFlat` or `HNSW` and accept approximate results
for query-time speed.

**Separate `position -> Document` map instead of storing metadata in FAISS.**
FAISS indexes are just vectors; they have no concept of document identity.
Keeping the mapping in the retrieval layer instead of trying to encode it
into the index keeps the index itself swappable (e.g., migrating to
`IndexIVFFlat` later doesn't touch this layer at all).

**Two embedder backends behind one interface.**
`OpenAIEmbedder` is the production path. `LocalHashEmbedder` is a
deterministic, dependency-free fallback with the same interface
(`embed_documents` / `embed_query`), used here for local development and
demoing the pipeline without needing an API key or network access. Only
`get_embedder()` needs to change to switch between them — everything
downstream (indexing, retrieval, the API, the frontend) is unaffected.
Worth being upfront about this in an interview: the local embedder is for
offline development ergonomics, not a claim about semantic search quality.

## Setup

```bash
cd backend
pip install -r requirements.txt

# Production path — real OpenAI embeddings
export OPENAI_API_KEY=sk-...

# Start the API
uvicorn app:app --reload --port 8000
```

Without `OPENAI_API_KEY` set, the app automatically falls back to
`LocalHashEmbedder` so the full pipeline still runs end-to-end.

Then open `frontend/index.html` in a browser (it talks to
`http://localhost:8000`).

## API

```
GET  /api/health              -> corpus size, embedder backend, index type, vector dim
GET  /api/documents            -> id, title, tags for every indexed document
POST /api/search {query, k}    -> ranked results: id, title, tags, score, text
```

## Example

```
POST /api/search
{"query": "How did we handle Redis failures during scaling incidents?", "k": 3}

{
  "results": [
    {"title": "Postmortem: Redis Downtime During Scale-up", "score": 0.87, "tags": ["incident","redis","failover"]},
    {"title": "Failover Strategy in Clustered Systems", "score": 0.80, "tags": ["infrastructure","failover","distributed-systems"]},
    {"title": "Best Practices for Rate Limiting", "score": 0.77, "tags": ["api","performance","retries"]}
  ]
}
```

## Possible extensions (good interview talking points)

- Hybrid search: combine FAISS cosine scores with BM25 keyword scores
  for queries with exact identifiers (error codes, service names).
- Swap `IndexFlatIP` for `IndexIVFFlat` once corpus size passes ~1M
  vectors, trading exactness for query latency.
- Persist the index (`FaissIndexer.save`/`load`) instead of rebuilding
  it in memory on every restart.
- Add a re-ranking stage (e.g., cross-encoder) on the top-20 candidates
  before returning the top-k, for higher precision at the cost of latency.
