"""
retriever.py

Ties embedder + faiss_indexer together into the single call an
application actually wants: "given this natural-language query, give me
the top-k most relevant documents with scores and metadata."
"""

from typing import List, TypedDict

from embedder import get_embedder
from faiss_indexer import FaissIndexer


class SearchResult(TypedDict):
    id: str
    title: str
    tags: list
    score: float
    text: str


class Retriever:
    def __init__(self, indexer: FaissIndexer, embedder=None):
        self.indexer = indexer
        self.embedder = embedder or get_embedder()

    def search(self, query: str, k: int = 3) -> List[SearchResult]:
        query_vector = self.embedder.embed_query(query)
        raw_results = self.indexer.search(query_vector, k=k)

        results: List[SearchResult] = []
        for doc, score in raw_results:
            results.append(
                {
                    "id": doc.metadata.get("id", ""),
                    "title": doc.metadata.get("title", "Untitled"),
                    "tags": doc.metadata.get("tags", []),
                    "score": round(score, 4),
                    "text": doc.page_content,
                }
            )
        return results


if __name__ == "__main__":
    import document_loader
    import embedder as embedder_module

    docs = document_loader.load_documents("data/sample_docs.json")
    emb = embedder_module.get_embedder()
    vectors = emb.embed_documents([d.page_content for d in docs])

    idx = FaissIndexer()
    idx.build(docs, vectors)

    retriever = Retriever(idx, emb)
    for r in retriever.search("How did we handle Redis failures during scaling incidents?", k=3):
        print(f"{r['score']:.4f}  {r['title']}  {r['tags']}")
