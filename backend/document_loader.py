"""
document_loader.py

Loads raw technical-document JSON records of the form:
    { "id": ..., "text": ..., "metadata": { "title": ..., "tags": [...] } }

and converts each one into a LangChain `Document` object where:
    - page_content = the raw text (what gets embedded)
    - metadata      = id + title + tags + any other fields, carried through
                       untouched so the retriever can surface them later.
"""

import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document


class DocumentLoadError(Exception):
    """Raised when a source file or record is malformed."""


def load_documents(json_path: str) -> List[Document]:
    """
    Load a JSON file containing a list of document records and convert
    each into a LangChain Document.

    Expected record shape:
        {
          "id": "doc_217",
          "text": "...",
          "metadata": {"title": "...", "tags": ["..."]}
        }

    Returns:
        List[Document] — page_content set to `text`, metadata carries
        `id` plus everything under the record's `metadata` key.
    """
    path = Path(json_path)
    if not path.exists():
        raise DocumentLoadError(f"No such file: {json_path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise DocumentLoadError("Expected a JSON array of document records.")

    documents: List[Document] = []
    for i, record in enumerate(raw):
        for field in ("id", "text"):
            if field not in record:
                raise DocumentLoadError(f"Record at index {i} is missing '{field}'")

        metadata = dict(record.get("metadata", {}))
        metadata["id"] = record["id"]

        documents.append(
            Document(page_content=record["text"], metadata=metadata)
        )

    return documents


if __name__ == "__main__":
    docs = load_documents("data/sample_docs.json")
    print(f"Loaded {len(docs)} documents")
    for d in docs[:2]:
        print(d.metadata, "->", d.page_content[:60], "...")
