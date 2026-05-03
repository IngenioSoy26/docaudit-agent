from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.ollama_http import embed_texts


@dataclass(frozen=True)
class RagChunk:
    id: str
    text: str
    metadata: dict[str, Any]


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    if chunk_size <= 0:
        return [t]
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    chunks: list[str] = []
    i = 0
    while i < len(t):
        end = min(len(t), i + chunk_size)
        chunk = t[i:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(t):
            break
        i = max(0, end - overlap)
    return chunks


def build_chunks_from_pages(pages: list[str] | None) -> list[RagChunk]:
    if not pages:
        return []
    chunks: list[RagChunk] = []
    for idx, page_text in enumerate(pages, start=1):
        for j, c in enumerate(chunk_text(page_text), start=1):
            chunks.append(RagChunk(id=f"p{idx}_c{j}", text=c, metadata={"page": idx}))
    return chunks


def build_chunks_from_text(text: str) -> list[RagChunk]:
    chunks: list[RagChunk] = []
    for j, c in enumerate(chunk_text(text), start=1):
        chunks.append(RagChunk(id=f"c{j}", text=c, metadata={"page": None}))
    return chunks


def retrieve_best_evidence(
    query: str,
    chunks: list[RagChunk],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    q = (query or "").strip()
    if not q or not chunks:
        return []

    try:
        import chromadb
    except Exception:
        return []

    class _OllamaEmbeddingFn:
        def __call__(self, input: list[str]) -> list[list[float]]:
            return embed_texts(input)

    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name="docaudit_ephemeral",
        embedding_function=_OllamaEmbeddingFn(),
    )
    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )
    res = collection.query(query_texts=[q], n_results=max(1, top_k))
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out: list[dict[str, Any]] = []
    for i in range(min(len(ids), len(docs), len(metas))):
        out.append(
            {
                "id": ids[i],
                "text": docs[i],
                "metadata": metas[i] or {},
                "distance": dists[i] if i < len(dists) else None,
            }
        )
    return out
