from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import hashlib

from core.ollama_http import embed_texts
from core.settings import settings


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


def _sanitize_collection_name(name: str) -> str:
    cleaned = []
    for ch in name:
        if ch.isalnum() or ch in {"_", "-"}:
            cleaned.append(ch)
    out = "".join(cleaned).lower()
    if not out:
        out = "doc"
    if len(out) > 48:
        out = out[:48]
    if not out[0].isalpha():
        out = "d_" + out
    return out


_COLLECTION_CACHE: dict[str, Any] = {}
_INDEXED_KEYS: set[str] = set()


def _get_collection(
    chunks: list[RagChunk],
    *,
    doc_id: str | None,
):
    import chromadb

    class _OllamaEmbeddingFn:
        def __call__(self, input: list[str]) -> list[list[float]]:
            return embed_texts(input)

    embedding_fn = _OllamaEmbeddingFn()

    if doc_id:
        key = f"p:{doc_id}"
        collection = _COLLECTION_CACHE.get(key)
        if collection is None:
            client = chromadb.PersistentClient(path=settings.rag_persist_dir)
            name = _sanitize_collection_name(f"doc_{doc_id}")
            collection = client.get_or_create_collection(name=name, embedding_function=embedding_fn)
            _COLLECTION_CACHE[key] = collection

        if key not in _INDEXED_KEYS:
            try:
                existing = collection.count()
            except Exception:
                existing = 0
            if existing == 0:
                ids = [f"{doc_id}_{c.id}" for c in chunks]
                metadatas = [{**(c.metadata or {}), "doc_id": doc_id} for c in chunks]
                collection.upsert(ids=ids, documents=[c.text for c in chunks], metadatas=metadatas)
            _INDEXED_KEYS.add(key)
        return collection

    h = hashlib.sha256()
    for c in chunks:
        h.update(c.id.encode("utf-8"))
        h.update(b"\0")
    short = h.hexdigest()[:12]
    key = f"e:{short}"
    collection = _COLLECTION_CACHE.get(key)
    if collection is None:
        client = chromadb.Client()
        name = _sanitize_collection_name(f"docaudit_ephemeral_{short}")
        collection = client.get_or_create_collection(name=name, embedding_function=embedding_fn)
        ids = [c.id for c in chunks]
        collection.upsert(
            ids=ids,
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        _COLLECTION_CACHE[key] = collection
        _INDEXED_KEYS.add(key)
    return collection


def retrieve_best_evidence(
    query: str,
    chunks: list[RagChunk],
    top_k: int = 3,
    doc_id: str | None = None,
) -> list[dict[str, Any]]:
    q = (query or "").strip()
    if not q or not chunks:
        return []

    try:
        import chromadb
    except Exception:
        return []

    collection = _get_collection(chunks, doc_id=doc_id)

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


def retrieve_best_evidence_batch(
    queries: list[str],
    chunks: list[RagChunk],
    top_k: int = 3,
    doc_id: str | None = None,
) -> list[list[dict[str, Any]]]:
    if not queries or not chunks:
        return [[] for _ in queries]

    cleaned: list[tuple[int, str]] = []
    for i, q in enumerate(queries):
        qq = (q or "").strip()
        if qq:
            cleaned.append((i, qq))

    if not cleaned:
        return [[] for _ in queries]

    try:
        import chromadb
    except Exception:
        return [[] for _ in queries]

    collection = _get_collection(chunks, doc_id=doc_id)
    qs = [q for _, q in cleaned]
    res = collection.query(query_texts=qs, n_results=max(1, top_k))

    all_ids = res.get("ids") or []
    all_docs = res.get("documents") or []
    all_metas = res.get("metadatas") or []
    all_dists = res.get("distances") or []

    out: list[list[dict[str, Any]]] = [[] for _ in queries]
    for row_idx, (orig_i, _) in enumerate(cleaned):
        ids = all_ids[row_idx] if row_idx < len(all_ids) else []
        docs = all_docs[row_idx] if row_idx < len(all_docs) else []
        metas = all_metas[row_idx] if row_idx < len(all_metas) else []
        dists = all_dists[row_idx] if row_idx < len(all_dists) else []
        hits: list[dict[str, Any]] = []
        for j in range(min(len(ids), len(docs), len(metas))):
            hits.append(
                {
                    "id": ids[j],
                    "text": docs[j],
                    "metadata": metas[j] or {},
                    "distance": dists[j] if j < len(dists) else None,
                }
            )
        out[orig_i] = hits
    return out
