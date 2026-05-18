from __future__ import annotations

"""Cliente HTTP ligero para Ollama.

Se utiliza en dos escenarios:
- Visión/OCR (chat con imágenes) para PDFs escaneados.
- Embeddings para RAG cuando se usa ChromaDB.

Nota: este módulo evita dependencias adicionales y usa `requests` directamente.
"""

import base64
from typing import Any

import requests

from core.settings import settings


def chat_with_images(prompt: str, images: list[bytes], model: str | None = None) -> str:
    """Ejecuta un chat con imágenes contra la API de Ollama.

    Args:
        prompt: Instrucciones para el modelo de visión.
        images: Lista de bytes de imágenes (p.ej. páginas de PDF).
        model: Nombre del modelo (si None, usa el configurado por settings).

    Returns:
        Texto devuelto por el modelo (transcripción/resumen).
    """
    url = settings.ollama_base_url.rstrip("/") + "/api/chat"
    b64_images = [base64.b64encode(b).decode("ascii") for b in images]
    payload: dict[str, Any] = {
        "model": model or settings.ollama_vision_model,
        "stream": False,
        "keep_alive": "10m",
        "options": {"num_predict": settings.ollama_vision_num_predict, "temperature": 0},
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": b64_images,
            }
        ],
    }
    resp = requests.post(url, json=payload, timeout=settings.ollama_timeout_s)
    resp.raise_for_status()
    data = resp.json()
    message = data.get("message") or {}
    content = message.get("content")
    return content if isinstance(content, str) else str(content)


def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Genera embeddings para una lista de textos usando Ollama.

    Implementa compatibilidad con endpoints antiguos y nuevos:
    - /api/embed (moderno)
    - /api/embeddings (legacy)

    Args:
        texts: Textos a embebber.
        model: Modelo de embeddings (si None, usa el configurado por settings).

    Returns:
        Lista de vectores (uno por texto). Si falla, devuelve listas vacías por entrada.
    """
    if not texts:
        return []

    base = settings.ollama_base_url.rstrip("/")
    chosen_model = model or settings.ollama_embedding_model

    url = base + "/api/embed"
    payload: dict[str, Any] = {"model": chosen_model, "input": texts}
    resp = requests.post(url, json=payload, timeout=settings.ollama_timeout_s)
    if resp.status_code == 404:
        url = base + "/api/embeddings"
        embeddings: list[list[float]] = []
        for t in texts:
            r = requests.post(url, json={"model": chosen_model, "prompt": t}, timeout=settings.ollama_timeout_s)
            r.raise_for_status()
            data = r.json()
            emb = data.get("embedding")
            embeddings.append(emb if isinstance(emb, list) else [])
        return embeddings

    resp.raise_for_status()
    data = resp.json()
    out = data.get("embeddings")
    if isinstance(out, list):
        return out
    single = data.get("embedding")
    if isinstance(single, list):
        return [single]
    return [[] for _ in texts]
