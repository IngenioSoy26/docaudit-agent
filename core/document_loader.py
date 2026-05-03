from __future__ import annotations

import io
from typing import Any

from pypdf import PdfReader

from core.ollama_http import chat_with_images


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> dict[str, Any]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    texts: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        page_text = page_text.strip()
        if page_text:
            texts.append(page_text)
    text = "\n\n".join(texts).strip()
    return {"text": text, "pages": len(reader.pages)}


def extract_images_from_pdf_bytes(pdf_bytes: bytes) -> list[dict[str, Any]]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    results: list[dict[str, Any]] = []
    for i, page in enumerate(reader.pages):
        page_images = getattr(page, "images", None)
        if page_images is None:
            continue
        images = list(page_images)
        if not images:
            continue
        best = max(images, key=lambda im: len(getattr(im, "data", b"") or b""))
        data = getattr(best, "data", b"") or b""
        if not data:
            continue
        results.append({"page": i + 1, "data": data, "name": getattr(best, "name", None)})
    return results


def extract_text_from_scanned_pdf_bytes(pdf_bytes: bytes) -> dict[str, Any]:
    images = extract_images_from_pdf_bytes(pdf_bytes)
    pages = len(PdfReader(io.BytesIO(pdf_bytes)).pages)
    if not images:
        return {"text": "", "pages": pages, "images": 0}

    prompt = (
        "Transcribe el contenido del documento en texto plano (español si aplica). "
        "No inventes información. Si no se ve, déjalo en blanco. "
        "Mantén saltos de línea cuando sea útil."
    )

    texts: list[str] = []
    for img in images:
        content = chat_with_images(prompt, [img["data"]])
        content = content.strip()
        if content:
            texts.append(content)

    return {"text": "\n\n".join(texts).strip(), "pages": pages, "images": len(images)}
