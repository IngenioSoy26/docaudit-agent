from __future__ import annotations

import io
import tempfile
from typing import Any

from pypdf import PdfReader

from core.ollama_http import chat_with_images


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> dict[str, Any]:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = len(reader.pages)
    except Exception:
        reader = None
        pages = 0

    try:
        from docling.document_converter import DocumentConverter  # type: ignore

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as f:
            f.write(pdf_bytes)
            f.flush()
            converter = DocumentConverter()
            result = converter.convert(f.name)
            md = result.document.export_to_markdown()
        text = (md or "").strip()
        if text:
            return {"text": text, "pages": pages, "page_texts": [text], "method": "docling"}
    except Exception:
        pass

    if reader is None:
        return {"text": "", "pages": 0, "method": "pypdf"}

    texts: list[str] = []
    page_texts: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        page_text = page_text.strip()
        page_texts.append(page_text)
        if page_text:
            texts.append(page_text)
    text = "\n\n".join(texts).strip()
    return {"text": text, "pages": pages, "page_texts": page_texts, "method": "pypdf"}


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
    page_texts: list[str] = [""] * pages
    for img in images:
        content = chat_with_images(prompt, [img["data"]])
        content = content.strip()
        if content:
            texts.append(content)
            page_idx = int(img.get("page") or 1) - 1
            if 0 <= page_idx < len(page_texts):
                page_texts[page_idx] = content

    return {
        "text": "\n\n".join(texts).strip(),
        "pages": pages,
        "page_texts": page_texts,
        "images": len(images),
    }
