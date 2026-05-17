from __future__ import annotations

"""
Carga de documentos (PDF) y extracción de texto.

Soporta dos rutas:
- PDF nativo: extrae texto embebido (Docling si está instalado; fallback a PyPDF).
- PDF escaneado: extrae imágenes y transcribe con un modelo de visión en Ollama (Qwen2.5-VL).

La salida se devuelve como {text, pages, page_texts, ...} para alimentar el grafo de agentes.
"""

import io
import tempfile
from typing import Any

from pypdf import PdfReader

from core.ollama_http import chat_with_images
from core.settings import settings


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


def _downscale_image_bytes(data: bytes) -> bytes:
    try:
        from PIL import Image
    except Exception:
        return data

    try:
        with Image.open(io.BytesIO(data)) as im:
            im = im.convert("RGB")
            w, h = im.size
            max_dim = int(getattr(settings, "ollama_vision_max_dim", 1280) or 1280)
            if max(w, h) > max_dim and max_dim > 0:
                im.thumbnail((max_dim, max_dim))
            quality = int(getattr(settings, "ollama_vision_jpeg_quality", 70) or 70)
            quality = max(30, min(95, quality))
            out = io.BytesIO()
            im.save(out, format="JPEG", quality=quality, optimize=True)
            return out.getvalue() or data
    except Exception:
        return data


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
        data = _downscale_image_bytes(data)
        results.append({"page": i + 1, "data": data, "name": getattr(best, "name", None)})
    return results


def extract_text_from_scanned_pdf_bytes(pdf_bytes: bytes) -> dict[str, Any]:
    images = extract_images_from_pdf_bytes(pdf_bytes)
    pages = len(PdfReader(io.BytesIO(pdf_bytes)).pages)
    if not images:
        return {"text": "", "pages": pages, "images": 0}

    prompt = (
        "Extrae y transcribe SOLO la información útil para auditoría y extracción de campos. "
        "Prioriza: nombres, identificadores (DNI/NIF/CIF), fechas, importes, porcentajes, direcciones, IBAN y títulos de secciones. "
        "Evita transcribir párrafos legales largos repetitivos. "
        "No inventes información. Si no se ve, déjalo en blanco. "
        "Devuelve texto plano con saltos de línea."
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
