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
    """Extrae texto embebido de un PDF (ruta preferida).

    Intenta:
    1) Docling (si está instalado) para una extracción más rica.
    2) Fallback a PyPDF para extraer texto por página.

    Args:
        pdf_bytes: Contenido del PDF en bytes.

    Returns:
        Dict con claves como `text`, `pages`, `page_texts` y `method`.
    """
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
    """Reduce resolución/calidad de una imagen para acelerar inferencia de visión (CPU).

    Si PIL no está disponible o ocurre un error, devuelve los bytes originales.
    Ajustes para Qwen2.5-VL: resolución balanceada para velocidad y calidad.
    """
    try:
        from PIL import Image
    except Exception:
        return data

    try:
        with Image.open(io.BytesIO(data)) as im:
            im = im.convert("RGB")
            w, h = im.size
            # Resolución balanceada para velocidad y calidad con Qwen2.5-VL
            max_dim = int(getattr(settings, "ollama_vision_max_dim", 1024) or 1024)
            if max(w, h) > max_dim and max_dim > 0:
                im.thumbnail((max_dim, max_dim))
            # Calidad JPEG balanceada
            quality = int(getattr(settings, "ollama_vision_jpeg_quality", 75) or 75)
            quality = max(30, min(95, quality))
            out = io.BytesIO()
            im.save(out, format="JPEG", quality=quality, optimize=True)
            return out.getvalue() or data
    except Exception:
        return data


def extract_images_from_pdf_bytes(pdf_bytes: bytes) -> list[dict[str, Any]]:
    """Extrae imágenes representativas (una por página si existe) desde un PDF.

    Nota: usa la imagen más grande por página como aproximación al contenido principal.
    """
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
    """Extrae texto desde un PDF escaneado usando EasyOCR (principal, rápido y preciso)
    y Qwen2.5-VL como fallback para casos difíciles.

    Ideal para tesis: EasyOCR es referente académico en OCR local.
    """
    images = extract_images_from_pdf_bytes(pdf_bytes)
    pages = len(PdfReader(io.BytesIO(pdf_bytes)).pages)
    if not images:
        return {"text": "", "pages": pages, "images": 0}

    page_texts = [""] * pages
    texts = []
    used_fallback = False

    # -------------------------------
    # PASO 1: EasyOCR (principal, rápido)
    # -------------------------------
    try:
        import easyocr
        from PIL import Image

        # Inicializamos lector de EasyOCR para español (y inglés como fallback)
        reader = easyocr.Reader(['es', 'en'], gpu=False)  # gpu=False para compatibilidad universal

        for img in images:
            try:
                # Cargamos la imagen
                img_pil = Image.open(io.BytesIO(img["data"]))
                # Extraemos texto con EasyOCR
                results = reader.readtext(img_pil, detail=0)  # detail=0 devuelve solo texto
                page_text = "\n".join(results).strip()

                if page_text:
                    texts.append(page_text)
                    page_idx = img["page"] - 1
                    if 0 <= page_idx < len(page_texts):
                        page_texts[page_idx] = page_text

            except Exception as e:
                print(f"Error en EasyOCR para página {img['page']}: {e}")
                continue

    except ImportError:
        print("EasyOCR no está instalado, usando fallback Qwen2.5-VL")
        used_fallback = True
    except Exception as e:
        print(f"Error general en EasyOCR: {e}, usando fallback Qwen2.5-VL")
        used_fallback = True

    # -------------------------------
    # PASO 2: Fallback Qwen2.5-VL si EasyOCR no extrajo nada
    # -------------------------------
    if not any(texts) or used_fallback:
        print("Usando fallback Qwen2.5-VL para mejor calidad en casos difíciles")
        prompt = (
            "Extrae el texto visible de la imagen. Incluye nombres, fechas, números, importes y direcciones. "
            "No agregues comentarios. Devuelve solo el texto extraído."
        )

        for img in images:
            try:
                content = chat_with_images(prompt, [img["data"]])
                content = content.strip()
                if content:
                    texts.append(content)
                    page_idx = img["page"] - 1
                    if 0 <= page_idx < len(page_texts):
                        page_texts[page_idx] = content
            except Exception as e:
                print(f"Error en Qwen2.5-VL para página {img['page']}: {e}")
                continue

        final_method = "ollama_vision"
    else:
        final_method = "easyocr"

    return {
        "text": "\n\n".join(texts).strip(),
        "pages": pages,
        "page_texts": page_texts,
        "images": len(images),
        "method": final_method
    }
