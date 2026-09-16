from __future__ import annotations

"""
Carga de documentos y extracción de texto.

Soporta:
- PDF nativo: extrae texto embebido (Docling si está instalado; fallback a PyPDF).
- PDF escaneado: usa EasyOCR como ruta principal y Qwen2.5-VL como fallback.
- Imagen suelta (PNG/JPG/JPEG/WEBP): usa EasyOCR como ruta principal y Qwen2.5-VL como fallback.

La salida se devuelve como {text, pages, page_texts, ...} para alimentar el grafo de agentes.
"""

import io
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from core.ollama_http import chat_with_images
from core.settings import settings

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_forced_reading_mode(override: str | None = None) -> str:
    """Resuelve el modo de lectura FORZADO para ablación experimental Frente-D.

    Orden de precedencia (mayor a menor):
      1. Parámetro `override` (cuando se pasa explícitamente).
      2. Variable de entorno DOCAUDIT_FORCE_READING_MODE (para scripts batch).
      3. Atributo settings.force_reading_mode (para UI/llamadas programáticas).
      4. Valor por defecto: "auto" (cascada normal del sistema).

    Valores admitidos: auto | pypdf_only | ocr_only | vlm_only
    """
    candidates = [
        (override, "param"),
        (os.environ.get("DOCAUDIT_FORCE_READING_MODE", "").strip() or None, "env"),
        (getattr(settings, "force_reading_mode", None), "settings"),
        ("auto", "default"),
    ]
    chosen = "auto"
    for value, _src in candidates:
        if isinstance(value, str) and value.strip():
            normalized = value.strip().lower()
            if normalized in {"auto", "pypdf_only", "ocr_only", "vlm_only"}:
                chosen = normalized
                break
    return chosen


def _trace_ocr(message: str) -> None:
    trace_path = os.environ.get("DOCAUDIT_OCR_TRACE_PATH", "").strip()
    if not trace_path:
        return
    try:
        with Path(trace_path).open("a", encoding="utf-8") as fh:
            fh.write(message + "\n")
    except Exception:
        pass


def infer_doc_type_from_name(file_name: str) -> str:
    name = (file_name or "").lower()
    if "scanned_blurry_pdf" in name:
        return "scanned_blurry_pdf"
    if "native_pdf" in name:
        return "native_pdf"
    if "image_photo" in name:
        return "image_photo"
    if "image_handwritten" in name:
        return "image_handwritten"
    if name.endswith((".jpg", ".jpeg", ".png", ".webp")):
        return "image"
    if name.endswith(".pdf"):
        return "pdf"
    return "unknown"


def _normalize_ocr_backend_name(name: str | None) -> str:
    value = (name or "easyocr").strip().lower()
    if value not in {"easyocr", "paddleocr", "rapidocr"}:
        return "easyocr"
    return value


def _resolve_ocr_backend_for_doc_type(doc_type: str) -> str:
    mapping = {
        "native_pdf": getattr(settings, "ocr_backend_native_pdf", "easyocr"),
        "scanned_blurry_pdf": getattr(settings, "ocr_backend_scanned_blurry_pdf", "paddleocr"),
        "image_photo": getattr(settings, "ocr_backend_image_photo", "paddleocr"),
        "image_handwritten": getattr(settings, "ocr_backend_image_handwritten", "paddleocr"),
    }
    selected = mapping.get(doc_type, getattr(settings, "ocr_backend", "easyocr"))
    return _normalize_ocr_backend_name(str(selected))


@contextmanager
def extraction_policy_for_doc(file_name: str):
    """Aplica temporalmente la politica de extraccion adecuada por tipo documental."""
    doc_type = infer_doc_type_from_name(file_name)
    original = {
        "ocr_backend": getattr(settings, "ocr_backend", "easyocr"),
        "scanned_pdf_text_mode": getattr(settings, "scanned_pdf_text_mode", "auto"),
        "enable_scanned_pdf_vision_fallback": getattr(settings, "enable_scanned_pdf_vision_fallback", True),
    }
    try:
        settings.ocr_backend = _resolve_ocr_backend_for_doc_type(doc_type)
        if doc_type == "scanned_blurry_pdf":
            settings.scanned_pdf_text_mode = "ocr_only"
            settings.enable_scanned_pdf_vision_fallback = True
        elif doc_type == "native_pdf":
            settings.scanned_pdf_text_mode = "auto"
        yield doc_type
    finally:
        settings.ocr_backend = original["ocr_backend"]
        settings.scanned_pdf_text_mode = original["scanned_pdf_text_mode"]
        settings.enable_scanned_pdf_vision_fallback = original["enable_scanned_pdf_vision_fallback"]


@lru_cache(maxsize=1)
def _get_easyocr_reader() -> Any:
    import easyocr

    return easyocr.Reader(["es", "en"], gpu=False)


@lru_cache(maxsize=1)
def _get_paddleocr_reader() -> Any:
    from paddleocr import PaddleOCR

    return PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, lang="latin", show_log=False)


@lru_cache(maxsize=1)
def _get_rapidocr_reader() -> Any:
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def _text_signal_score(text: str) -> float:
    """Mide cuánta señal útil tiene un texto extraído para decidir si sirve."""
    normalized = " ".join((text or "").split())
    if not normalized:
        return 0.0

    letters = sum(ch.isalpha() for ch in normalized)
    digits = sum(ch.isdigit() for ch in normalized)
    symbols = sum(1 for ch in normalized if not ch.isalnum() and not ch.isspace())
    word_like = len(re.findall(r"[A-Za-z0-9ÁÉÍÓÚÜÑáéíóúüñ]{2,}", normalized))
    return (letters * 1.6) + (digits * 0.7) + (word_like * 7.5) - (symbols * 2.0)


def _pick_best_text(a: str, b: str) -> str:
    aa = (a or "").strip()
    bb = (b or "").strip()
    if not aa:
        return bb
    if not bb:
        return aa
    return bb if _text_signal_score(bb) > _text_signal_score(aa) else aa


def _has_meaningful_text(text: str, *, min_words: int = 4, min_score: float = 35.0) -> bool:
    """Determina si un texto es lo bastante útil como para evitar un fallback OCR."""
    normalized = " ".join((text or "").split())
    if not normalized:
        return False
    word_like = len(re.findall(r"[A-Za-z0-9ÁÉÍÓÚÜÑáéíóúüñ]{2,}", normalized))
    return word_like >= min_words and _text_signal_score(normalized) >= min_score


def _score_ocr_candidate(text: str, confidences: list[float] | None = None) -> float:
    """Calcula una puntuación heurística para elegir la mejor orientación OCR."""
    normalized = " ".join((text or "").split())
    avg_conf = (sum(confidences or []) / len(confidences)) if confidences else 0.0
    return _text_signal_score(normalized) + (avg_conf * 40.0)


def _prepare_ocr_variants(image: Any) -> list[Any]:
    """Genera variantes suaves de preprocesado para mejorar OCR sin sobrecoste extremo."""
    from PIL import ImageFilter, ImageOps

    gray = ImageOps.grayscale(image)
    base = ImageOps.autocontrast(gray)

    # En documentos pequeños, aumentar resolución ayuda mucho a EasyOCR.
    if min(base.size) < 1200:
        base = base.resize((base.width * 2, base.height * 2))

    sharpened = base.filter(ImageFilter.SHARPEN)
    binary = sharpened.point(lambda px: 255 if px >= 180 else 0)
    return [base, sharpened, binary]


def _easyocr_best_text(image_bytes: bytes, reader: Any | None = None) -> str:
    """Ejecuta EasyOCR probando varias orientaciones y devuelve la mejor lectura."""
    import numpy as np
    from PIL import Image, ImageOps

    if reader is None:
        reader = _get_easyocr_reader()
    base_img = Image.open(io.BytesIO(image_bytes))
    base_img = ImageOps.exif_transpose(base_img).convert("RGB")

    candidates: list[tuple[float, str]] = []
    for angle in (0, 90, 180, 270):
        rotated = base_img.rotate(angle, expand=True) if angle else base_img.copy()
        for prepared in _prepare_ocr_variants(rotated):
            results = reader.readtext(np.array(prepared), detail=1)
            texts = [str(item[1]).strip() for item in results if len(item) >= 2 and str(item[1]).strip()]
            confidences = [float(item[2]) for item in results if len(item) >= 3 and isinstance(item[2], (int, float))]
            page_text = "\n".join(texts).strip()
            candidates.append((_score_ocr_candidate(page_text, confidences), page_text))

    best_score, best_text = max(candidates, key=lambda item: item[0], default=(0.0, ""))
    return best_text if best_score > 0 else ""


def _extract_paddleocr_lines(result: Any) -> tuple[list[str], list[float]]:
    texts: list[str] = []
    confidences: list[float] = []

    def _append(text: Any, score: Any = None) -> None:
        value = str(text or "").strip()
        if not value:
            return
        texts.append(value)
        if isinstance(score, (int, float)):
            confidences.append(float(score))

    if isinstance(result, dict):
        rec_texts = result.get("rec_texts") or []
        rec_scores = result.get("rec_scores") or []
        for idx, text in enumerate(rec_texts):
            score = rec_scores[idx] if idx < len(rec_scores) else None
            _append(text, score)
        return texts, confidences

    if not isinstance(result, list):
        return texts, confidences

    for block in result:
        if isinstance(block, dict):
            block_texts, block_scores = _extract_paddleocr_lines(block)
            texts.extend(block_texts)
            confidences.extend(block_scores)
            continue
        if not isinstance(block, list):
            continue
        for item in block:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            rec = item[1]
            if isinstance(rec, (list, tuple)) and rec:
                score = rec[1] if len(rec) > 1 else None
                _append(rec[0], score)
            elif isinstance(rec, str):
                _append(rec)
    return texts, confidences


def _paddleocr_best_text(image_bytes: bytes, reader: Any | None = None) -> str:
    import numpy as np
    from PIL import Image, ImageOps

    if reader is None:
        reader = _get_paddleocr_reader()
    base_img = Image.open(io.BytesIO(image_bytes))
    base_img = ImageOps.exif_transpose(base_img).convert("RGB")

    candidates: list[tuple[float, str]] = []
    for angle in (0, 90, 180, 270):
        rotated = base_img.rotate(angle, expand=True) if angle else base_img.copy()
        for prepared in _prepare_ocr_variants(rotated):
            result = reader.ocr(np.array(prepared))
            texts, confidences = _extract_paddleocr_lines(result)
            page_text = "\n".join(texts).strip()
            candidates.append((_score_ocr_candidate(page_text, confidences), page_text))

    best_score, best_text = max(candidates, key=lambda item: item[0], default=(0.0, ""))
    return best_text if best_score > 0 else ""


def _rapidocr_best_text(image_bytes: bytes, reader: Any | None = None) -> str:
    import numpy as np
    from PIL import Image, ImageOps

    if reader is None:
        reader = _get_rapidocr_reader()
    base_img = Image.open(io.BytesIO(image_bytes))
    base_img = ImageOps.exif_transpose(base_img).convert("RGB")

    # RapidOCR en Windows ha mostrado abortos duros con rotaciones sucesivas
    # y variantes agresivas; usamos la orientacion base RGB para priorizar estabilidad.
    result, _ = reader(np.array(base_img))
    texts: list[str] = []
    confidences: list[float] = []
    for item in result or []:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        text = str(item[1] or "").strip()
        if not text:
            continue
        texts.append(text)
        if isinstance(item[2], (int, float)):
            confidences.append(float(item[2]))

    page_text = "\n".join(texts).strip()
    return page_text if _score_ocr_candidate(page_text, confidences) > 0 else ""


def _is_ocr_backend_enabled(backend: str) -> bool:
    normalized = _normalize_ocr_backend_name(backend)
    if normalized == "rapidocr":
        return bool(getattr(settings, "enable_rapidocr", True))
    if normalized == "paddleocr":
        return bool(getattr(settings, "enable_paddleocr", True))
    return bool(getattr(settings, "enable_easyocr", True))


def _is_ocr_backend_available(backend: str) -> bool:
    normalized = _normalize_ocr_backend_name(backend)
    module_name = "rapidocr_onnxruntime" if normalized == "rapidocr" else "paddleocr" if normalized == "paddleocr" else "easyocr"
    return importlib.util.find_spec(module_name) is not None


def _select_ocr_backend(preferred: str) -> str:
    normalized = _normalize_ocr_backend_name(preferred)
    candidates = [normalized]
    if normalized != "easyocr":
        candidates.append("easyocr")
    for candidate in candidates:
        if _is_ocr_backend_enabled(candidate) and _is_ocr_backend_available(candidate):
            return candidate
    return normalized


def _get_ocr_reader(backend: str) -> Any:
    normalized = _normalize_ocr_backend_name(backend)
    if normalized == "rapidocr":
        return _get_rapidocr_reader()
    if normalized == "paddleocr":
        return _get_paddleocr_reader()
    return _get_easyocr_reader()


def _ocr_best_text(image_bytes: bytes, backend: str, reader: Any | None = None) -> str:
    normalized = _normalize_ocr_backend_name(backend)
    if normalized == "rapidocr":
        return _rapidocr_best_text(image_bytes, reader=reader)
    if normalized == "paddleocr":
        return _paddleocr_best_text(image_bytes, reader=reader)
    return _easyocr_best_text(image_bytes, reader=reader)


def _use_ocr_page_subprocess(backend: str) -> bool:
    normalized = _normalize_ocr_backend_name(backend)
    if normalized == "rapidocr":
        return bool(getattr(settings, "enable_rapidocr_page_subprocess", True))
    if normalized == "paddleocr":
        return bool(getattr(settings, "enable_paddleocr_page_subprocess", True))
    return bool(getattr(settings, "enable_easyocr_page_subprocess", True))


def _ocr_image_bytes(image_bytes: bytes) -> dict[str, Any]:
    """Extrae texto de una imagen usando OCR local y Qwen2.5-VL como fallback."""
    image_bytes = _downscale_image_bytes(image_bytes)
    texts: list[str] = []
    ocr_backend = _select_ocr_backend(str(getattr(settings, "ocr_backend", "easyocr")))
    use_ocr = _is_ocr_backend_enabled(ocr_backend)
    method = ocr_backend if use_ocr else "ollama_vision"
    ocr_text = ""
    vision_text = ""

    if use_ocr:
        try:
            ocr_text = _ocr_best_text(image_bytes, ocr_backend)
        except ImportError:
            method = "ollama_vision"
        except Exception:
            method = "ollama_vision"

    if use_ocr and _has_meaningful_text(ocr_text):
        texts.append(ocr_text)
    else:
        prompt = (
            "Extrae el texto visible de la imagen. Incluye nombres, fechas, numeros, importes y direcciones. "
            "No agregues comentarios. Devuelve solo el texto extraido."
        )
        try:
            vision_text = chat_with_images(prompt, [image_bytes]).strip()
            method = "ollama_vision"
        except Exception:
            pass

    best = _pick_best_text(ocr_text, vision_text)
    if best:
        texts = [best]
        if best == ocr_text and use_ocr:
            method = ocr_backend
        else:
            method = "ollama_vision"
    text = "\n\n".join(texts).strip()
    page_texts = [text] if text else [""]
    return {
        "text": text,
        "pages": 1,
        "page_texts": page_texts,
        "images": 1,
        "method": method,
    }


def extract_text_from_pdf_bytes(pdf_bytes: bytes, *, force_reading_mode: str | None = None) -> dict[str, Any]:
    """Extrae texto embebido de un PDF (ruta preferida).

    Intenta, según el modo de lectura en uso:
      - auto  → cascada normal: Docling → PyPDF → OCR → VLM fallback.
      - pypdf_only → SOLO PyPDF, sin OCR ni VLM. Devuelve texto embebido aunque sea vacío.
      - ocr_only  → SOLO OCR local (EasyOCR/Paddle/Rapid). NO usa PyPDF, NO usa VLM.
      - vlm_only  → SOLO Qwen2.5-VL directamente sobre páginas, sin OCR ni PyPDF.

    Args:
        pdf_bytes: Contenido del PDF en bytes.
        force_reading_mode: Si se suministra, sobreescribe ENVs y settings.

    Returns:
        Dict con claves como `text`, `pages`, `page_texts` y `method`.
    """
    reading_mode = resolve_forced_reading_mode(force_reading_mode)
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = len(reader.pages)
    except Exception:
        reader = None
        pages = 0

    # ------------------------------------------------------------------
    # MODO 1: PyPDF exclusivamente (ablación para medir baseline nativo)
    # ------------------------------------------------------------------
    if reading_mode == "pypdf_only":
        if reader is None:
            return {"text": "", "pages": 0, "method": "pypdf_only"}
        texts_pdf: list[str] = []
        page_texts_pdf: list[str] = []
        for page in reader.pages:
            page_text = (page.extract_text() or "").strip()
            page_texts_pdf.append(page_text)
            if page_text:
                texts_pdf.append(page_text)
        final_text = "\n\n".join(texts_pdf).strip()
        return {
            "text": final_text,
            "pages": pages,
            "page_texts": page_texts_pdf,
            "method": "pypdf_only",
        }

    # ------------------------------------------------------------------
    # MODO 2: OCR local exclusivo (ablación para medir EasyOCR/Paddle)
    # ------------------------------------------------------------------
    if reading_mode == "ocr_only":
        # Deshabilitamos el fallback de visión (settings temporal) + forzamos scanned_mode
        orig_vision = bool(getattr(settings, "enable_scanned_pdf_vision_fallback", True))
        orig_mode_cfg = str(getattr(settings, "scanned_pdf_text_mode", "auto"))
        try:
            settings.scanned_pdf_text_mode = "ocr_only"
            settings.enable_scanned_pdf_vision_fallback = False
            scanned = extract_text_from_scanned_pdf_bytes(pdf_bytes, force_mode_hint="ocr_only")
            scanned.setdefault("pages", pages)
            scanned["method"] = "ocr_only__" + str(scanned.get("method") or "ocr")
            if "page_texts" not in scanned:
                scanned["page_texts"] = [""] * pages
            return scanned
        finally:
            settings.scanned_pdf_text_mode = orig_mode_cfg
            settings.enable_scanned_pdf_vision_fallback = orig_vision

    # ------------------------------------------------------------------
    # MODO 3: VLM multimodal exclusivo (ablación Qwen2.5-VL directo)
    # ------------------------------------------------------------------
    if reading_mode == "vlm_only":
        try:
            images_vlm = extract_images_from_pdf_bytes(pdf_bytes)
            pages_count = pages or max(1, len(images_vlm))
            if not images_vlm:
                return {
                    "text": "",
                    "pages": pages_count,
                    "page_texts": [""] * pages_count,
                    "method": "vlm_only__no_images",
                }
            page_strings: list[str] = []
            for img in images_vlm:
                try:
                    page_prompt = (
                        "Extrae el texto visible de esta pagina. Incluye nombres completos, DNIs/NIFs, "
                        "fechas completas (dd/mm/aaaa), importes con moneda, porcentajes, domicilios, "
                        "conceptos, numeros de factura/documento. NO agregues comentarios. "
                        "Devuelve SOLO el texto extraido tal cual aparece en la pagina, separando lineas."
                    )
                    page_text = chat_with_images(page_prompt, [img["data"]]).strip()
                    page_strings.append(page_text)
                except Exception:
                    page_strings.append("")
            merged_vlm = "\n\n".join(s for s in page_strings if s).strip()
            return {
                "text": merged_vlm,
                "pages": pages_count,
                "page_texts": page_strings if page_strings else [""],
                "method": "vlm_only",
            }
        except Exception:
            return {"text": "", "pages": pages or 1, "page_texts": [""], "method": "vlm_only__error"}

    # ------------------------------------------------------------------
    # MODO 4: auto (flujo original)
    # ------------------------------------------------------------------
    scanned_mode = str(getattr(settings, "scanned_pdf_text_mode", "auto") or "auto").strip().lower()
    if scanned_mode not in {"auto", "pypdf_only", "ocr_only"}:
        scanned_mode = "auto"

    try:
        from docling.document_converter import DocumentConverter  # type: ignore

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as f:
            f.write(pdf_bytes)
            f.flush()
            converter = DocumentConverter()
            result = converter.convert(f.name)
            md = result.document.export_to_markdown()
        text = (md or "").strip()
        if _has_meaningful_text(text):
            return {"text": text, "pages": pages, "page_texts": [text], "method": "docling"}
    except Exception:
        pass

    if reader is None:
        return {"text": "", "pages": 0, "method": "pypdf"}

    if scanned_mode == "ocr_only":
        scanned_result = extract_text_from_scanned_pdf_bytes(pdf_bytes)
        scanned_result.setdefault("pages", pages)
        if "page_texts" not in scanned_result:
            scanned_result["page_texts"] = [""] * pages
        return scanned_result

    texts: list[str] = []
    page_texts: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        page_text = page_text.strip()
        page_texts.append(page_text)
        if page_text:
            texts.append(page_text)
    text = "\n\n".join(texts).strip()
    if _has_meaningful_text(text):
        return {"text": text, "pages": pages, "page_texts": page_texts, "method": "pypdf"}

    if scanned_mode == "pypdf_only" or not bool(getattr(settings, "enable_scanned_pdf_ocr_fallback", True)):
        return {"text": text, "pages": pages, "page_texts": page_texts, "method": "pypdf"}

    # Si el PDF no trae texto embebido útil, lo tratamos como escaneado.
    scanned_result = extract_text_from_scanned_pdf_bytes(pdf_bytes)
    scanned_text = scanned_result.get("text") or ""
    if _has_meaningful_text(scanned_text) or (not text and scanned_text):
        return scanned_result

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


def _normalize_image_for_subprocess(
    data: bytes,
    *,
    max_dim: int,
    quality: int,
) -> bytes:
    """Normaliza imágenes para workers aislados con tamaños muy conservadores."""
    try:
        from PIL import Image
    except Exception:
        return data

    try:
        with Image.open(io.BytesIO(data)) as im:
            im = im.convert("RGB")
            im.thumbnail((max_dim, max_dim))
            out = io.BytesIO()
            im.save(out, format="JPEG", quality=quality)
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


def _empty_scanned_result(*, pages: int = 0, images: int = 0, method: str = "ollama_vision") -> dict[str, Any]:
    return {
        "text": "",
        "pages": pages,
        "page_texts": [""] * pages,
        "images": images,
        "method": method,
    }


def _extract_text_from_scanned_pdf_bytes_impl(pdf_bytes: bytes) -> dict[str, Any]:
    """Extrae texto desde un PDF escaneado usando EasyOCR (principal, rápido y preciso)
    y Qwen2.5-VL como fallback para casos difíciles.

    Ideal para tesis: EasyOCR es referente académico en OCR local.
    """
    _trace_ocr("impl:start")
    images = extract_images_from_pdf_bytes(pdf_bytes)
    _trace_ocr(f"impl:images={len(images)}")
    pages = len(PdfReader(io.BytesIO(pdf_bytes)).pages)
    _trace_ocr(f"impl:pages={pages}")
    allow_vision_fallback = bool(getattr(settings, "enable_scanned_pdf_vision_fallback", True))
    if not images:
        _trace_ocr("impl:no_images")
        return _empty_scanned_result(pages=pages, images=0, method="ollama_vision")

    page_texts = [""] * pages
    fallback_pages: list[dict[str, Any]] = []
    ocr_backend = _select_ocr_backend(str(getattr(settings, "ocr_backend", "easyocr")))
    use_ocr = _is_ocr_backend_enabled(ocr_backend)

    # -------------------------------
    # PASO 1: OCR local (principal)
    # -------------------------------
    if use_ocr:
        try:
            use_page_subprocess = _use_ocr_page_subprocess(ocr_backend)
            reader = None
            if not use_page_subprocess:
                _trace_ocr(f"impl:before_ocr_reader backend={ocr_backend}")
                reader = _get_ocr_reader(ocr_backend)
                _trace_ocr(f"impl:after_ocr_reader backend={ocr_backend}")

            for img in images:
                try:
                    _trace_ocr(f"impl:ocr_page={img['page']} backend={ocr_backend}")
                    if use_page_subprocess:
                        page_text = _run_ocr_page_subprocess(img["data"], ocr_backend)
                    else:
                        page_text = _ocr_best_text(img["data"], ocr_backend, reader=reader)
                    page_idx = img["page"] - 1

                    if _has_meaningful_text(page_text):
                        if 0 <= page_idx < len(page_texts):
                            page_texts[page_idx] = page_text
                    elif page_text and not allow_vision_fallback:
                        if 0 <= page_idx < len(page_texts):
                            page_texts[page_idx] = page_text
                    else:
                        fallback_pages.append(img)

                except Exception as e:
                    print(f"Error en OCR {ocr_backend} para página {img['page']}: {e}")
                    fallback_pages.append(img)
                    continue

        except ImportError:
            _trace_ocr(f"impl:ocr_import_error backend={ocr_backend}")
            print(f"{ocr_backend} no está instalado, usando fallback Qwen2.5-VL")
            fallback_pages = list(images)
        except Exception as e:
            _trace_ocr(f"impl:ocr_general_error backend={ocr_backend} error={type(e).__name__}")
            print(f"Error general en OCR {ocr_backend}: {e}, usando fallback Qwen2.5-VL")
            fallback_pages = list(images)
    else:
        _trace_ocr(f"impl:ocr_disabled backend={ocr_backend}")
        fallback_pages = list(images)

    # -------------------------------
    # PASO 2: Fallback Qwen2.5-VL solo para páginas problemáticas
    # -------------------------------
    if fallback_pages and allow_vision_fallback:
        _trace_ocr(f"impl:fallback_pages={len(fallback_pages)}")
        print("Usando fallback Qwen2.5-VL para mejor calidad en casos difíciles")
        prompt = "Extrae el texto visible exactamente como aparece. No inventes. Devuelve solo texto."

        for img in fallback_pages:
            try:
                _trace_ocr(f"impl:vision_page={img['page']}")
                if bool(getattr(settings, "enable_vision_page_subprocess", True)):
                    content = _run_vision_page_subprocess(img["data"], prompt)
                else:
                    content = chat_with_images(prompt, [img["data"]])
                content = content.strip()
                if content:
                    page_idx = img["page"] - 1
                    if 0 <= page_idx < len(page_texts):
                        page_texts[page_idx] = _pick_best_text(page_texts[page_idx], content)
            except Exception as e:
                print(f"Error en Qwen2.5-VL para página {img['page']}: {e}")
                continue

        final_method = f"{ocr_backend}+ollama_vision" if any(page_texts) else "ollama_vision"
    elif use_ocr:
        final_method = ocr_backend
    else:
        final_method = "ollama_vision"

    texts = [text for text in page_texts if text]
    _trace_ocr(f"impl:done method={final_method} text_pages={len(texts)}")

    return {
        "text": "\n\n".join(texts).strip(),
        "pages": pages,
        "page_texts": page_texts,
        "images": len(images),
        "method": final_method
    }


def _run_scanned_pdf_ocr_subprocess(pdf_bytes: bytes) -> dict[str, Any]:
    """Ejecuta la OCR de PDF escaneado en un subprocess para aislar abortos duros."""
    timeout_s = max(int(getattr(settings, "scanned_pdf_ocr_timeout_s", 300) or 300), 1)
    pages = len(PdfReader(io.BytesIO(pdf_bytes)).pages)
    images = len(extract_images_from_pdf_bytes(pdf_bytes))
    worker_path = _PROJECT_ROOT / "core" / "document_loader_worker.py"

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as input_tmp:
        input_tmp.write(pdf_bytes)
        input_path = Path(input_tmp.name)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as output_tmp:
        output_path = Path(output_tmp.name)

    env = dict(os.environ)
    env["ENABLE_SCANNED_PDF_OCR_SUBPROCESS"] = "false"

    try:
        completed = subprocess.run(
            [sys.executable, str(worker_path), "--input-pdf", str(input_path), "--output-json", str(output_path)],
            cwd=str(_PROJECT_ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
        )
        if completed.returncode != 0 or not output_path.exists():
            return _empty_scanned_result(pages=pages, images=images, method="ocr_subprocess_failed")
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return _empty_scanned_result(pages=pages, images=images, method="ocr_subprocess_failed")
        return {
            "text": str(payload.get("text") or ""),
            "pages": int(payload.get("pages") or pages),
            "page_texts": list(payload.get("page_texts") or ([""] * pages)),
            "images": int(payload.get("images") or images),
            "method": str(payload.get("method") or "ollama_vision"),
        }
    except subprocess.TimeoutExpired:
        return _empty_scanned_result(pages=pages, images=images, method="ocr_subprocess_timeout")
    except Exception:
        return _empty_scanned_result(pages=pages, images=images, method="ocr_subprocess_failed")
    finally:
        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass


def _run_ocr_page_subprocess(image_bytes: bytes, backend: str) -> str:
    normalized = _normalize_ocr_backend_name(backend)
    if normalized == "rapidocr":
        timeout_s = max(int(getattr(settings, "rapidocr_page_timeout_s", 120) or 120), 1)
        max_dim = int(getattr(settings, "rapidocr_page_max_dim", 512) or 512)
        quality = int(getattr(settings, "rapidocr_page_jpeg_quality", 75) or 75)
    elif normalized == "paddleocr":
        timeout_s = max(int(getattr(settings, "paddleocr_page_timeout_s", 120) or 120), 1)
        max_dim = int(getattr(settings, "paddleocr_page_max_dim", 384) or 384)
        quality = int(getattr(settings, "paddleocr_page_jpeg_quality", 70) or 70)
    else:
        timeout_s = max(int(getattr(settings, "easyocr_page_timeout_s", 120) or 120), 1)
        max_dim = int(getattr(settings, "easyocr_page_max_dim", 256) or 256)
        quality = int(getattr(settings, "easyocr_page_jpeg_quality", 60) or 60)
    worker_path = _PROJECT_ROOT / "core" / "ocr_page_worker.py"
    max_dim = max(96, min(1536, max_dim))
    quality = max(30, min(95, quality))
    image_bytes = _normalize_image_for_subprocess(image_bytes, max_dim=max_dim, quality=quality)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as input_tmp:
        input_tmp.write(image_bytes)
        input_path = Path(input_tmp.name)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as output_tmp:
        output_path = Path(output_tmp.name)

    env = dict(os.environ)
    try:
        _trace_ocr(f"page_subprocess:before_run backend={normalized}")
        completed = subprocess.run(
            [
                sys.executable,
                str(worker_path),
                "--backend",
                normalized,
                "--input-image",
                str(input_path),
                "--output-json",
                str(output_path),
            ],
            cwd=str(_PROJECT_ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
        )
        _trace_ocr(f"page_subprocess:after_run backend={normalized} rc={completed.returncode} exists={output_path.exists()}")
        if completed.returncode != 0 or not output_path.exists():
            return ""
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        _trace_ocr("page_subprocess:after_read_json")
        if not isinstance(payload, dict):
            return ""
        return str(payload.get("text") or "")
    except subprocess.TimeoutExpired:
        _trace_ocr(f"page_subprocess:timeout backend={normalized}")
        return ""
    except Exception as exc:
        _trace_ocr(f"page_subprocess:error backend={normalized} error={type(exc).__name__}")
        return ""
    finally:
        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass


def _run_easyocr_page_subprocess(image_bytes: bytes) -> str:
    return _run_ocr_page_subprocess(image_bytes, "easyocr")


def _run_vision_page_subprocess(image_bytes: bytes, prompt: str) -> str:
    timeout_s = max(int(getattr(settings, "vision_page_timeout_s", 180) or 180), 1)
    worker_path = _PROJECT_ROOT / "core" / "vision_page_worker.py"
    max_dim = int(getattr(settings, "vision_page_max_dim", 256) or 256)
    quality = int(getattr(settings, "vision_page_jpeg_quality", 60) or 60)
    max_dim = max(128, min(1536, max_dim))
    quality = max(30, min(95, quality))
    image_bytes = _normalize_image_for_subprocess(image_bytes, max_dim=max_dim, quality=quality)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as input_tmp:
        input_tmp.write(image_bytes)
        input_path = Path(input_tmp.name)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as output_tmp:
        output_path = Path(output_tmp.name)

    env = dict(os.environ)
    env.setdefault("OLLAMA_BASE_URL", str(getattr(settings, "ollama_base_url", "http://localhost:11434")))
    env.setdefault("OLLAMA_VISION_MODEL", str(getattr(settings, "ollama_vision_model", "qwen2.5vl:7b")))
    env.setdefault("OLLAMA_TIMEOUT_S", str(timeout_s))
    env.setdefault("OLLAMA_VISION_NUM_PREDICT", str(int(getattr(settings, "ollama_vision_num_predict", 256) or 256)))
    try:
        _trace_ocr("vision_subprocess:before_run")
        completed = subprocess.run(
            [
                sys.executable,
                str(worker_path),
                "--input-image",
                str(input_path),
                "--output-json",
                str(output_path),
                "--prompt",
                prompt,
            ],
            cwd=str(_PROJECT_ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
        )
        _trace_ocr(f"vision_subprocess:after_run rc={completed.returncode} exists={output_path.exists()}")
        if completed.returncode != 0 or not output_path.exists():
            return ""
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return ""
        return str(payload.get("text") or "")
    except subprocess.TimeoutExpired:
        _trace_ocr("vision_subprocess:timeout")
        return ""
    except Exception as exc:
        _trace_ocr(f"vision_subprocess:error={type(exc).__name__}")
        return ""
    finally:
        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass


def extract_text_from_scanned_pdf_bytes(pdf_bytes: bytes, *, force_mode_hint: str = "auto") -> dict[str, Any]:
    # El parámetro force_mode_hint controla si el modo de lectura subyacente puede
    # emplear fallback VLM. Valores: auto | ocr_only.
    # Si force_mode_hint == "ocr_only" → nunca llama a VLM multimodal.
    ocr_backend = _select_ocr_backend(str(getattr(settings, "ocr_backend", "easyocr")))
    if force_mode_hint == "ocr_only":
        orig_vision = bool(getattr(settings, "enable_scanned_pdf_vision_fallback", True))
        try:
            settings.enable_scanned_pdf_vision_fallback = False
            if _use_ocr_page_subprocess(ocr_backend):
                return _extract_text_from_scanned_pdf_bytes_impl(pdf_bytes)
            if bool(getattr(settings, "enable_scanned_pdf_ocr_subprocess", True)):
                return _run_scanned_pdf_ocr_subprocess(pdf_bytes)
            return _extract_text_from_scanned_pdf_bytes_impl(pdf_bytes)
        finally:
            settings.enable_scanned_pdf_vision_fallback = orig_vision
    # Priorizamos el aislamiento por página para evitar subprocess anidados
    # (padre -> worker PDF -> worker página), que en entornos restringidos
    # puede ser menos estable que orquestar las páginas desde el proceso padre.
    if _use_ocr_page_subprocess(ocr_backend):
        return _extract_text_from_scanned_pdf_bytes_impl(pdf_bytes)
    if bool(getattr(settings, "enable_scanned_pdf_ocr_subprocess", True)):
        return _run_scanned_pdf_ocr_subprocess(pdf_bytes)
    return _extract_text_from_scanned_pdf_bytes_impl(pdf_bytes)


def extract_text_from_image_bytes(image_bytes: bytes, *, force_reading_mode: str | None = None) -> dict[str, Any]:
    """Extrae texto desde una imagen suelta con soporte de modo forzado.

    force_reading_mode:
      - auto      → comportamiento original (OCR + VLM fallback).
      - ocr_only  → solo OCR local, sin VLM multimodal.
      - vlm_only  → solo Qwen2.5-VL, sin OCR.
      - pypdf_only → no tiene sentido en imagen, se trata como ocr_only.
    """
    mode = resolve_forced_reading_mode(force_reading_mode)
    image_bytes_forced = _downscale_image_bytes(image_bytes)

    if mode == "vlm_only":
        try:
            prompt_vlm = (
                "Extrae el texto visible de esta imagen. Incluye nombres, DNIs/NIFs, fechas completas, "
                "importes con moneda, porcentajes, domicilios, referencias y numeros. "
                "No agregues comentarios. Devuelve solo el texto extraido."
            )
            vision_text = chat_with_images(prompt_vlm, [image_bytes_forced]).strip()
            return {
                "text": vision_text,
                "pages": 1,
                "page_texts": [vision_text] if vision_text else [""],
                "images": 1,
                "method": "vlm_only",
            }
        except Exception:
            return {"text": "", "pages": 1, "page_texts": [""], "images": 1, "method": "vlm_only__error"}

    if mode in {"ocr_only", "pypdf_only"}:
        # Modo OCR estricto: sin fallback a VLM multimodal aunque el texto sea pobre.
        ocr_backend_forced = _select_ocr_backend(str(getattr(settings, "ocr_backend", "easyocr")))
        ocr_reader = None if _use_ocr_page_subprocess(ocr_backend_forced) else _get_ocr_reader(ocr_backend_forced)
        ocr_text = ""
        method = f"ocr_only__{ocr_backend_forced}"
        try:
            ocr_text = _ocr_best_text(image_bytes_forced, ocr_backend_forced, reader=ocr_reader)
        except Exception:
            pass
        return {
            "text": ocr_text,
            "pages": 1,
            "page_texts": [ocr_text] if ocr_text else [""],
            "images": 1,
            "method": method,
        }

    return _ocr_image_bytes(image_bytes)
