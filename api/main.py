"""
API HTTP (FastAPI) para ejecutar DocAudit Agent.

Endpoints:
- GET /health: healthcheck
- POST /extract: ejecuta pipeline sobre texto
- POST /extract_pdf: recibe PDF y ejecuta extracción de texto (auto/vision) + pipeline

Compatibilidad con la Propuesta Técnica:
- POST /upload
- POST /process
- GET /report/{doc_id}
"""

from typing import Any

import hashlib
import time
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from core.document_loader import extract_text_from_pdf_bytes, extract_text_from_scanned_pdf_bytes
from core.orchestrator import run_pipeline


app = FastAPI(title="DocAudit Agent")

_UPLOAD_CACHE: dict[str, dict[str, Any]] = {}
_REPORT_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_TTL_S = 60 * 30


def _cleanup_cache() -> None:
    now = time.time()
    for store in (_UPLOAD_CACHE, _REPORT_CACHE):
        expired = [k for k, v in store.items() if (now - float(v.get("_ts") or now)) > _CACHE_TTL_S]
        for k in expired:
            store.pop(k, None)



class ExtractRequest(BaseModel):
    """Payload de entrada para extracción por texto."""
    text: str = Field(..., min_length=1, description="Texto del documento (en bruto).")



class ExtractResponse(BaseModel):
    """Respuesta estándar del pipeline (lista para serializar a JSON)."""

    model_config = ConfigDict(populate_by_name=True)

    schema_: dict[str, Any] = Field(..., alias="schema")
    extracted_raw: dict[str, Any]
    extracted: dict[str, Any]
    normalization: dict[str, Any]
    validation: dict[str, Any]
    report: dict[str, Any]


class UploadResponse(BaseModel):
    doc_id: str
    pages: int | None = None
    method: str | None = None


class ProcessRequest(BaseModel):
    doc_id: str | None = None
    text: str | None = None
    mode: str = "auto"



@app.get("/health")
def health() -> dict[str, str]:
    """Healthcheck básico para verificar que la API está levantada."""
    _cleanup_cache()
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest) -> ExtractResponse:
    """Ejecuta el pipeline sobre un texto ya extraído de un documento."""
    result = run_pipeline(req.text)
    _cleanup_cache()
    return ExtractResponse(**result)


@app.post("/extract_pdf", response_model=ExtractResponse)
async def extract_pdf(file: UploadFile = File(...), mode: str = "auto") -> ExtractResponse:
    """Ejecuta extracción desde PDF (texto o visión) y luego el pipeline.

    Args:
        file: PDF a procesar.
        mode: "auto" (texto; si vacío, visión) o "vision" (forzar visión si no hay texto).
    """
    pdf_bytes = await file.read()
    doc_id = hashlib.sha256(pdf_bytes).hexdigest()
    extracted = extract_text_from_pdf_bytes(pdf_bytes)
    text = extracted["text"]
    pages = extracted.get("page_texts")
    if mode in {"auto", "vision"} and not text:
        extracted_v = extract_text_from_scanned_pdf_bytes(pdf_bytes)
        text = extracted_v["text"]
        pages = extracted_v.get("page_texts")
    result = run_pipeline(text, pages=pages if isinstance(pages, list) else None, doc_id=doc_id)
    _cleanup_cache()
    _REPORT_CACHE[doc_id] = {"_ts": time.time(), "result": result}
    return ExtractResponse(**result)


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...), mode: str = "auto") -> UploadResponse:
    """Carga un PDF y deja su contenido preparado para /process y /report."""
    pdf_bytes = await file.read()
    doc_id = hashlib.sha256(pdf_bytes).hexdigest()
    extracted = extract_text_from_pdf_bytes(pdf_bytes)
    text = extracted.get("text") or ""
    pages = extracted.get("page_texts")
    if mode in {"auto", "vision"} and not text:
        extracted_v = extract_text_from_scanned_pdf_bytes(pdf_bytes)
        text = extracted_v.get("text") or ""
        pages = extracted_v.get("page_texts")
        extracted = {"pages": extracted_v.get("pages"), "method": "vision"}
    _cleanup_cache()
    _UPLOAD_CACHE[doc_id] = {
        "_ts": time.time(),
        "text": text,
        "pages": pages if isinstance(pages, list) else None,
        "pages_n": extracted.get("pages"),
        "method": extracted.get("method"),
    }
    return UploadResponse(doc_id=doc_id, pages=extracted.get("pages"), method=extracted.get("method"))


@app.post("/process", response_model=ExtractResponse)
def process(req: ProcessRequest) -> ExtractResponse:
    """Procesa un documento previamente subido (/upload) o un texto provisto en la petición."""
    _cleanup_cache()
    text = req.text
    pages = None
    doc_id = req.doc_id
    if doc_id and doc_id in _UPLOAD_CACHE:
        cached = _UPLOAD_CACHE[doc_id]
        text = cached.get("text") or ""
        pages = cached.get("pages")
    if not text or not isinstance(text, str) or not text.strip():
        raise HTTPException(
            status_code=400,
            detail="Debe proporcionar `text` o un `doc_id` válido previamente subido en /upload.",
        )
    result = run_pipeline(text, pages=pages if isinstance(pages, list) else None, doc_id=doc_id)
    if doc_id:
        _REPORT_CACHE[doc_id] = {"_ts": time.time(), "result": result}
    return ExtractResponse(**result)


@app.get("/report/{doc_id}", response_model=ExtractResponse)
def report(doc_id: str) -> ExtractResponse:
    """Devuelve el último resultado del pipeline para un doc_id (si existe en cache)."""
    _cleanup_cache()
    cached = _REPORT_CACHE.get(doc_id)
    if not cached:
        raise HTTPException(
            status_code=404,
            detail="No existe reporte en cache para este doc_id. Ejecute /process primero.",
        )
    return ExtractResponse(**(cached.get("result") or {}))
