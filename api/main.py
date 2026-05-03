from typing import Any

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field

from core.document_loader import extract_text_from_pdf_bytes, extract_text_from_scanned_pdf_bytes
from core.orchestrator import run_pipeline


app = FastAPI(title="DocAudit Agent")


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1)


class ExtractResponse(BaseModel):
    schema: dict[str, Any]
    extracted_raw: dict[str, Any]
    extracted: dict[str, Any]
    normalization: dict[str, Any]
    validation: dict[str, Any]
    report: dict[str, Any]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest) -> ExtractResponse:
    result = run_pipeline(req.text)
    return ExtractResponse(**result)


@app.post("/extract_pdf", response_model=ExtractResponse)
async def extract_pdf(file: UploadFile = File(...), mode: str = "auto") -> ExtractResponse:
    pdf_bytes = await file.read()
    extracted = extract_text_from_pdf_bytes(pdf_bytes)
    text = extracted["text"]
    if mode in {"auto", "vision"} and not text:
        extracted_v = extract_text_from_scanned_pdf_bytes(pdf_bytes)
        text = extracted_v["text"]
    result = run_pipeline(text)
    return ExtractResponse(**result)
