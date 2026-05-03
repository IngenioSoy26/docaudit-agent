from typing import Any

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field

from core.document_loader import extract_text_from_pdf_bytes
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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest) -> ExtractResponse:
    result = run_pipeline(req.text)
    return ExtractResponse(**result)


@app.post("/extract_pdf", response_model=ExtractResponse)
async def extract_pdf(file: UploadFile = File(...)) -> ExtractResponse:
    pdf_bytes = await file.read()
    extracted = extract_text_from_pdf_bytes(pdf_bytes)
    result = run_pipeline(extracted["text"])
    return ExtractResponse(**result)
