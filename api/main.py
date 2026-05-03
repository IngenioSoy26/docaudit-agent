from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from core.orchestrator import run_pipeline


app = FastAPI(title="DocAudit Agent")


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1)


class ExtractResponse(BaseModel):
    schema: dict[str, Any]
    extracted: dict[str, Any]
    validation: dict[str, Any]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest) -> ExtractResponse:
    result = run_pipeline(req.text)
    return ExtractResponse(**result)
