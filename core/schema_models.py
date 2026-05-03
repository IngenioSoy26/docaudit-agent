from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


PrimitiveType = Literal["string", "number", "integer", "boolean", "date", "datetime"]


class FieldRule(BaseModel):
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)


class SchemaField(BaseModel):
    name: str
    type: PrimitiveType
    required: bool = False
    description: str | None = None
    examples: list[Any] = Field(default_factory=list)
    rules: list[FieldRule] = Field(default_factory=list)


class DocSchema(BaseModel):
    name: str
    version: str
    domain: str | None = None
    document_types: list[str] = Field(default_factory=list)
    fields: list[SchemaField]
