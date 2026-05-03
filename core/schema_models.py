from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


PrimitiveType = Literal["string", "number", "integer", "boolean", "date", "datetime"]
RuleSeverity = Literal["critica", "advertencia", "info"]


class FieldRule(BaseModel):
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)


class SchemaField(BaseModel):
    name: str
    type: PrimitiveType
    required: bool = False
    description: str | None = None
    document_type: str | None = None
    examples: list[Any] = Field(default_factory=list)
    rules: list[FieldRule] = Field(default_factory=list)


class DecisionRule(BaseModel):
    id: str | None = None
    descripcion: str
    expresion: str
    severidad: RuleSeverity = "advertencia"
    normativa: str | None = None


class ReportConfig(BaseModel):
    formato: list[str] = Field(default_factory=lambda: ["json"])
    incluir_evidencias: bool = False
    incluir_score_confianza: bool = True
    nivel_detalle: str = "resumen"
    idioma: str = "es"


class DocSchema(BaseModel):
    name: str
    version: str
    domain: str | None = None
    document_types: list[str] = Field(default_factory=list)
    fields: list[SchemaField]
    decision_rules: list[DecisionRule] = Field(default_factory=list)
    report: ReportConfig | None = None
