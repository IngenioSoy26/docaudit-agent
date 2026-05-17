from __future__ import annotations

"""
Carga de esquemas YAML.

Soporta:
- Formato simple (MVP): {name, version, fields, ...}
- Formato extendido (empresa): {caso_uso, documentos:[{tipo, campos:[...]}], reglas_decision, informe}

Además, tolera codificaciones típicas en Windows (cp1252/latin-1) en YAMLs externos.
"""

from pathlib import Path

import yaml

from core.schema_models import DecisionRule, DocSchema, FieldRule, ReportConfig, SchemaField


def load_schema(schema_path: str | Path) -> DocSchema:
    """Carga un esquema desde disco y lo valida contra los modelos Pydantic."""
    path = Path(schema_path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        data = path.read_bytes()
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                text = data.decode(enc)
                break
            except Exception:
                text = ""
        if not text:
            raise
    raw = yaml.safe_load(text)
    if isinstance(raw, dict) and "caso_uso" in raw:
        return _load_new_format(raw)
    return DocSchema.model_validate(raw)


def _map_tipo_dato(tipo_dato: str) -> str:
    t = (tipo_dato or "").strip().lower()
    if t in {"float", "number", "decimal"}:
        return "number"
    if t in {"int", "integer"}:
        return "integer"
    if t in {"bool", "boolean"}:
        return "boolean"
    if t in {"date", "fecha"}:
        return "date"
    if t in {"datetime", "timestamp"}:
        return "datetime"
    return "string"


def _field_from_campo(campo: dict, *, document_type: str | None = None) -> SchemaField:
    name = campo.get("nombre")
    field_type = _map_tipo_dato(campo.get("tipo_dato", "string"))
    required = bool(campo.get("requerido", False))
    description = campo.get("etiqueta") or campo.get("descripcion")

    rules: list[FieldRule] = []
    patron = campo.get("patron")
    if isinstance(patron, str) and patron.strip():
        rules.append(FieldRule(kind="regex", params={"pattern": patron}))

    validacion = campo.get("validacion") or {}
    if isinstance(validacion, dict):
        if "rango_min" in validacion and validacion["rango_min"] is not None:
            rules.append(FieldRule(kind="min", params={"value": validacion["rango_min"]}))
        if "rango_max" in validacion and validacion["rango_max"] is not None:
            rules.append(FieldRule(kind="max", params={"value": validacion["rango_max"]}))
        if "valores_permitidos" in validacion and isinstance(validacion["valores_permitidos"], list):
            rules.append(FieldRule(kind="enum", params={"values": validacion["valores_permitidos"]}))

    formato = campo.get("formato")
    if isinstance(formato, str) and formato.strip():
        rules.append(FieldRule(kind="format", params={"value": formato}))

    return SchemaField(
        name=str(name),
        type=field_type,  # type: ignore[arg-type]
        required=required,
        description=description,
        document_type=document_type,
        rules=rules,
    )


def _load_new_format(raw: dict) -> DocSchema:
    name = raw.get("caso_uso") or raw.get("name")
    version = raw.get("version") or "1.0"
    domain = raw.get("descripcion")

    document_types: list[str] = []
    fields: list[SchemaField] = []

    documentos = raw.get("documentos") or []
    if isinstance(documentos, dict):
        documentos = [documentos]
    if isinstance(documentos, list):
        for doc in documentos:
            if not isinstance(doc, dict):
                continue
            doc_type = doc.get("tipo")
            if isinstance(doc_type, str) and doc_type.strip():
                document_types.append(doc_type.strip())
            campos = doc.get("campos") or []
            if isinstance(campos, list):
                for campo in campos:
                    if isinstance(campo, dict):
                        fields.append(_field_from_campo(campo, document_type=str(doc_type) if doc_type else None))

    decision_rules: list[DecisionRule] = []
    reglas = raw.get("reglas_decision") or []
    if isinstance(reglas, list):
        for r in reglas:
            if isinstance(r, dict) and r.get("descripcion") and r.get("expresion"):
                decision_rules.append(DecisionRule(**r))

    report = None
    informe = raw.get("informe")
    if isinstance(informe, dict):
        report = ReportConfig(**informe)

    return DocSchema(
        name=str(name),
        version=str(version),
        domain=str(domain) if isinstance(domain, str) else None,
        document_types=document_types,
        fields=fields,
        decision_rules=decision_rules,
        report=report,
    )
