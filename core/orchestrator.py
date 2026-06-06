from __future__ import annotations

"""
Orquestación del pipeline DocAudit Agent.

Este módulo define dos modos de ejecución:
- run_pipeline: procesa un único texto/PDF ya convertido a texto.
- run_expediente: procesa múltiples documentos (expediente), fusiona campos y evalúa reglas con contexto agregado.

El grafo principal está implementado con LangGraph como un pipeline dirigido con estado compartido:
clasificador → extractor → normalizador → validador → auditor.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

from agents.auditor import audit_document
from agents.classifier import classify_text
from agents.extractor import extract_from_text
from core.normalizer import normalize_extracted
from core.privacy import redact_pii
from core.schema_loader import load_schema
from core.schema_models import DocSchema
from core.settings import settings
from core.validator import validate_extracted


class PipelineState(TypedDict, total=False):
    """Estado compartido entre nodos del grafo.

    Cada nodo del pipeline añade/actualiza claves en este estado. El uso de TypedDict
    facilita trazabilidad y reduce errores de integración entre agentes.
    """

    text: str
    pages: list[str]
    doc_id: str
    schemas_dir: str
    schema_name: str
    schema: DocSchema
    extracted_raw: dict[str, Any]
    normalization: dict[str, Any]
    extracted: dict[str, Any]
    field_details: dict[str, Any]
    validation: dict[str, Any]
    report: dict[str, Any]


def _infer_document_type(schema: DocSchema, text: str) -> str | None:
    """Infere el tipo documental más probable dentro de un esquema multi-documento.

    Esta función permite filtrar campos cuando el esquema define `document_types` y
    cada campo incluye `document_type`. Se usan heurísticas rápidas por palabras clave.

    Args:
        schema: Esquema con posibles tipos documentales.
        text: Texto del documento.

    Returns:
        El tipo documental inferido o None si no aplica.
    """
    if not schema.document_types:
        return None
    t = (text or "").lower()
    candidates = schema.document_types

    def _pick(*, triggers: list[str], type_hints: list[str]) -> str | None:
        if not any(x in t for x in triggers):
            return None
        for dt in candidates:
            low = dt.lower()
            if any(h in low for h in type_hints):
                return dt
        return None

    hit = _pick(
        triggers=[
            "escritura",
            "préstamo hipotecario",
            "prestamo hipotecario",
            "capital del préstamo",
            "capital del prestamo",
            "tipo de interés nominal",
            "tipo de interes nominal",
            "tin",
            "plazo de amortización",
            "plazo de amortizacion",
            "cláusula",
            "clausula",
        ],
        type_hints=["escritura", "hipotec"],
    )
    if hit:
        return hit
    hit = _pick(
        triggers=["cirbe", "incidencias", "riesgos", "deuda vigente", "deuda_vigente"],
        type_hints=["cirbe"],
    )
    if hit:
        return hit
    hit = _pick(
        triggers=["iban", "saldo", "transferencia", "movimiento", "extracto bancario"],
        type_hints=["extracto", "bancario"],
    )
    if hit:
        return hit
    hit = _pick(
        triggers=["factura", "iva", "base imponible", "cuota"],
        type_hints=["factura"],
    )
    if hit:
        return hit
    hit = _pick(
        triggers=["irpf", "casilla", "declaración", "declaracion", "modelo 100"],
        type_hints=["irpf"],
    )
    if hit:
        return hit
    return None


@lru_cache(maxsize=8)
def _build_graph() -> Any:
    """Construye y compila el grafo LangGraph del pipeline.

    Se cachea para evitar reconstruir el grafo en cada ejecución.
    """
    from langgraph.graph import END, StateGraph

    graph: StateGraph[PipelineState] = StateGraph(PipelineState)

    def node_classify(state: PipelineState) -> PipelineState:
        """Selecciona el schema_name a partir del texto."""
        existing = state.get("schema_name")
        if isinstance(existing, str) and existing.strip():
            return {}
        return {"schema_name": classify_text(state["text"])}

    def node_extract(state: PipelineState) -> PipelineState:
        """Carga esquema, filtra por tipo documental (si aplica) y extrae campos con el LLM."""
        schemas_dir = Path(state.get("schemas_dir") or "schemas")
        schema_name = state["schema_name"]
        schema_path = schemas_dir / f"{schema_name}.yaml"
        schema = load_schema(schema_path)
        inferred = _infer_document_type(schema, state["text"])
        if inferred and any(getattr(f, "document_type", None) for f in schema.fields):
            # Si el esquema está segmentado por tipo, extrae solo los campos del tipo inferido.
            filtered_fields = [f for f in schema.fields if f.document_type == inferred]
            if filtered_fields:
                schema = schema.model_copy(
                    update={"fields": filtered_fields, "document_types": [inferred]}
                )
        extracted_raw = extract_from_text(
            state["text"],
            schema,
            pages=state.get("pages"),
            doc_id=state.get("doc_id"),
        )
        if isinstance(extracted_raw, dict) and "fields" in extracted_raw and "details" in extracted_raw:
            extracted_fields = extracted_raw.get("fields") or {}
            return {
                "schema": schema,
                "extracted_raw": extracted_raw,
                "extracted": extracted_fields,
                "field_details": extracted_raw.get("details") or {},
            }
        return {"schema": schema, "extracted_raw": extracted_raw, "extracted": extracted_raw}

    def node_normalize(state: PipelineState) -> PipelineState:
        """Normaliza formatos (números/fechas/booleans) antes de validar."""
        schema = state["schema"]
        normalization = normalize_extracted(state["extracted"], schema)
        return {"normalization": normalization, "extracted": normalization["normalized"]}

    def node_validate(state: PipelineState) -> PipelineState:
        """Valida campos normalizados según reglas declarativas del esquema."""
        schema = state["schema"]
        validation = validate_extracted(state["extracted"], schema)
        return {"validation": validation}

    def node_audit(state: PipelineState) -> PipelineState:
        """Evalúa reglas de decisión y compone el informe final."""
        schema = state["schema"]
        report = audit_document(
            schema,
            state["extracted"],
            state["validation"],
            field_details=state.get("field_details") or {},
        )
        return {"report": report}

    graph.add_node("classifier", node_classify)
    graph.add_node("extractor", node_extract)
    graph.add_node("normalizer", node_normalize)
    graph.add_node("validator", node_validate)
    graph.add_node("auditor", node_audit)

    graph.set_entry_point("classifier")
    graph.add_edge("classifier", "extractor")
    graph.add_edge("extractor", "normalizer")
    graph.add_edge("normalizer", "validator")
    graph.add_edge("validator", "auditor")
    graph.add_edge("auditor", END)

    return graph.compile()


def run_pipeline(
    text: str,
    schemas_dir: str | Path = "schemas",
    pages: list[str] | None = None,
    doc_id: str | None = None,
    schema_name: str | None = None,
) -> dict[str, Any]:
    """Ejecuta el pipeline completo sobre un único documento (texto).

    Args:
        text: Texto del documento (ya extraído del PDF).
        schemas_dir: Directorio donde residen los YAML.
        pages: Texto por página si está disponible (mejora evidencias RAG).
        doc_id: Identificador del documento para persistencia de RAG.
        schema_name: Si se indica, fuerza el esquema y omite la clasificación automática.

    Returns:
        Resultado completo del pipeline (extracción, normalización, validación, auditoría).
    """
    if settings.enable_pii_redaction:
        text, _ = redact_pii(text)
        if pages:
            pages = [redact_pii(p)[0] for p in pages]

    app = _build_graph()
    state: PipelineState = {"text": text, "schemas_dir": str(schemas_dir)}
    if pages:
        state["pages"] = pages
    if doc_id:
        state["doc_id"] = doc_id
    if schema_name:
        state["schema_name"] = schema_name
    final_state: PipelineState = app.invoke(state)
    schema = final_state["schema"]
    normalization = final_state["normalization"]
    return {
        "schema": {"name": schema.name, "version": schema.version},
        "extracted_raw": final_state["extracted_raw"],
        "extracted": final_state["extracted"],
        "normalization": normalization,
        "validation": final_state["validation"],
        "report": final_state.get("report", {}),
    }


def _coerce_extraction_payload(payload: Any) -> tuple[dict[str, Any], dict[str, Any], Any]:
    """Normaliza la salida del extractor a una forma consistente (fields/details).

    El extractor puede devolver:
    - {"fields": {...}, "details": {...}} (formato actual)
    - {"campo": valor, ...} (formato legacy)
    """
    if isinstance(payload, dict) and "fields" in payload and "details" in payload:
        fields = payload.get("fields") or {}
        details = payload.get("details") or {}
        if isinstance(fields, dict) and isinstance(details, dict):
            return fields, details, payload
    if isinstance(payload, dict):
        return payload, {}, payload
    return {}, {}, payload


def _choose_schema_for_expediente(texts: list[str]) -> str:
    """Elige el esquema para expediente usando señales simples y fallback a clasificador."""
    combined = "\n".join(t for t in texts if isinstance(t, str) and t.strip())
    t = combined.lower()
    hipotecario_signals = [
        "hipoteca",
        "hipotecario",
        "irpf",
        "modelo 100",
        "cirbe",
        "incidencias",
        "nota simple",
        "registro de la propiedad",
        "préstamo",
        "prestamo",
    ]
    if any(s in t for s in hipotecario_signals):
        return "credito_hipotecario"
    return classify_text(combined)


def _pick_better_detail(existing: dict[str, Any] | None, incoming: dict[str, Any] | None) -> dict[str, Any] | None:
    """Selecciona el mejor detalle de un campo basándose en confianza (si existe)."""
    if not isinstance(incoming, dict):
        return existing
    if not isinstance(existing, dict):
        return incoming
    ec = existing.get("confianza")
    ic = incoming.get("confianza")
    if isinstance(ic, (int, float)) and isinstance(ec, (int, float)):
        return incoming if float(ic) > float(ec) else existing
    if isinstance(ic, (int, float)) and ec is None:
        return incoming
    return existing


def run_expediente(
    texts: list[str],
    schemas_dir: str | Path = "schemas",
    pages_by_doc: list[list[str] | None] | None = None,
    doc_ids: list[str | None] | None = None,
    schema_name: str | None = None,
) -> dict[str, Any]:
    """Ejecuta el pipeline en modo expediente (multi-documento) y fusiona resultados.

    Args:
        texts: Lista de textos (uno por documento).
        schemas_dir: Directorio de esquemas YAML.
        pages_by_doc: Lista paralela de páginas por documento.
        doc_ids: Lista paralela de doc_id por documento (para RAG persistente).
        schema_name: Si se indica, fuerza el esquema. Si no, se infiere automáticamente.

    Returns:
        Resultado agregado: documentos procesados + extracción fusionada + auditoría.

    Raises:
        ValueError: Si `texts` está vacío.
    """
    if not texts:
        raise ValueError("texts no puede estar vacío")

    schemas_dir_path = Path(schemas_dir)
    chosen_schema = schema_name or _choose_schema_for_expediente(texts)
    schema_path = schemas_dir_path / f"{chosen_schema}.yaml"
    schema = load_schema(schema_path)

    per_doc: list[dict[str, Any]] = []
    merged_fields: dict[str, Any] = {}
    merged_details: dict[str, Any] = {}
    present_types: list[str] = []

    for i, text in enumerate(texts):
        if settings.enable_pii_redaction:
            text, _ = redact_pii(text)
        pages = pages_by_doc[i] if isinstance(pages_by_doc, list) and i < len(pages_by_doc) else None
        if settings.enable_pii_redaction and isinstance(pages, list):
            pages = [redact_pii(p)[0] for p in pages]
        doc_id = doc_ids[i] if isinstance(doc_ids, list) and i < len(doc_ids) else None

        inferred = _infer_document_type(schema, text)
        if isinstance(inferred, str) and inferred and inferred not in present_types:
            present_types.append(inferred)
        doc_schema = schema
        if inferred and any(getattr(f, "document_type", None) for f in schema.fields):
            # Filtra los campos para evitar pedir al LLM información de otros tipos documentales.
            filtered_fields = [f for f in schema.fields if f.document_type == inferred]
            if filtered_fields:
                doc_schema = schema.model_copy(
                    update={"fields": filtered_fields, "document_types": [inferred]}
                )

        extracted_payload = extract_from_text(text, doc_schema, pages=pages, doc_id=doc_id)
        fields, details, extracted_raw = _coerce_extraction_payload(extracted_payload)

        doc_normalization = normalize_extracted(fields, doc_schema)
        doc_normalized = doc_normalization["normalized"]
        doc_validation = validate_extracted(doc_normalized, doc_schema)
        doc_report = audit_document(doc_schema, doc_normalized, doc_validation, field_details=details)

        # Fusión de campos: prioriza valores no nulos; resuelve conflictos con confianza si existe.
        for k, v in fields.items():
            if v is None and k in merged_fields and merged_fields[k] is not None:
                continue
            if k not in merged_fields or merged_fields[k] is None:
                merged_fields[k] = v
                if k in details:
                    merged_details[k] = details.get(k)
                continue
            if k in details and isinstance(details.get(k), dict):
                best = _pick_better_detail(
                    merged_details.get(k) if isinstance(merged_details.get(k), dict) else None,
                    details.get(k),
                )
                if best is details.get(k):
                    merged_fields[k] = v
                    merged_details[k] = best

        per_doc.append(
            {
                "doc_index": i,
                "doc_id": doc_id,
                "document_type": inferred,
                "schema": {
                    "name": doc_schema.name,
                    "version": doc_schema.version,
                    "document_types": doc_schema.document_types,
                },
                "extracted_raw": extracted_raw,
                "extracted": fields,
                "normalization": doc_normalization,
                "validation": doc_validation,
                "report": doc_report,
            }
        )

    expediente_schema = schema
    if present_types and any(getattr(f, "document_type", None) for f in schema.fields):
        # Schema efectivo: evita requeridos de tipos documentales no presentes en el expediente.
        expediente_fields = [f for f in schema.fields if f.document_type in present_types or f.document_type is None]
        if expediente_fields:
            expediente_schema = schema.model_copy(update={"fields": expediente_fields, "document_types": present_types})

    for f in expediente_schema.fields:
        merged_fields.setdefault(f.name, None)
        merged_details.setdefault(
            f.name,
            {
                "nombre": f.name,
                "valor": merged_fields.get(f.name),
                "confianza": None,
                "evidencia_textual": "",
                "pagina": 1,
            },
        )

    normalization = normalize_extracted(merged_fields, expediente_schema)
    normalized = normalization["normalized"]
    validation = validate_extracted(normalized, expediente_schema)
    report = audit_document(expediente_schema, normalized, validation, field_details=merged_details)

    return {
        "schema": {"name": expediente_schema.name, "version": expediente_schema.version},
        "documents": per_doc,
        "extracted_raw": {"merged": merged_fields, "per_document": per_doc},
        "extracted": normalized,
        "normalization": normalization,
        "validation": validation,
        "report": report,
    }
