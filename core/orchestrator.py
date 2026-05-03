from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

from agents.auditor import audit_document
from agents.classifier import classify_text
from agents.extractor import extract_from_text
from core.normalizer import normalize_extracted
from core.schema_loader import load_schema
from core.schema_models import DocSchema
from core.validator import validate_extracted


class PipelineState(TypedDict, total=False):
    text: str
    schemas_dir: str
    schema_name: str
    schema: DocSchema
    extracted_raw: dict[str, Any]
    normalization: dict[str, Any]
    extracted: dict[str, Any]
    validation: dict[str, Any]
    report: dict[str, Any]


@lru_cache(maxsize=8)
def _build_graph() -> Any:
    from langgraph.graph import END, StateGraph

    graph: StateGraph[PipelineState] = StateGraph(PipelineState)

    def node_classify(state: PipelineState) -> PipelineState:
        return {"schema_name": classify_text(state["text"])}

    def node_extract(state: PipelineState) -> PipelineState:
        schemas_dir = Path(state.get("schemas_dir") or "schemas")
        schema_name = state["schema_name"]
        schema_path = schemas_dir / f"{schema_name}.yaml"
        schema = load_schema(schema_path)
        extracted_raw = extract_from_text(state["text"], schema)
        return {"schema": schema, "extracted_raw": extracted_raw}

    def node_normalize(state: PipelineState) -> PipelineState:
        schema = state["schema"]
        normalization = normalize_extracted(state["extracted_raw"], schema)
        return {"normalization": normalization, "extracted": normalization["normalized"]}

    def node_validate(state: PipelineState) -> PipelineState:
        schema = state["schema"]
        validation = validate_extracted(state["extracted"], schema)
        return {"validation": validation}

    def node_audit(state: PipelineState) -> PipelineState:
        schema = state["schema"]
        report = audit_document(schema, state["extracted"], state["validation"])
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


def run_pipeline(text: str, schemas_dir: str | Path = "schemas") -> dict[str, Any]:
    app = _build_graph()
    state: PipelineState = {"text": text, "schemas_dir": str(schemas_dir)}
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
