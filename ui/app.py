from __future__ import annotations

"""
UI (Streamlit) para DocAudit Agent.

Permite:
- Subir 1 PDF (modo documento) o múltiples PDFs (modo expediente).
- Pegar texto manualmente.
- Ejecutar: extracción → normalización → validación → auditoría, mostrando JSON + Markdown.
"""

import json
import hashlib
import importlib
import io
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml

try:
    import psutil
except ImportError:
    psutil = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_FOLDER = PROJECT_ROOT / "test_data" / "execution_logs"
LOGS_FOLDER.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    # Streamlit ejecuta el script como módulo; esto asegura imports relativos al proyecto.
    sys.path.insert(0, str(PROJECT_ROOT))


def save_execution_log(
    file_name: str,
    file_type: str,
    method: str,
    processing_time: float,
    ram_used_mb: float,
    extracted_fields: dict,
    full_result: dict
) -> None:
    """
    Guarda un registro de la ejecución en un archivo JSON para la tesis.
    
    Args:
        file_name: Nombre del archivo procesado
        file_type: "native_pdf", "scanned_pdf", "image"
        method: Método de extracción usado ("pypdf", "easyocr", "qwen25-vl")
        processing_time: Tiempo de procesamiento en segundos
        ram_used_mb: RAM usada en MB
        extracted_fields: Campos extraídos
        full_result: Resultado completo del pipeline
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "file_name": file_name,
        "file_type": file_type,
        "method": method,
        "processing_time_seconds": round(processing_time, 4),
        "ram_used_mb": round(ram_used_mb, 2),
        "extracted_fields": extracted_fields,
        "full_result": full_result
    }
    
    log_file = LOGS_FOLDER / f"log_{timestamp}_{file_name.replace('.', '_')}.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

from core.schema_models import DocSchema  # noqa: E402
import core.schema_loader as schema_loader  # noqa: E402
from ui.components import (  # noqa: E402
    compare_extracted_to_ground_truth,
    load_ground_truth,
    render_confidence_gauge,
    render_ground_truth_evaluation,
)

try:  # noqa: E402
    import core.document_loader as document_loader
except Exception as exc:  # noqa: E402
    document_loader = None
    _document_loader_import_error: Exception | None = exc
else:
    _document_loader_import_error = None


def _resolve_document_loader_attr(attr_name: str) -> Any:
    global document_loader, _document_loader_import_error

    if document_loader is None:
        try:
            document_loader = importlib.import_module("core.document_loader")
            _document_loader_import_error = None
        except Exception as exc:
            _document_loader_import_error = exc
            return None

    attr = getattr(document_loader, attr_name, None)
    if attr is not None:
        return attr

    try:
        document_loader = importlib.reload(document_loader)
        _document_loader_import_error = None
    except Exception as exc:
        _document_loader_import_error = exc
        return None

    return getattr(document_loader, attr_name, None)


def _document_loader_error_message(base_message: str) -> str:
    if _document_loader_import_error is None:
        return (
            f"{base_message} Detén y vuelve a iniciar Streamlit para recargar los módulos, "
            "y verifica que tu entorno tenga el código actualizado."
        )
    return (
        f"{base_message} Causa detectada: "
        f"{_document_loader_import_error.__class__.__name__}: {_document_loader_import_error}"
    )

extract_text_from_pdf_bytes = _resolve_document_loader_attr("extract_text_from_pdf_bytes")  # noqa: E402
extract_text_from_scanned_pdf_bytes = _resolve_document_loader_attr("extract_text_from_scanned_pdf_bytes")  # noqa: E402
extract_text_from_image_bytes = _resolve_document_loader_attr("extract_text_from_image_bytes")  # noqa: E402
from core.orchestrator import run_expediente, run_pipeline  # noqa: E402
from core.schema_loader import load_schema  # noqa: E402
from core.normalizer import normalize_extracted  # noqa: E402



st.set_page_config(page_title="DocAudit Agent", layout="wide")
st.markdown(
    """
<style>
.stApp { background: #0b1220; color: #e5e7eb; }
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stHeader"] { background: #0b1220; }
.block-container { padding-top: 1.5rem; }
div[data-testid="stMetric"] { background: #0f172a; border: 1px solid #334155; padding: 0.75rem; border-radius: 0.75rem; }
.da-panel {
    background: linear-gradient(180deg, rgba(15,23,42,0.96), rgba(15,23,42,0.88));
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 18px;
    padding: 1rem 1.1rem;
    margin-bottom: 1rem;
}
.da-hero {
    background: linear-gradient(135deg, rgba(37,99,235,0.18), rgba(14,165,233,0.08));
    border: 1px solid rgba(96,165,250,0.30);
    border-radius: 20px;
    padding: 1.2rem 1.25rem;
    margin-bottom: 1rem;
}
.da-kpi {
    background: linear-gradient(180deg, rgba(15,23,42,0.98), rgba(30,41,59,0.92));
    border: 1px solid rgba(96,165,250,0.20);
    border-radius: 18px;
    padding: 0.95rem 1rem;
    min-height: 120px;
}
.da-kpi-label {
    font-size: 0.82rem;
    color: #94a3b8;
    margin-bottom: 0.35rem;
}
.da-kpi-value {
    font-size: 1.7rem;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1.1;
}
.da-kpi-note {
    font-size: 0.82rem;
    color: #cbd5e1;
    margin-top: 0.45rem;
}
.da-badge {
    display: inline-block;
    padding: 0.2rem 0.55rem;
    margin: 0 0.35rem 0.35rem 0;
    border-radius: 999px;
    font-size: 0.78rem;
    background: rgba(59,130,246,0.12);
    border: 1px solid rgba(96,165,250,0.22);
    color: #dbeafe;
}
.da-section-title {
    font-size: 1rem;
    font-weight: 600;
    color: #f8fafc;
    margin-bottom: 0.25rem;
}
.da-section-copy {
    color: #cbd5e1;
    font-size: 0.9rem;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("DocAudit Agent")
st.caption(
    "Plataforma multi-agente para extracción, normalización, validación y auditoría documental. "
    "Diseñada para generar datos estructurados y trazables que apoyen el análisis y la toma de decisiones."
)
app_section = st.radio(
    "Sección",
    options=["Operación documental", "Centro analítico"],
    horizontal=True,
    key="app_section_selector",
)


def _to_table_rows(*, extracted: dict[str, Any], schema: Any | None) -> list[dict[str, Any]]:
    if not isinstance(extracted, dict):
        return []
    ordered_keys: list[str] = []
    if schema is not None and hasattr(schema, "fields"):
        try:
            ordered_keys = [f.name for f in getattr(schema, "fields")]
        except Exception:
            ordered_keys = []
    keys = ordered_keys or sorted(extracted.keys())
    rows: list[dict[str, Any]] = []
    for k in keys:
        if k not in extracted:
            continue
        v = extracted.get(k)
        if v is None or (isinstance(v, str) and v.strip() == ""):
            status = "vacío"
        else:
            status = "ok"
        rows.append({"campo": k, "valor": v, "estado": status})
    return rows


def _render_json(*, label: str, payload: Any) -> None:
    with st.expander(label, expanded=False):
        st.json(payload)


def _render_issues_table(*, issues: list[dict[str, Any]]) -> None:
    if not issues:
        st.success("Sin incidencias.")
        return
    st.dataframe(issues, use_container_width=True, hide_index=True)


def _render_score_gauge(*, score: float, key: str) -> None:
    try:
        render_confidence_gauge(score=score, key=key)
    except TypeError:
        render_confidence_gauge(score=score)


def _document_status_info(*, doc_result: dict[str, Any]) -> tuple[str, str]:
    validation = doc_result.get("validation") if isinstance(doc_result.get("validation"), dict) else {}
    report = doc_result.get("report") if isinstance(doc_result.get("report"), dict) else {}
    report_json = report.get("json") if isinstance(report.get("json"), dict) else {}
    decision_rules = report_json.get("decision_rules") if isinstance(report_json.get("decision_rules"), list) else []
    issues = validation.get("issues") if isinstance(validation.get("issues"), list) else []

    has_critical_rule_failure = any(
        isinstance(rule, dict) and rule.get("cumple") is False and str(rule.get("severidad") or "").lower() == "critica"
        for rule in decision_rules
    )
    if has_critical_rule_failure:
        return "CRITICO", "Fallo de regla critica"
    if issues:
        return "REVISAR", f"{len(issues)} incidencia(s)"
    if validation.get("valid") is True:
        return "OK", "Validacion correcta"
    return "REVISAR", "Pendiente de revision"


def _render_document_status_notice(*, status: str, reason: str) -> None:
    message = f"Estado del documento: {status} | {reason}"
    if status == "OK":
        st.success(message)
        return
    if status == "CRITICO":
        st.error(message)
        return
    st.warning(message)


def _rows_to_csv_bytes(*, rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        return b""
    headers: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in headers:
                headers.append(key)
    buffer = io.StringIO()
    buffer.write(",".join(headers) + "\n")
    for row in rows:
        values: list[str] = []
        for key in headers:
            value = row.get(key, "")
            text = "" if value is None else str(value)
            text = text.replace('"', '""')
            if any(ch in text for ch in [",", "\n", '"']):
                text = f'"{text}"'
            values.append(text)
        buffer.write(",".join(values) + "\n")
    return buffer.getvalue().encode("utf-8")


def _doc_meta_name(*, meta: list[dict[str, Any]] | None, idx: int, fallback: str) -> str:
    if isinstance(meta, list) and idx < len(meta) and isinstance(meta[idx], dict):
        return str(meta[idx].get("documento") or fallback)
    return fallback


def _collect_issue_rows(*, docs: list[dict[str, Any]], meta: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, doc in enumerate(docs):
        if not isinstance(doc, dict):
            continue
        raw_index = doc.get("doc_index")
        doc_number = (raw_index + 1) if isinstance(raw_index, int) else (idx + 1)
        doc_name = _doc_meta_name(meta=meta, idx=idx, fallback=f"Documento {doc_number}")
        validation = doc.get("validation") if isinstance(doc.get("validation"), dict) else {}
        issues = validation.get("issues") if isinstance(validation.get("issues"), list) else []
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            rows.append(
                {
                    "doc_index": doc_number,
                    "documento": doc_name,
                    "campo": issue.get("field") or issue.get("campo") or "",
                    "tipo": issue.get("type") or issue.get("tipo") or issue.get("rule") or "",
                    "mensaje": issue.get("message") or issue.get("mensaje") or issue.get("detail") or "",
                    "severidad": issue.get("severity") or issue.get("severidad") or "",
                }
            )
    return rows


def _collect_autocorrection_rows(*, docs: list[dict[str, Any]], meta: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, doc in enumerate(docs):
        if not isinstance(doc, dict):
            continue
        raw_index = doc.get("doc_index")
        doc_number = (raw_index + 1) if isinstance(raw_index, int) else (idx + 1)
        doc_name = _doc_meta_name(meta=meta, idx=idx, fallback=f"Documento {doc_number}")
        normalization = doc.get("normalization") if isinstance(doc.get("normalization"), dict) else {}
        autocorrections = normalization.get("autocorrections") if isinstance(normalization.get("autocorrections"), list) else []
        for change in autocorrections:
            if not isinstance(change, dict):
                continue
            rows.append(
                {
                    "doc_index": doc_number,
                    "documento": doc_name,
                    "campo": change.get("field") or change.get("campo") or "",
                    "origen": change.get("from") if change.get("from") is not None else change.get("old"),
                    "corregido": change.get("to") if change.get("to") is not None else change.get("new"),
                    "motivo": change.get("reason") or change.get("motivo") or change.get("method") or "",
                }
            )
    return rows


def _summarize_issue_patterns(*, issue_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    sample_by_pattern: dict[str, dict[str, Any]] = {}
    for row in issue_rows:
        campo = str(row.get("campo") or "").strip()
        tipo = str(row.get("tipo") or "").strip()
        mensaje = str(row.get("mensaje") or "").strip()
        pattern = " | ".join([part for part in [campo, tipo, mensaje] if part]) or "incidencia_sin_detalle"
        counter[pattern] += 1
        sample_by_pattern.setdefault(pattern, row)

    summary_rows: list[dict[str, Any]] = []
    for pattern, count in counter.most_common():
        sample = sample_by_pattern.get(pattern, {})
        summary_rows.append(
            {
                "patron": pattern,
                "frecuencia": count,
                "campo": sample.get("campo") or "",
                "tipo": sample.get("tipo") or "",
                "mensaje": sample.get("mensaje") or "",
            }
        )
    return summary_rows


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except Exception:
        return None


def _infer_variant_name(*, file_name: str | None, file_type: str | None) -> str:
    base_name = str(file_name or "").lower()
    if "scanned_blurry_pdf" in base_name:
        return "scanned_blurry_pdf"
    if "image_handwritten" in base_name:
        return "image_handwritten"
    if "image_photo" in base_name:
        return "image_photo"
    if "native_pdf" in base_name:
        return "native_pdf"
    if str(file_type or "").lower() == "scanned_pdf":
        return "scanned_pdf"
    if str(file_type or "").lower() == "native_pdf":
        return "native_pdf"
    if str(file_type or "").lower() == "image":
        return "image"
    return "desconocido"


def _load_execution_log_rows(*, max_logs: int = 300) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    log_files = sorted(LOGS_FOLDER.glob("log_*.json"), reverse=True)[:max_logs]
    for log_file in log_files:
        try:
            payload = json.loads(log_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        full_result = payload.get("full_result") if isinstance(payload.get("full_result"), dict) else {}
        validation = full_result.get("validation") if isinstance(full_result.get("validation"), dict) else {}
        report = full_result.get("report") if isinstance(full_result.get("report"), dict) else {}
        report_json = report.get("json") if isinstance(report.get("json"), dict) else {}
        schema = full_result.get("schema") if isinstance(full_result.get("schema"), dict) else {}
        extracted_fields = payload.get("extracted_fields") if isinstance(payload.get("extracted_fields"), dict) else {}
        processing_time = _safe_float(payload.get("processing_time_seconds"))
        ram_used = _safe_float(payload.get("ram_used_mb"))
        rows.append(
            {
                "timestamp": payload.get("timestamp"),
                "file_name": payload.get("file_name"),
                "file_type": payload.get("file_type"),
                "variant": _infer_variant_name(file_name=payload.get("file_name"), file_type=payload.get("file_type")),
                "method": payload.get("method"),
                "schema_name": schema.get("name"),
                "document_type": full_result.get("document_type"),
                "processing_time_seconds": processing_time,
                "ram_used_mb": ram_used if isinstance(ram_used, float) and ram_used >= 0 else None,
                "valid": validation.get("valid"),
                "issues_count": len(validation.get("issues") or []) if isinstance(validation.get("issues"), list) else 0,
                "score_confianza": _safe_float(report_json.get("score_confianza")),
                "extracted_fields_count": len([v for v in extracted_fields.values() if v not in (None, "", [])]),
            }
        )
    return rows


def _group_log_rows(*, rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        group_name = str(row.get(key) or "desconocido")
        stats = grouped.setdefault(
            group_name,
            {
                key: group_name,
                "documentos": 0,
                "validos": 0,
                "latencias": [],
                "rams": [],
                "scores": [],
                "campos": [],
            },
        )
        stats["documentos"] += 1
        if row.get("valid") is True:
            stats["validos"] += 1
        if isinstance(row.get("processing_time_seconds"), (int, float)):
            stats["latencias"].append(float(row["processing_time_seconds"]))
        if isinstance(row.get("ram_used_mb"), (int, float)):
            stats["rams"].append(float(row["ram_used_mb"]))
        if isinstance(row.get("score_confianza"), (int, float)):
            stats["scores"].append(float(row["score_confianza"]))
        if isinstance(row.get("extracted_fields_count"), (int, float)):
            stats["campos"].append(float(row["extracted_fields_count"]))

    summary_rows: list[dict[str, Any]] = []
    for _, stats in sorted(grouped.items(), key=lambda item: str(item[0]).lower()):
        documentos = stats["documentos"]
        summary_rows.append(
            {
                key: stats[key],
                "documentos": documentos,
                "validos": stats["validos"],
                "tasa_validez": round((stats["validos"] / documentos), 4) if documentos else 0.0,
                "latencia_media_s": round(sum(stats["latencias"]) / len(stats["latencias"]), 4) if stats["latencias"] else None,
                "ram_media_mb": round(sum(stats["rams"]) / len(stats["rams"]), 2) if stats["rams"] else None,
                "score_medio": round(sum(stats["scores"]) / len(stats["scores"]), 4) if stats["scores"] else None,
                "campos_extraidos_medios": round(sum(stats["campos"]) / len(stats["campos"]), 2) if stats["campos"] else None,
            }
        )
    return summary_rows


def _collect_log_issue_rows(*, max_logs: int = 300) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    log_files = sorted(LOGS_FOLDER.glob("log_*.json"), reverse=True)[:max_logs]
    for log_file in log_files:
        try:
            payload = json.loads(log_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        full_result = payload.get("full_result") if isinstance(payload.get("full_result"), dict) else {}
        validation = full_result.get("validation") if isinstance(full_result.get("validation"), dict) else {}
        issues = validation.get("issues") if isinstance(validation.get("issues"), list) else []
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            rows.append(
                {
                    "file_name": payload.get("file_name"),
                    "schema_name": (full_result.get("schema") or {}).get("name") if isinstance(full_result.get("schema"), dict) else None,
                    "file_type": payload.get("file_type"),
                    "variant": _infer_variant_name(file_name=payload.get("file_name"), file_type=payload.get("file_type")),
                    "method": payload.get("method"),
                    "campo": issue.get("field") or issue.get("campo") or "",
                    "tipo": issue.get("code") or issue.get("type") or issue.get("tipo") or "",
                    "mensaje": issue.get("message") or issue.get("mensaje") or "",
                    "severidad": issue.get("level") or issue.get("severidad") or "",
                }
            )
    return rows


def _resolve_ground_truth_for_filename(file_name: str | None) -> Path | None:
    if not isinstance(file_name, str) or not file_name.strip():
        return None
    base_name = Path(file_name).name
    stem = Path(base_name).stem

    sample_docs_root = PROJECT_ROOT / "data" / "sample_docs"
    experiment_roots = [
        PROJECT_ROOT / "test_data" / "experimentos_120_high_validity",
        PROJECT_ROOT / "test_data" / "experimentos_120",
    ]

    if stem.startswith("contrato_hipoteca_esp_"):
        suffix = stem.rsplit("_", 1)[-1]
        candidate = sample_docs_root / "caso_uso_1_auditoria_hipotecaria" / f"ground_truth_esp_{suffix}.json"
        if candidate.exists():
            return candidate

    for candidate_name in [f"{stem}.json", f"{stem}_ground_truth.json"]:
        hits = list(sample_docs_root.rglob(candidate_name))
        if hits:
            return hits[0]
        for experiment_root in experiment_roots:
            hits = list(experiment_root.rglob(candidate_name))
            if hits:
                return hits[0]
    return None


def _load_log_evaluation_rows(*, max_logs: int = 300) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    log_files = sorted(LOGS_FOLDER.glob("log_*.json"), reverse=True)[:max_logs]
    for log_file in log_files:
        try:
            payload = json.loads(log_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        gt_path = _resolve_ground_truth_for_filename(payload.get("file_name"))
        if gt_path is None or not gt_path.exists():
            continue

        full_result = payload.get("full_result") if isinstance(payload.get("full_result"), dict) else {}
        schema = full_result.get("schema") if isinstance(full_result.get("schema"), dict) else {}
        normalized = (full_result.get("normalization") or {}).get("normalized") if isinstance(full_result.get("normalization"), dict) else None
        extracted = normalized if isinstance(normalized, dict) else payload.get("extracted_fields")
        if not isinstance(extracted, dict):
            continue

        try:
            ground_truth = load_ground_truth(gt_path)
            cmp = compare_extracted_to_ground_truth(extracted=extracted, ground_truth=ground_truth)
        except Exception:
            continue

        summary = cmp.get("summary") if isinstance(cmp.get("summary"), dict) else {}
        total_fields = int(summary.get("total") or 0)
        matched_fields = int(summary.get("matched") or 0)
        exact_match = total_fields > 0 and matched_fields == total_fields
        row = {
            "file_name": payload.get("file_name"),
            "schema_name": schema.get("name"),
            "file_type": payload.get("file_type"),
            "variant": _infer_variant_name(file_name=payload.get("file_name"), file_type=payload.get("file_type")),
            "method": payload.get("method"),
            "ground_truth_path": str(gt_path),
            "fields_total": total_fields,
            "fields_matched": matched_fields,
            "match_rate": _safe_float(summary.get("match_rate")),
            "precision": _safe_float(summary.get("precision")),
            "recall": _safe_float(summary.get("recall")),
            "f1": _safe_float(summary.get("f1")),
            "exact_match": exact_match,
        }
        summary_rows.append(row)

        for field_row in cmp.get("rows") or []:
            if not isinstance(field_row, dict):
                continue
            field_rows.append(
                {
                    "file_name": payload.get("file_name"),
                    "schema_name": schema.get("name"),
                    "file_type": payload.get("file_type"),
                    "variant": _infer_variant_name(file_name=payload.get("file_name"), file_type=payload.get("file_type")),
                    "method": payload.get("method"),
                    "campo": field_row.get("campo"),
                    "match": field_row.get("match"),
                    "esperado": field_row.get("esperado"),
                    "extraido": field_row.get("extraido"),
                }
            )

    return summary_rows, field_rows


def _group_evaluation_rows(*, rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        group_name = str(row.get(key) or "desconocido")
        stats = grouped.setdefault(
            group_name,
            {
                key: group_name,
                "documentos": 0,
                "exact_matches": 0,
                "fields_total": 0,
                "fields_matched": 0,
                "f1_values": [],
                "match_rate_values": [],
            },
        )
        stats["documentos"] += 1
        if row.get("exact_match") is True:
            stats["exact_matches"] += 1
        stats["fields_total"] += int(row.get("fields_total") or 0)
        stats["fields_matched"] += int(row.get("fields_matched") or 0)
        if isinstance(row.get("f1"), (int, float)):
            stats["f1_values"].append(float(row["f1"]))
        if isinstance(row.get("match_rate"), (int, float)):
            stats["match_rate_values"].append(float(row["match_rate"]))

    out: list[dict[str, Any]] = []
    for _, stats in sorted(grouped.items(), key=lambda item: str(item[0]).lower()):
        documentos = stats["documentos"]
        out.append(
            {
                key: stats[key],
                "documentos_evaluados": documentos,
                "exact_matches": stats["exact_matches"],
                "exact_match_rate": round(stats["exact_matches"] / documentos, 4) if documentos else 0.0,
                "field_level_f1_medio": round(sum(stats["f1_values"]) / len(stats["f1_values"]), 4) if stats["f1_values"] else None,
                "match_rate_medio": round(sum(stats["match_rate_values"]) / len(stats["match_rate_values"]), 4) if stats["match_rate_values"] else None,
                "campos_evaluados": stats["fields_total"],
                "campos_exactos": stats["fields_matched"],
            }
        )
    return out


def _summarize_field_errors(*, field_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    for row in field_rows:
        field_name = str(row.get("campo") or "campo_desconocido")
        totals[field_name] += 1
        if row.get("match") is not True:
            counter[field_name] += 1
    out: list[dict[str, Any]] = []
    for field_name, mismatches in counter.most_common():
        total = totals.get(field_name, 0)
        out.append(
            {
                "campo": field_name,
                "errores": mismatches,
                "evaluaciones": total,
                "tasa_error": round(mismatches / total, 4) if total else 0.0,
            }
        )
    return out


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
SUPPORTED_UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg", "webp"]


def _infer_input_kind(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in SUPPORTED_IMAGE_SUFFIXES:
        return "image"
    return "unknown"


def _sanitize_folder_path(folder: str) -> Path:
    raw = (folder or "").strip().strip('"').strip("'")
    return Path(raw).expanduser()


def _build_extraction_cache_key(*, doc_id: str, input_kind: str, use_vision: bool) -> str:
    return f"{doc_id}|{input_kind}|vision={int(use_vision)}"


def _pick_folder_via_dialog(initial_dir: str | None = None) -> tuple[str | None, str | None]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        return None, f"No se pudo abrir el selector de carpetas: {exc}"

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        start_dir = _sanitize_folder_path(initial_dir) if initial_dir else PROJECT_ROOT
        selected = filedialog.askdirectory(initialdir=str(start_dir))
    except Exception as exc:
        return None, f"No se pudo abrir el selector de carpetas: {exc}"
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass

    if not selected:
        return None, None
    return selected, None


def _browse_folder_callback() -> None:
    selected, picker_error = _pick_folder_via_dialog(st.session_state.get("folder_path", ""))
    st.session_state["folder_picker_error"] = picker_error
    if selected:
        st.session_state["folder_path"] = selected


def _load_documents_from_folder(folder: str) -> tuple[list[dict[str, Any]], str | None]:
    p = _sanitize_folder_path(folder)
    if not str(p).strip():
        return [], "Debes indicar una ruta de carpeta."
    if not p.exists():
        return [], f"La carpeta no existe: {p}"
    if not p.is_dir():
        return [], f"La ruta indicada no es una carpeta: {p}"

    files = sorted(
        [
            x
            for x in p.rglob("*")
            if x.is_file() and x.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
        ],
        key=lambda item: str(item).lower(),
    )
    if not files:
        return [], "No se encontraron PDFs o imágenes compatibles en la carpeta."

    out: list[dict[str, Any]] = []
    for f in files:
        try:
            display_name = str(f.relative_to(p))
        except ValueError:
            display_name = f.name
        out.append(
            {
                "name": display_name,
                "bytes": f.read_bytes(),
                "source": "carpeta",
                "path": str(f),
                "input_kind": _infer_input_kind(f.name),
            }
        )
    return out, None


def _build_doc_option_labels(*, docs: list[dict[str, Any]], meta: list[dict[str, Any]] | None) -> list[str]:
    labels: list[str] = []
    for idx, d in enumerate(docs):
        if not isinstance(d, dict):
            continue
        raw_index = d.get("doc_index")
        doc_number = (raw_index + 1) if isinstance(raw_index, int) else (idx + 1)
        file_name = None
        if isinstance(meta, list) and idx < len(meta) and isinstance(meta[idx], dict):
            file_name = meta[idx].get("documento")
        labels.append(f"Documento {doc_number} - {file_name or 'sin nombre'}")
    return labels


def _render_document_result_detail(
    *,
    chosen: dict[str, Any],
    selection_key: str,
    source_text: str = "",
) -> None:
    validation = chosen.get("validation") if isinstance(chosen.get("validation"), dict) else {}
    report = chosen.get("report") if isinstance(chosen.get("report"), dict) else {}
    report_json = report.get("json") if isinstance(report.get("json"), dict) else {}
    issues = validation.get("issues") if isinstance(validation.get("issues"), list) else []
    estado_doc, motivo_doc = _document_status_info(doc_result=chosen)
    score = report_json.get("score_confianza")

    c_det_1, c_det_2, c_det_3, c_det_4 = st.columns(4)
    c_det_1.metric("Estado", estado_doc)
    c_det_2.metric("Validez", "Válido" if validation.get("valid") else "Revisar")
    c_det_3.metric("Incidencias", str(len(issues)))
    c_det_4.metric("Confianza", f"{float(score):.2f}" if isinstance(score, (int, float)) else "-")
    _render_document_status_notice(status=estado_doc, reason=motivo_doc)

    sub_tabs = st.tabs(["Extracción", "Normalización", "Validación", "Auditoría", "JSON"])
    with sub_tabs[0]:
        if source_text.strip():
            st.text_area(
                "Texto extraído del documento",
                value=source_text,
                height=220,
                key=f"source_text_{selection_key}",
            )
        else:
            st.info("No hay texto extraído disponible para este documento.")
        st.dataframe(
            _to_table_rows(extracted=chosen.get("extracted") or {}, schema=None),
            use_container_width=True,
            hide_index=True,
        )
        _render_json(label="Ver extracted_raw", payload=chosen.get("extracted_raw"))
    with sub_tabs[1]:
        norm = chosen.get("normalization") if isinstance(chosen.get("normalization"), dict) else {}
        changes = norm.get("changes") if isinstance(norm.get("changes"), list) else []
        autocorrections = norm.get("autocorrections") if isinstance(norm.get("autocorrections"), list) else []
        if changes:
            st.dataframe(changes, use_container_width=True, hide_index=True)
        else:
            st.info("Sin cambios de normalización.")
        if autocorrections:
            st.write("Autocorrecciones")
            st.dataframe(autocorrections, use_container_width=True, hide_index=True)
        _render_json(label="Ver normalizado (JSON)", payload=norm.get("normalized"))
    with sub_tabs[2]:
        _render_issues_table(issues=issues)
        _render_json(label="Ver validación (JSON)", payload=validation)
    with sub_tabs[3]:
        rep_md = report.get("markdown") if isinstance(report.get("markdown"), str) else ""
        if isinstance(score, (int, float)):
            _render_score_gauge(score=float(score), key=f"score_confianza_{selection_key}")
        if rep_md:
            st.markdown(rep_md)
        _render_json(label="Ver informe (JSON)", payload=report_json)
    with sub_tabs[4]:
        st.json(chosen)


def _render_loaded_documents_preview(*, pdf_items: list[dict[str, Any]]) -> None:
    if not pdf_items:
        return []
    st.write("Explorador de documentos cargados")
    options = [str(item.get("name") or f"documento_{i+1}") for i, item in enumerate(pdf_items)]
    selected = st.selectbox("Selecciona un documento cargado", options=options, key="preview_loaded_document")
    selected_idx = options.index(selected) if selected in options else 0
    item = pdf_items[selected_idx] if selected_idx < len(pdf_items) else {}

    preview_text = st.session_state.get("expediente_texts")
    preview_value = ""
    if isinstance(preview_text, list) and selected_idx < len(preview_text):
        preview_value = str(preview_text[selected_idx] or "")

    c_prev_1, c_prev_2, c_prev_3, c_prev_4 = st.columns(4)
    c_prev_1.metric("Documento", str(selected_idx + 1))
    c_prev_2.metric("Origen", str(item.get("source") or "-"))
    c_prev_3.metric("Tipo", str(item.get("input_kind") or "-"))
    c_prev_4.metric("Caracteres", str(len(preview_value)))

    if item.get("path"):
        st.caption(str(item.get("path")))

    file_bytes = item.get("bytes") or b""
    input_kind = str(item.get("input_kind") or "")
    if input_kind == "image":
        st.image(file_bytes, caption=str(item.get("name") or "imagen"), use_container_width=True)

    if preview_value.strip():
        st.text_area(
            "Texto extraído preliminar",
            value=preview_value,
            height=260,
            key=f"preview_text_{selected_idx}",
        )
    else:
        st.info("El texto extraído se mostrará aquí una vez procesado el documento.")


def _load_documents_from_uploads(uploaded: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for up in uploaded:
        out.append(
            {
                "name": up.name,
                "bytes": up.read(),
                "source": "subida",
                "path": None,
                "input_kind": _infer_input_kind(up.name),
            }
        )
    return out


def _avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _max_value(values: list[float]) -> float | None:
    return round(max(values), 4) if values else None


def _format_seconds(value: float | None) -> str:
    return f"{value:.2f} s" if isinstance(value, (int, float)) else "-"


def _format_mb(value: float | None) -> str:
    return f"{value:.2f} MB" if isinstance(value, (int, float)) else "-"


def _format_percent(value: float | None) -> str:
    return f"{value:.0%}" if isinstance(value, (int, float)) else "-"


def _format_number(value: float | None, decimals: int = 1) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{float(value):.{decimals}f}"


def _render_kpi_card(*, title: str, value: str, note: str) -> None:
    st.markdown(
        (
            '<div class="da-kpi">'
            f'<div class="da-kpi-label">{title}</div>'
            f'<div class="da-kpi-value">{value}</div>'
            f'<div class="da-kpi-note">{note}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _render_dataframe_or_message(*, rows: list[dict[str, Any]], empty_message: str) -> None:
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info(empty_message)


def _render_chart_from_rows(
    *,
    rows: list[dict[str, Any]],
    index_key: str,
    value_keys: list[str],
    empty_message: str,
) -> None:
    if not rows:
        st.info(empty_message)
        return
    chart_df = _to_dataframe(rows)
    available_columns = [col for col in value_keys if col in chart_df.columns]
    if not available_columns or index_key not in chart_df.columns:
        st.info(empty_message)
        return
    chart_df = chart_df[[index_key, *available_columns]].set_index(index_key)
    st.bar_chart(chart_df, use_container_width=True)


def _render_analytics_dashboard() -> None:
    st.header("Centro Analítico")
    st.caption(
        "Espacio dedicado al análisis del comportamiento del sistema, la calidad del dato, "
        "la comparación experimental entre variantes documentales y la exportación de métricas para tesis."
    )

    log_rows = _load_execution_log_rows()
    log_issue_rows = _collect_log_issue_rows()
    eval_rows, eval_field_rows = _load_log_evaluation_rows()

    all_schema_options = sorted(
        {
            str(row.get("schema_name"))
            for row in [*log_rows, *eval_rows]
            if row.get("schema_name")
        }
    )
    all_variant_options = sorted(
        {
            str(row.get("variant"))
            for row in [*log_rows, *eval_rows]
            if row.get("variant")
        }
    )
    all_method_options = sorted(
        {
            str(row.get("method"))
            for row in [*log_rows, *eval_rows]
            if row.get("method")
        }
    )

    st.markdown(
        """
<div class="da-hero">
  <div class="da-section-title">Panel separado para comportamiento, métricas y tratamiento del dato</div>
  <div class="da-section-copy">
    Esta vista concentra el análisis histórico del sistema, la calidad de extracción y la comparación experimental
    entre variantes documentales. La operación diaria queda fuera de esta sección para mantener la app clara.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    filter_panel_left, filter_panel_right = st.columns([2.2, 1.2])
    with filter_panel_left:
        st.markdown('<div class="da-panel">', unsafe_allow_html=True)
        st.markdown("**Filtros de análisis**")
        f1, f2, f3 = st.columns(3)
        selected_schemas = f1.multiselect(
            "Caso de uso",
            options=all_schema_options,
            default=all_schema_options,
            key="analytics_schema_filter",
        )
        selected_variants = f2.multiselect(
            "Variante documental",
            options=all_variant_options,
            default=all_variant_options,
            key="analytics_variant_filter",
        )
        selected_methods = f3.multiselect(
            "Método de extracción",
            options=all_method_options,
            default=all_method_options,
            key="analytics_method_filter",
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with filter_panel_right:
        st.markdown('<div class="da-panel">', unsafe_allow_html=True)
        st.markdown("**Uso sugerido**")
        st.caption(
            "1. Filtra por caso o variante.\n"
            "2. Revisa los KPIs principales.\n"
            "3. Baja a los gráficos y exporta CSV para tesis."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    filtered_log_rows = [
        row
        for row in log_rows
        if (not selected_schemas or row.get("schema_name") in selected_schemas)
        and (not selected_variants or row.get("variant") in selected_variants)
        and (not selected_methods or row.get("method") in selected_methods)
    ]
    filtered_eval_rows = [
        row
        for row in eval_rows
        if (not selected_schemas or row.get("schema_name") in selected_schemas)
        and (not selected_variants or row.get("variant") in selected_variants)
        and (not selected_methods or row.get("method") in selected_methods)
    ]
    filtered_log_issue_rows = [
        row
        for row in log_issue_rows
        if (not selected_schemas or row.get("schema_name") in selected_schemas)
        and (not selected_variants or row.get("variant") in selected_variants)
        and (not selected_methods or row.get("method") in selected_methods)
    ]
    filtered_eval_field_rows = [
        row
        for row in eval_field_rows
        if (not selected_schemas or row.get("schema_name") in selected_schemas)
        and (not selected_variants or row.get("variant") in selected_variants)
        and (not selected_methods or row.get("method") in selected_methods)
    ]

    if not filtered_log_rows:
        st.info("No hay datos históricos que coincidan con los filtros seleccionados.")
        return

    active_filter_values = [
        *(selected_schemas or ["Todos los casos"]),
        *(selected_variants or ["Todas las variantes"]),
        *(selected_methods or ["Todos los métodos"]),
    ]
    badges = "".join(f'<span class="da-badge">{value}</span>' for value in active_filter_values[:12])
    if badges:
        st.markdown(badges, unsafe_allow_html=True)

    log_issue_summary_rows = _summarize_issue_patterns(issue_rows=filtered_log_issue_rows)
    by_method_rows = _group_log_rows(rows=filtered_log_rows, key="method")
    by_file_type_rows = _group_log_rows(rows=filtered_log_rows, key="file_type")
    by_schema_rows = _group_log_rows(rows=filtered_log_rows, key="schema_name")
    by_variant_rows = _group_log_rows(rows=filtered_log_rows, key="variant")
    eval_by_method_rows = _group_evaluation_rows(rows=filtered_eval_rows, key="method")
    eval_by_file_type_rows = _group_evaluation_rows(rows=filtered_eval_rows, key="file_type")
    eval_by_schema_rows = _group_evaluation_rows(rows=filtered_eval_rows, key="schema_name")
    eval_by_variant_rows = _group_evaluation_rows(rows=filtered_eval_rows, key="variant")
    field_error_rows = _summarize_field_errors(field_rows=filtered_eval_field_rows)

    valid_count = len([row for row in filtered_log_rows if row.get("valid") is True])
    latency_values = [float(row["processing_time_seconds"]) for row in filtered_log_rows if isinstance(row.get("processing_time_seconds"), (int, float))]
    ram_values = [float(row["ram_used_mb"]) for row in filtered_log_rows if isinstance(row.get("ram_used_mb"), (int, float))]
    score_values = [float(row["score_confianza"]) for row in filtered_log_rows if isinstance(row.get("score_confianza"), (int, float))]
    extracted_values = [float(row["extracted_fields_count"]) for row in filtered_log_rows if isinstance(row.get("extracted_fields_count"), (int, float))]
    logs_count = len(filtered_log_rows)
    valid_rate = (valid_count / logs_count) if logs_count else None
    avg_latency = _avg(latency_values)
    peak_ram = _max_value(ram_values)
    avg_score = _avg(score_values)
    avg_extracted = _avg(extracted_values)

    exact_matches = len([row for row in filtered_eval_rows if row.get("exact_match") is True])
    f1_values = [float(row["f1"]) for row in filtered_eval_rows if isinstance(row.get("f1"), (int, float))]
    match_rate_values = [float(row["match_rate"]) for row in filtered_eval_rows if isinstance(row.get("match_rate"), (int, float))]
    fields_total = sum(int(row.get("fields_total") or 0) for row in filtered_eval_rows)
    fields_matched = sum(int(row.get("fields_matched") or 0) for row in filtered_eval_rows)
    exact_match_rate = (exact_matches / len(filtered_eval_rows)) if filtered_eval_rows else None
    avg_f1 = _avg(f1_values)
    avg_match_rate = _avg(match_rate_values)

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        _render_kpi_card(
            title="Logs analizados",
            value=str(logs_count),
            note="Ejecuciones históricas que entran en el filtro actual.",
        )
    with kpi_cols[1]:
        _render_kpi_card(
            title="Latencia media",
            value=_format_seconds(avg_latency),
            note=f"RAM peak: {_format_mb(peak_ram)}",
        )
    with kpi_cols[2]:
        _render_kpi_card(
            title="Tasa de validez",
            value=_format_percent(valid_rate),
            note=f"Score medio: {_format_percent(avg_score)}",
        )
    with kpi_cols[3]:
        _render_kpi_card(
            title="Campos extraídos",
            value=_format_number(avg_extracted, 1),
            note=f"Incidencias detectadas: {len(filtered_log_issue_rows)}",
        )

    st.markdown("**Evaluación formal**")
    eval_cols = st.columns(4)
    with eval_cols[0]:
        _render_kpi_card(
            title="Docs evaluados",
            value=str(len(filtered_eval_rows)),
            note="Comparados contra ground truth disponible.",
        )
    with eval_cols[1]:
        _render_kpi_card(
            title="Exact Match Rate",
            value=_format_percent(exact_match_rate),
            note=f"Exact matches: {exact_matches}",
        )
    with eval_cols[2]:
        _render_kpi_card(
            title="Field-level F1",
            value=_format_percent(avg_f1),
            note=f"Match rate medio: {_format_percent(avg_match_rate)}",
        )
    with eval_cols[3]:
        _render_kpi_card(
            title="Campos evaluados",
            value=str(fields_total),
            note=f"Coincidencias exactas: {fields_matched}",
        )

    executive_left, executive_right = st.columns([1.8, 1.1])
    with executive_left:
        st.markdown(
            '<div class="da-panel"><div class="da-section-title">Lectura rápida</div>'
            '<div class="da-section-copy">Esta parte resume si el sistema está siendo estable, preciso y consistente '
            'para los casos y variantes filtrados.</div></div>',
            unsafe_allow_html=True,
        )
    with executive_right:
        if filtered_eval_rows and eval_by_variant_rows:
            ordered_variant_rows = sorted(
                eval_by_variant_rows,
                key=lambda row: float(row.get("field_level_f1_medio") or -1),
                reverse=True,
            )
            best_variant = ordered_variant_rows[0]
            worst_variant = ordered_variant_rows[-1]
            st.markdown(
                '<div class="da-panel"><div class="da-section-title">Señal principal</div>'
                f'<div class="da-section-copy">Mejor variante actual: <b>{best_variant.get("variant") or "-"}</b><br>'
                f'Variante con más riesgo: <b>{worst_variant.get("variant") or "-"}</b></div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="da-panel"><div class="da-section-title">Señal principal</div>'
                '<div class="da-section-copy">Aún faltan logs con ground truth para una lectura comparativa completa.</div></div>',
                unsafe_allow_html=True,
            )

    dl_1, dl_2, dl_3, dl_4 = st.columns(4)
    with dl_1:
        st.download_button(
            "Descargar logs CSV",
            data=_rows_to_csv_bytes(rows=filtered_log_rows),
            file_name="docaudit_metricas_logs.csv",
            mime="text/csv",
            key="analytics_download_logs_csv",
            disabled=not filtered_log_rows,
        )
    with dl_2:
        st.download_button(
            "Descargar errores CSV",
            data=_rows_to_csv_bytes(rows=log_issue_summary_rows),
            file_name="docaudit_errores_frecuentes.csv",
            mime="text/csv",
            key="analytics_download_error_csv",
            disabled=not log_issue_summary_rows,
        )
    with dl_3:
        st.download_button(
            "Descargar evaluación CSV",
            data=_rows_to_csv_bytes(rows=filtered_eval_rows),
            file_name="docaudit_evaluacion_formal.csv",
            mime="text/csv",
            key="analytics_download_eval_csv",
            disabled=not filtered_eval_rows,
        )
    with dl_4:
        st.download_button(
            "Descargar variantes CSV",
            data=_rows_to_csv_bytes(rows=eval_by_variant_rows),
            file_name="docaudit_comparacion_variantes.csv",
            mime="text/csv",
            key="analytics_download_variant_csv",
            disabled=not eval_by_variant_rows,
        )

    dashboard_tabs = st.tabs(
        [
            "Resumen Ejecutivo",
            "Rendimiento",
            "Calidad del Dato",
            "Comparación Experimental",
            "Errores y Trazabilidad",
        ]
    )

    with dashboard_tabs[0]:
        executive_chart_left, executive_chart_right = st.columns(2)
        with executive_chart_left:
            st.markdown("**Cobertura por caso de uso**")
            _render_chart_from_rows(
                rows=by_schema_rows,
                index_key="schema_name",
                value_keys=["documentos", "validos"],
                empty_message="No hay datos por caso de uso para mostrar.",
            )
        with executive_chart_right:
            st.markdown("**Comportamiento por método**")
            _render_chart_from_rows(
                rows=by_method_rows,
                index_key="method",
                value_keys=["tasa_validez", "score_medio"],
                empty_message="No hay datos por método para mostrar.",
            )
        st.markdown("**Resumen por caso de uso**")
        _render_dataframe_or_message(rows=by_schema_rows, empty_message="No hay filas para el resumen por caso de uso.")
        st.markdown("**Resumen por método**")
        _render_dataframe_or_message(rows=by_method_rows, empty_message="No hay filas para el resumen por método.")
        if filtered_eval_rows:
            st.markdown("**Resumen formal por variante**")
            _render_dataframe_or_message(rows=eval_by_variant_rows, empty_message="No hay filas de evaluación por variante.")
        else:
            st.info("Con los filtros actuales no hay suficiente ground truth asociado para evaluación formal.")

    with dashboard_tabs[1]:
        perf_chart_1, perf_chart_2 = st.columns(2)
        with perf_chart_1:
            st.markdown("**Latencia por variante**")
            _render_chart_from_rows(
                rows=by_variant_rows,
                index_key="variant",
                value_keys=["latencia_media_s"],
                empty_message="No hay latencias por variante para mostrar.",
            )
        with perf_chart_2:
            st.markdown("**RAM media por tipo documental**")
            _render_chart_from_rows(
                rows=by_file_type_rows,
                index_key="file_type",
                value_keys=["ram_media_mb"],
                empty_message="No hay métricas de RAM por tipo documental.",
            )
        detail_perf_left, detail_perf_right = st.columns(2)
        with detail_perf_left:
            st.markdown("**Rendimiento por tipo documental**")
            _render_dataframe_or_message(rows=by_file_type_rows, empty_message="No hay resumen por tipo documental.")
        with detail_perf_right:
            st.markdown("**Rendimiento por variante**")
            _render_dataframe_or_message(rows=by_variant_rows, empty_message="No hay resumen por variante.")
        st.markdown("**Logs recientes**")
        _render_dataframe_or_message(rows=filtered_log_rows[:50], empty_message="No hay logs recientes para mostrar.")

    with dashboard_tabs[2]:
        if filtered_eval_rows:
            quality_chart_1, quality_chart_2 = st.columns(2)
            with quality_chart_1:
                st.markdown("**F1 por variante**")
                _render_chart_from_rows(
                    rows=eval_by_variant_rows,
                    index_key="variant",
                    value_keys=["field_level_f1_medio", "exact_match_rate"],
                    empty_message="No hay datos de F1 por variante.",
                )
            with quality_chart_2:
                st.markdown("**F1 por caso de uso**")
                _render_chart_from_rows(
                    rows=eval_by_schema_rows,
                    index_key="schema_name",
                    value_keys=["field_level_f1_medio", "match_rate_medio"],
                    empty_message="No hay datos de evaluación por caso de uso.",
                )
            st.markdown("**Evaluación formal por método**")
            _render_dataframe_or_message(rows=eval_by_method_rows, empty_message="No hay evaluación por método.")
            st.markdown("**Evaluación formal por tipo documental**")
            _render_dataframe_or_message(rows=eval_by_file_type_rows, empty_message="No hay evaluación por tipo documental.")
            st.markdown("**Evaluación formal por esquema**")
            _render_dataframe_or_message(rows=eval_by_schema_rows, empty_message="No hay evaluación por esquema.")
            st.markdown("**Campos con más error**")
            if field_error_rows:
                st.dataframe(field_error_rows, use_container_width=True, hide_index=True)
            else:
                st.success("No se detectaron errores de campo en la evaluación filtrada.")
        else:
            st.info("No hay datos de evaluación formal para los filtros actuales.")

    with dashboard_tabs[3]:
        if eval_by_variant_rows:
            st.markdown("**Comparación entre variantes documentales**")
            _render_chart_from_rows(
                rows=eval_by_variant_rows,
                index_key="variant",
                value_keys=["field_level_f1_medio", "exact_match_rate", "match_rate_medio"],
                empty_message="No hay suficientes datos para comparar variantes.",
            )
            st.dataframe(eval_by_variant_rows, use_container_width=True, hide_index=True)
            preferred_order = ["native_pdf", "scanned_blurry_pdf", "image_photo", "image_handwritten"]
            ordered_variant_rows = sorted(
                eval_by_variant_rows,
                key=lambda row: preferred_order.index(str(row.get("variant"))) if str(row.get("variant")) in preferred_order else 999,
            )
            if ordered_variant_rows:
                best_variant = max(
                    ordered_variant_rows,
                    key=lambda row: (
                        float(row.get("field_level_f1_medio") or -1),
                        float(row.get("exact_match_rate") or -1),
                    ),
                )
                worst_variant = min(
                    ordered_variant_rows,
                    key=lambda row: (
                        float(row.get("field_level_f1_medio") or 999),
                        float(row.get("exact_match_rate") or 999),
                    ),
                )
                c_cmp_1, c_cmp_2, c_cmp_3 = st.columns(3)
                c_cmp_1.metric("Mejor variante por F1", str(best_variant.get("variant") or "-"))
                c_cmp_2.metric("Variante más débil por F1", str(worst_variant.get("variant") or "-"))
                c_cmp_3.metric(
                    "Brecha F1",
                    _format_percent(
                        float(best_variant.get("field_level_f1_medio") or 0) - float(worst_variant.get("field_level_f1_medio") or 0)
                    ),
                )
        else:
            st.info("Todavía no hay suficientes logs evaluados por variante.")

    with dashboard_tabs[4]:
        trace_left, trace_right = st.columns([1.4, 1.2])
        with trace_left:
            st.markdown("**Errores frecuentes**")
            if log_issue_summary_rows:
                st.dataframe(log_issue_summary_rows, use_container_width=True, hide_index=True)
            else:
                st.success("No hay incidencias acumuladas con los filtros seleccionados.")
        with trace_right:
            st.markdown("**Incidencias por patrón**")
            top_issue_rows = log_issue_summary_rows[:8]
            if top_issue_rows:
                _render_chart_from_rows(
                    rows=top_issue_rows,
                    index_key="patron",
                    value_keys=["frecuencia"],
                    empty_message="No hay incidencias para representar.",
                )
            else:
                st.info("No hay incidencias para representar.")
        st.markdown("**Trazabilidad de evaluación**")
        if log_issue_summary_rows:
            _render_json(label="Ver incidencias resumidas (JSON)", payload=log_issue_summary_rows[:50])
        if filtered_eval_rows:
            _render_json(label="Ver evaluación documental detallada", payload=filtered_eval_rows[:100])

if app_section == "Centro analítico":
    _render_analytics_dashboard()
    st.stop()


with st.sidebar:
    st.subheader("Ejecución")
    use_vision = st.checkbox("Si el PDF no tiene texto, intentar con OCR", value=True)
    schema_choice = st.selectbox(
        "Caso de uso",
        options=["Auto", "credito_hipotecario", "auditoria_fiscal", "kyc_onboarding"],
        index=0,
    )
    st.divider()
    st.subheader("Carpeta")
    if "folder_path" not in st.session_state:
        st.session_state["folder_path"] = ""
    if "folder_picker_error" not in st.session_state:
        st.session_state["folder_picker_error"] = None
    folder_path = st.text_input(
        "Ruta de carpeta (PDFs e imagenes)",
        key="folder_path",
        help="Puedes escribir la ruta manualmente o usar el boton 'Buscar carpeta'.",
    )
    col_folder_1, col_folder_2 = st.columns([1, 1])
    with col_folder_1:
        st.button("Buscar carpeta", on_click=_browse_folder_callback, use_container_width=True)
    with col_folder_2:
        load_folder_clicked = st.button("Cargar documentos desde carpeta", use_container_width=True)
    picker_error = st.session_state.get("folder_picker_error")
    if isinstance(picker_error, str) and picker_error.strip():
        st.error(picker_error)
    st.divider()
    st.subheader("Esquemas YAML")
    st.caption("Edición avanzada. Recomendado solo si necesitas ajustar campos/reglas.")
    schemas_dir = PROJECT_ROOT / "schemas"
    schema_files = sorted(schemas_dir.glob("*.yaml"))
    schema_names = [p.name for p in schema_files]
    if not schema_names:
        st.warning("No se encontraron archivos .yaml en schemas/.")
        selected_schema_path = None
    else:
        selected_schema_name = st.selectbox("Esquema", options=schema_names, index=0)
        selected_schema_path = schemas_dir / selected_schema_name
    if selected_schema_path and selected_schema_path.exists():
        original_yaml = selected_schema_path.read_text(encoding="utf-8")
        with st.expander("Abrir editor YAML", expanded=False):
            edited_yaml = st.text_area("Contenido YAML", value=original_yaml, height=260)
            col_v, col_s = st.columns(2)
            with col_v:
                if st.button("Validar YAML"):
                    try:
                        raw = yaml.safe_load(edited_yaml)
                        if isinstance(raw, dict) and "caso_uso" in raw:
                            schema_loader._load_new_format(raw)
                        else:
                            DocSchema.model_validate(raw)
                        st.success("YAML válido.")
                    except Exception as e:
                        st.error("YAML inválido o no compatible con el schema loader.")
                        st.exception(e)
            with col_s:
                if st.button("Guardar YAML"):
                    selected_schema_path.write_text(edited_yaml, encoding="utf-8")
                    st.success(f"Guardado: {selected_schema_path.name}")

uploaded_documents = st.file_uploader(
    "Subir documentos",
    type=SUPPORTED_UPLOAD_TYPES,
    accept_multiple_files=True,
    help="Puedes subir PDF, PNG, JPG, JPEG o WEBP.",
)

if load_folder_clicked:
    docs_from_folder, folder_error = _load_documents_from_folder(folder_path.strip())
    if folder_error:
        st.session_state["folder_load_error"] = folder_error
        st.session_state["folder_loaded_items"] = []
    else:
        st.session_state["folder_load_error"] = None
        st.session_state["folder_loaded_items"] = docs_from_folder

pdf_items: list[dict[str, Any]] = []
folder_error = st.session_state.get("folder_load_error")
if isinstance(folder_error, str) and folder_error.strip():
    st.error(folder_error)
if uploaded_documents:
    pdf_items = _load_documents_from_uploads(list(uploaded_documents))
elif isinstance(st.session_state.get("folder_loaded_items"), list) and st.session_state.get("folder_loaded_items"):
    pdf_items = st.session_state.get("folder_loaded_items", [])

loaded_docs_count = len(pdf_items)
loaded_image_count = len([item for item in pdf_items if str(item.get("input_kind") or "") == "image"])
loaded_pdf_count = len([item for item in pdf_items if str(item.get("input_kind") or "") == "pdf"])
batch_mode_active = loaded_docs_count >= 2

if pdf_items:
    expediente_texts: list[str] = []
    expediente_pages: list[list[str] | None] = []
    expediente_doc_ids: list[str] = []
    expediente_meta: list[dict[str, Any]] = []
    extraction_cache = st.session_state.setdefault("document_extraction_cache", {})
    load_progress = None
    load_status = None

    for i, item in enumerate(pdf_items):
        name = str(item.get("name") or f"documento_{i+1}.pdf")
        file_bytes = item.get("bytes") or b""
        source = str(item.get("source") or "-")
        path = item.get("path")
        input_kind = str(item.get("input_kind") or "pdf")

        doc_id = hashlib.sha256(file_bytes).hexdigest() if isinstance(file_bytes, (bytes, bytearray)) else ""
        cache_key = _build_extraction_cache_key(doc_id=doc_id or name, input_kind=input_kind, use_vision=use_vision)
        cached_extraction = extraction_cache.get(cache_key) if isinstance(extraction_cache, dict) else None

        if isinstance(cached_extraction, dict):
            text = str(cached_extraction.get("text") or "")
            pages = cached_extraction.get("page_texts") if isinstance(cached_extraction.get("page_texts"), list) else None
            pages_n = cached_extraction.get("pages")
            method = cached_extraction.get("method")
        else:
            if load_progress is None:
                load_progress = st.progress(0.0, text="Preparando documentos cargados...")
                load_status = st.empty()
            if load_status is not None:
                load_status.caption(f"Preparando documento {i+1} de {loaded_docs_count}: {name}")

        if not isinstance(cached_extraction, dict) and input_kind == "image":
            image_ocr_fn = _resolve_document_loader_attr("extract_text_from_image_bytes")
            if image_ocr_fn is None:
                st.error(_document_loader_error_message("No esta disponible la funcion de OCR para imagenes."))
                continue
            else:
                with st.spinner(f"Extrayendo texto desde imagen — {name}..."):
                    extracted = image_ocr_fn(file_bytes)
                text = extracted.get("text") or ""
                pages = extracted.get("page_texts") if isinstance(extracted.get("page_texts"), list) else None
                pages_n = extracted.get("pages")
                method = extracted.get("method") or "easyocr"
        elif not isinstance(cached_extraction, dict):
            pdf_extract_fn = _resolve_document_loader_attr("extract_text_from_pdf_bytes")
            if pdf_extract_fn is None:
                st.error(_document_loader_error_message("No esta disponible la funcion de lectura de PDF."))
                continue
            extracted = pdf_extract_fn(file_bytes)
            text = extracted.get("text") or ""
            pages = extracted.get("page_texts") if isinstance(extracted.get("page_texts"), list) else None
            pages_n = extracted.get("pages")
            method = extracted.get("method")

            if not text and use_vision:
                scanned_pdf_ocr_fn = _resolve_document_loader_attr("extract_text_from_scanned_pdf_bytes")
                if scanned_pdf_ocr_fn is None:
                    st.error(_document_loader_error_message("No está disponible la función de OCR."))
                else:
                    with st.spinner(f"Extrayendo texto con OCR — {name}..."):
                        extracted_v = scanned_pdf_ocr_fn(file_bytes)
                    text = extracted_v.get("text") or ""
                    pages = extracted_v.get("page_texts") if isinstance(extracted_v.get("page_texts"), list) else pages
                    pages_n = extracted_v.get("pages") or pages_n
                    method = extracted_v.get("method") or "vision"

        if not isinstance(cached_extraction, dict) and isinstance(extraction_cache, dict):
            extraction_cache[cache_key] = {
                "text": text,
                "page_texts": pages,
                "pages": pages_n,
                "method": method,
            }

        expediente_texts.append(text)
        expediente_pages.append(pages)
        expediente_doc_ids.append(doc_id)
        expediente_meta.append(
            {
                "doc_index": i + 1,
                "documento": name,
                "origen": source,
                "tipo_entrada": input_kind,
                "folios": pages_n,
                "metodo": method,
                "caracteres": len(text or ""),
                "ruta": path,
            }
        )
        if load_progress is not None:
            load_progress.progress((i + 1) / loaded_docs_count, text=f"Documentos preparados: {i + 1}/{loaded_docs_count}")

    if load_status is not None:
        load_status.empty()
    if load_progress is not None:
        load_progress.empty()

    st.session_state["expediente_texts"] = expediente_texts
    st.session_state["expediente_pages"] = expediente_pages
    st.session_state["expediente_doc_ids"] = expediente_doc_ids
    st.session_state["expediente_meta"] = expediente_meta
    st.session_state["uploaded_names"] = [m["documento"] for m in expediente_meta]
    if len(expediente_texts) == 1:
        st.session_state["input_pages"] = expediente_pages[0]
        st.session_state["doc_id"] = expediente_doc_ids[0]

    st.subheader("Documentos cargados")
    st.caption("Paso 2 de 3: revisa el lote cargado antes de ejecutar el análisis.")
    c_load_1, c_load_2, c_load_3, c_load_4 = st.columns(4)
    c_load_1.metric("Documentos", str(loaded_docs_count))
    c_load_2.metric("PDF", str(loaded_pdf_count))
    c_load_3.metric("Imágenes", str(loaded_image_count))
    c_load_4.metric("Modo", "Lote" if batch_mode_active else "Individual")
    if batch_mode_active:
        st.info(
            "La app procesará todos los documentos cargados en una sola ejecución y conservará resultados individuales por archivo."
        )
    else:
        st.info("La app procesará el documento cargado y mostrará su resultado individual.")
    st.dataframe(expediente_meta, use_container_width=True, hide_index=True)
    _render_loaded_documents_preview(pdf_items=pdf_items)
    st.session_state["input_text"] = "\n\n".join(
        f"=== Documento {i+1}: {expediente_meta[i]['documento']} ===\n{expediente_texts[i]}"
        for i in range(len(expediente_texts))
    )

st.subheader("Preparación y ejecución")
with st.container(border=True):
    st.caption("Paso 3 de 3: lanza el análisis del documento actual o de todo el lote cargado.")

    panel_left, panel_right = st.columns([2, 1])
    with panel_left:
        st.markdown("### Panel de ejecución")
        if batch_mode_active:
            st.success(
                f"Modo lote activo: al pulsar el botón se analizarán los {loaded_docs_count} documentos cargados, con resultados globales e individuales."
            )
        elif loaded_docs_count == 1:
            st.info("Modo individual activo: al pulsar el botón se analizará el documento cargado.")
        else:
            st.info("Puedes pegar texto manualmente o cargar uno o varios documentos antes de ejecutar el análisis.")

    with panel_right:
        c_panel_1, c_panel_2 = st.columns(2)
        c_panel_1.metric("Modo", "Lote" if batch_mode_active else "Individual")
        c_panel_2.metric("Documentos", str(loaded_docs_count))

    c_flow_1, c_flow_2, c_flow_3 = st.columns(3)
    c_flow_1.markdown("**1. Cargar**")
    c_flow_1.caption("Sube archivos o selecciona una carpeta.")
    c_flow_2.markdown("**2. Revisar**")
    c_flow_2.caption("Comprueba el lote y el texto extraído.")
    c_flow_3.markdown("**3. Ejecutar**")
    c_flow_3.caption("Procesa uno o todos y revisa resultados individuales.")

text_area_label = "Texto consolidado para análisis" if loaded_docs_count else "Texto del documento"
text_area_help = (
    "Este bloque reúne el texto de todos los documentos cargados para la ejecución por lote."
    if batch_mode_active
    else None
)
text = st.text_area(
    text_area_label,
    key="input_text",
    height=300,
    placeholder="Pega aquí el texto del documento o sube PDF, PNG, JPG, JPEG o WEBP arriba.",
    label_visibility="visible",
    help=text_area_help,
)
run_button_label = "Ejecutar análisis"
if batch_mode_active:
    run_button_label = f"Ejecutar análisis de los {loaded_docs_count} documentos"
elif loaded_docs_count == 1:
    run_button_label = "Ejecutar análisis del documento"
button_col, help_col = st.columns([2, 1])
with button_col:
    run_clicked = st.button(run_button_label, type="primary", disabled=not text.strip(), use_container_width=True)
with help_col:
    if text.strip():
        st.caption("Todo listo para ejecutar.")
    else:
        st.caption("Carga documentos o pega texto para habilitar la ejecución.")

if run_clicked:
    try:
        # Iniciar medición de tiempo y RAM
        start_time = time.perf_counter()
        process = psutil.Process() if psutil is not None else None
        ram_start = process.memory_info().rss / (1024 * 1024) if process is not None else 0.0
        
        with st.spinner("Ejecutando pipeline (extracción → normalización → validación → auditoría)..."):
            expediente_texts = st.session_state.get("expediente_texts")
            uploaded_names = st.session_state.get("uploaded_names", [])
            expediente_meta = st.session_state.get("expediente_meta", [])
            
            if isinstance(expediente_texts, list) and len(expediente_texts) >= 2:
                pages_by_doc = st.session_state.get("expediente_pages")
                doc_ids = st.session_state.get("expediente_doc_ids")
                result = run_expediente(
                    [str(t) for t in expediente_texts],
                    pages_by_doc=pages_by_doc if isinstance(pages_by_doc, list) else None,
                    doc_ids=doc_ids if isinstance(doc_ids, list) else None,
                    schema_name=None if schema_choice == "Auto" else schema_choice,
                )
            else:
                pages = st.session_state.get("input_pages")
                doc_id = st.session_state.get("doc_id")
                result = run_pipeline(
                    text,
                    pages=pages if isinstance(pages, list) else None,
                    doc_id=doc_id if isinstance(doc_id, str) else None,
                    schema_name=None if schema_choice == "Auto" else schema_choice,
                )
        
        # Calcular tiempo y RAM final
        end_time = time.perf_counter()
        ram_end = process.memory_info().rss / (1024 * 1024) if process is not None else ram_start
        processing_time = end_time - start_time
        ram_used = ram_end - ram_start
        
        # Guardar registros
        if "documents" in result:
            # Si es un expediente con múltiples documentos
            for i, doc_result in enumerate(result.get("documents", [])):
                doc_index = doc_result.get("doc_index", i)
                file_name = uploaded_names[i] if i < len(uploaded_names) else f"document_{doc_index}.pdf"
                doc_meta = expediente_meta[i] if i < len(expediente_meta) else {}
                method = doc_meta.get("metodo", "pypdf")
                input_kind = doc_meta.get("tipo_entrada", "pdf")
                
                # Determinar tipo de archivo
                if input_kind == "image":
                    file_type = "image"
                elif method == "easyocr":
                    file_type = "scanned_pdf"
                elif method == "vision" or method == "ollama_vision":
                    file_type = "scanned_pdf"
                else:
                    file_type = "native_pdf"
                
                extracted_fields = doc_result.get("extracted", {})
                save_execution_log(
                    file_name=file_name,
                    file_type=file_type,
                    method=method,
                    processing_time=processing_time / len(result.get("documents", [1])),  # Dividir tiempo por documento
                    ram_used_mb=ram_used / len(result.get("documents", [1])),
                    extracted_fields=extracted_fields,
                    full_result=doc_result
                )
        else:
            # Si es un solo documento
            file_name = uploaded_names[0] if uploaded_names else "document_1.pdf"
            doc_meta = expediente_meta[0] if expediente_meta else {}
            method = doc_meta.get("metodo", "pypdf")
            input_kind = doc_meta.get("tipo_entrada", "pdf")
            
            # Determinar tipo de archivo
            if input_kind == "image":
                file_type = "image"
            elif method == "easyocr":
                file_type = "scanned_pdf"
            elif method == "vision" or method == "ollama_vision":
                file_type = "scanned_pdf"
            else:
                file_type = "native_pdf"
            
            extracted_fields = result.get("extracted", {})
            save_execution_log(
                file_name=file_name,
                file_type=file_type,
                method=method,
                processing_time=processing_time,
                ram_used_mb=ram_used,
                extracted_fields=extracted_fields,
                full_result=result
            )
        
        # Mostrar mensaje de éxito de logging
        st.success(f"✅ Registro guardado en: `{LOGS_FOLDER}")
    except Exception as e:
        st.error("Falló la extracción: el modelo devolvió una respuesta no válida o hubo un problema de conexión.")
        st.exception(e)
        st.stop()

    st.session_state["last_result"] = result

result = st.session_state.get("last_result")
if isinstance(result, dict) and result:
    st.subheader("Resultados")
    tab_labels = ["Resumen", "Extracción", "Normalización", "Validación", "Auditoría"]
    show_gt = False
    if "documents" not in result:
        uploaded_names = st.session_state.get("uploaded_names")
        show_gt = isinstance(uploaded_names, list) and len(uploaded_names) == 1 and isinstance(uploaded_names[0], str)
    if show_gt:
        tab_labels.append("Evaluación")
    if "documents" in result:
        tab_labels.insert(0, "Expediente")
    tab_labels.append("JSON")
    tabs = st.tabs(tab_labels)
    idx = 0
    if "documents" in result:
        tab_expediente = tabs[idx]
        idx += 1
    tab_summary = tabs[idx]
    tab_extracted = tabs[idx + 1]
    tab_normalization = tabs[idx + 2]
    tab_validation = tabs[idx + 3]
    tab_audit = tabs[idx + 4]
    tab_eval = tabs[idx + 5] if show_gt else None
    tab_json = tabs[idx + 6] if show_gt else tabs[idx + 5]

    if "documents" in result:
        with tab_expediente:
            st.write("Resumen del expediente")
            meta = st.session_state.get("expediente_meta")
            docs = result.get("documents", [])
            rows: list[dict[str, Any]] = []
            if isinstance(docs, list):
                for d in docs:
                    if not isinstance(d, dict):
                        continue
                    idx_doc = d.get("doc_index")
                    validation = d.get("validation") if isinstance(d.get("validation"), dict) else {}
                    report = d.get("report") if isinstance(d.get("report"), dict) else {}
                    report_json = report.get("json") if isinstance(report.get("json"), dict) else {}
                    rules = report_json.get("decision_rules") if isinstance(report_json.get("decision_rules"), list) else []
                    reglas_total = len(rules)
                    reglas_ok = len([r for r in rules if isinstance(r, dict) and r.get("cumple") is True])
                    estado, motivo_estado = _document_status_info(doc_result=d)
                    rows.append(
                        {
                            "doc_index": (idx_doc + 1) if isinstance(idx_doc, int) else idx_doc,
                            "estado": estado,
                            "estado_resumen": (
                                "OK - Valido"
                                if estado == "OK"
                                else ("CRITICO - Regla critica" if estado == "CRITICO" else "REVISAR - Revision manual")
                            ),
                            "motivo_estado": motivo_estado,
                            "document_type": d.get("document_type"),
                            "valido": validation.get("valid") if isinstance(validation, dict) else None,
                            "incidencias": len(validation.get("issues", []) or []) if isinstance(validation, dict) else None,
                            "score_confianza": report_json.get("score_confianza") if isinstance(report_json, dict) else None,
                            "reglas_ok": reglas_ok,
                            "reglas_total": reglas_total,
                        }
                    )
            c_exp_1, c_exp_2, c_exp_3 = st.columns([1, 1, 1])
            with c_exp_1:
                st.download_button(
                    "Descargar expediente JSON",
                    data=json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name="docaudit_expediente.json",
                    mime="application/json",
                    key="download_expediente_json",
                )
            with c_exp_2:
                st.download_button(
                    "Descargar resumen CSV",
                    data=_rows_to_csv_bytes(rows=rows),
                    file_name="docaudit_expediente_resumen.csv",
                    mime="text/csv",
                    key="download_expediente_csv",
                    disabled=not rows,
                )

            total_docs = len(rows)
            ok_count = len([r for r in rows if r.get("estado") == "OK"])
            revisar_count = len([r for r in rows if r.get("estado") == "REVISAR"])
            critico_count = len([r for r in rows if r.get("estado") == "CRITICO"])
            ok_ratio = (ok_count / total_docs) if total_docs else 0.0
            revisar_ratio = (revisar_count / total_docs) if total_docs else 0.0
            critico_ratio = (critico_count / total_docs) if total_docs else 0.0

            quick_filter = st.radio(
                "Vista rápida",
                options=["Todos", "Solo OK", "Solo revisar", "Solo críticos"],
                horizontal=True,
                key="expediente_quick_filter",
            )
            quick_filter_map = {
                "Todos": ["OK", "REVISAR", "CRITICO"],
                "Solo OK": ["OK"],
                "Solo revisar": ["REVISAR"],
                "Solo críticos": ["CRITICO"],
            }
            default_statuses = quick_filter_map.get(quick_filter, ["OK", "REVISAR", "CRITICO"])
            with st.expander("Filtro avanzado", expanded=False):
                selected_statuses = st.multiselect(
                    "Estados visibles",
                    options=["OK", "REVISAR", "CRITICO"],
                    default=default_statuses,
                )
            filtered_rows = [
                row for row in rows
                if not selected_statuses or str(row.get("estado") or "") in selected_statuses
            ]
            issue_rows = _collect_issue_rows(docs=[d for d in docs if isinstance(d, dict)], meta=meta if isinstance(meta, list) else None)
            issue_summary_rows = _summarize_issue_patterns(issue_rows=issue_rows)
            autocorrection_rows = _collect_autocorrection_rows(
                docs=[d for d in docs if isinstance(d, dict)],
                meta=meta if isinstance(meta, list) else None,
            )

            c_stat_1, c_stat_2, c_stat_3, c_stat_4 = st.columns(4)
            c_stat_1.metric("Total docs", str(total_docs))
            c_stat_2.metric("OK", str(ok_count), delta=f"{ok_ratio:.0%}")
            c_stat_3.metric("Revisar", str(revisar_count), delta=f"{revisar_ratio:.0%}")
            c_stat_4.metric("Crítico", str(critico_count), delta=f"{critico_ratio:.0%}")
            st.caption("Estados: OK = válido, REVISAR = requiere revisión humana, CRITICO = incumplimiento grave o regla crítica.")
            st.progress(ok_ratio, text=f"Porcentaje del lote en estado OK: {ok_ratio:.0%}")

            c_ana_1, c_ana_2, c_ana_3 = st.columns(3)
            c_ana_1.metric("Incidencias totales", str(len(issue_rows)))
            c_ana_2.metric("Patrones de error", str(len(issue_summary_rows)))
            c_ana_3.metric("Autocorrecciones", str(len(autocorrection_rows)))

            with st.expander("Analítica del expediente", expanded=False):
                c_dl_1, c_dl_2 = st.columns(2)
                with c_dl_1:
                    st.download_button(
                        "Descargar incidencias CSV",
                        data=_rows_to_csv_bytes(rows=issue_rows),
                        file_name="docaudit_incidencias.csv",
                        mime="text/csv",
                        key="download_issues_csv",
                        disabled=not issue_rows,
                    )
                with c_dl_2:
                    st.download_button(
                        "Descargar autocorrecciones CSV",
                        data=_rows_to_csv_bytes(rows=autocorrection_rows),
                        file_name="docaudit_autocorrecciones.csv",
                        mime="text/csv",
                        key="download_autocorrections_csv",
                        disabled=not autocorrection_rows,
                    )

                st.write("Errores frecuentes")
                if issue_summary_rows:
                    st.dataframe(issue_summary_rows, use_container_width=True, hide_index=True)
                else:
                    st.success("No se detectaron incidencias en el expediente.")

                st.write("Autocorrecciones aplicadas")
                if autocorrection_rows:
                    st.dataframe(autocorrection_rows, use_container_width=True, hide_index=True)
                else:
                    st.info("No se aplicaron autocorrecciones en este expediente.")

            filtered_rows_display = [
                {
                    **row,
                    "estado": row.get("estado_resumen") or row.get("estado"),
                }
                for row in filtered_rows
            ]
            st.dataframe(filtered_rows_display, use_container_width=True, hide_index=True)
            if isinstance(meta, list) and meta:
                st.write("Archivos y folios")
                st.dataframe(meta, use_container_width=True, hide_index=True)

            filtered_docs: list[dict[str, Any]] = []
            filtered_meta: list[dict[str, Any]] = []
            source_texts = st.session_state.get("expediente_texts")
            if isinstance(docs, list):
                for idx_doc, d in enumerate(docs):
                    if not isinstance(d, dict):
                        continue
                    estado_doc, _ = _document_status_info(doc_result=d)
                    if selected_statuses and estado_doc not in selected_statuses:
                        continue
                    filtered_docs.append(d)
                    if isinstance(meta, list) and idx_doc < len(meta) and isinstance(meta[idx_doc], dict):
                        filtered_meta.append(meta[idx_doc])

            if filtered_docs:
                st.subheader("Explorador de documentos del expediente")
                st.caption("Selecciona un archivo para revisar su extracción, validación y auditoría individual.")
                opciones = _build_doc_option_labels(
                    docs=filtered_docs,
                    meta=filtered_meta if filtered_meta else (meta if isinstance(meta, list) else None),
                )
                sel = st.selectbox("Ver detalle por documento", options=opciones, key="expediente_result_selector")
                sel_idx = opciones.index(sel) if sel in opciones else 0
                chosen = filtered_docs[sel_idx] if sel_idx < len(filtered_docs) else None
                if isinstance(chosen, dict):
                    estado_doc, motivo_doc = _document_status_info(doc_result=chosen)
                    if filtered_meta and sel_idx < len(filtered_meta) and isinstance(filtered_meta[sel_idx], dict):
                        chosen_meta = filtered_meta[sel_idx]
                        st.caption(
                            f"Archivo: {chosen_meta.get('documento') or '-'} | "
                            f"Origen: {chosen_meta.get('origen') or '-'} | "
                            f"Método: {chosen_meta.get('metodo') or '-'}"
                        )
                    c_doc_1, c_doc_2 = st.columns([2, 1])
                    with c_doc_1:
                        _render_document_status_notice(status=estado_doc, reason=motivo_doc)
                    with c_doc_2:
                        raw_index = chosen.get("doc_index")
                        export_idx = (raw_index + 1) if isinstance(raw_index, int) else (sel_idx + 1)
                        st.download_button(
                            "Descargar documento JSON",
                            data=json.dumps(chosen, ensure_ascii=False, indent=2).encode("utf-8"),
                            file_name=f"docaudit_documento_{export_idx}.json",
                            mime="application/json",
                            key=f"download_doc_{sel_idx}",
                        )
                    chosen_text = ""
                    if isinstance(source_texts, list):
                        if isinstance(raw_index, int) and 0 <= raw_index < len(source_texts):
                            chosen_text = str(source_texts[raw_index] or "")
                        elif 0 <= sel_idx < len(source_texts):
                            chosen_text = str(source_texts[sel_idx] or "")
                    _render_document_result_detail(
                        chosen=chosen,
                        selection_key=f"doc_{sel_idx}",
                        source_text=chosen_text,
                    )
            elif rows:
                st.info("No hay documentos que coincidan con el filtro seleccionado.")

    with tab_summary:
        report = result.get("report", {}) or {}
        report_json = report.get("json") or {}
        validation = result.get("validation", {}) or {}
        schema_meta = result.get("schema", {}) or {}
        issues = validation.get("issues", []) if isinstance(validation, dict) else []
        decision_rules = report_json.get("decision_rules", []) if isinstance(report_json, dict) else []
        score = report_json.get("score_confianza") if isinstance(report_json, dict) else None

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Caso de uso", str(schema_meta.get("name") or "-"))
        c2.metric("Validez", "Válido" if validation.get("valid") else "Revisar")
        c3.metric("Incidencias", str(len(issues) if isinstance(issues, list) else 0))
        if isinstance(score, (int, float)):
            c4.metric("Confianza", f"{float(score):.0%}")
        else:
            c4.metric("Confianza", "-")

        if isinstance(score, (int, float)):
            _render_score_gauge(score=float(score), key="score_confianza_resumen")

        if isinstance(decision_rules, list) and decision_rules:
            st.write("Reglas de decisión (resumen)")
            st.dataframe(
                [
                    {
                        "id": r.get("id"),
                        "severidad": r.get("severidad"),
                        "cumple": r.get("cumple"),
                        "descripcion": r.get("descripcion"),
                    }
                    for r in decision_rules
                    if isinstance(r, dict)
                ],
                use_container_width=True,
                hide_index=True,
            )

    with tab_extracted:
        schema_name = (result.get("schema") or {}).get("name") if isinstance(result.get("schema"), dict) else None
        schema_obj = None
        if isinstance(schema_name, str) and schema_name.strip():
            try:
                schema_obj = load_schema(PROJECT_ROOT / "schemas" / f"{schema_name}.yaml")
            except Exception:
                schema_obj = None

        extracted = result.get("extracted", {}) if isinstance(result, dict) else {}
        rows = _to_table_rows(extracted=extracted if isinstance(extracted, dict) else {}, schema=schema_obj)
        st.write("Campos extraídos (normalizados)")
        st.dataframe(rows, use_container_width=True, hide_index=True)

        _render_json(label="Ver salida cruda de extracción (extracted_raw)", payload=result.get("extracted_raw", {}))
        _render_json(label="Ver metadatos de schema", payload=result.get("schema", {}))

    with tab_normalization:
        normalization = result.get("normalization", {})
        changes = normalization.get("changes", [])
        st.write("Cambios aplicados")
        if isinstance(changes, list) and changes:
            st.dataframe(changes, use_container_width=True, hide_index=True)
        else:
            st.info("No se aplicaron cambios porque la salida ya venía en formatos consistentes.")
        _render_json(label="Ver JSON normalizado", payload=normalization.get("normalized", {}))

    with tab_validation:
        validation = result.get("validation", {})
        st.write(f"Estado: {'Válido' if validation.get('valid') else 'Revisar'}")
        issues = validation.get("issues", [])
        if isinstance(issues, list):
            _render_issues_table(issues=issues)
        else:
            st.warning("Formato de incidencias inesperado.")
        _render_json(label="Ver validación (JSON)", payload=validation)

    with tab_audit:
        report = result.get("report", {}) or {}
        report_json = report.get("json") or {}
        report_md = report.get("markdown") or ""
        score = report_json.get("score_confianza")

        if report_md:
            with st.expander("Ver informe (Markdown)", expanded=True):
                st.markdown(report_md)

        decision_rules = report_json.get("decision_rules", []) if isinstance(report_json, dict) else []
        if isinstance(decision_rules, list) and decision_rules:
            st.write("Reglas evaluadas")
            st.dataframe(
                [
                    {
                        "id": r.get("id"),
                        "severidad": r.get("severidad"),
                        "cumple": r.get("cumple"),
                        "descripcion": r.get("descripcion"),
                        "expresion": r.get("expresion"),
                    }
                    for r in decision_rules
                    if isinstance(r, dict)
                ],
                use_container_width=True,
                hide_index=True,
            )

        campos = report_json.get("campos") if isinstance(report_json, dict) else None
        if isinstance(campos, dict) and campos:
            evidencias_rows: list[dict[str, Any]] = []
            for k, v in campos.items():
                if not isinstance(v, dict):
                    continue
                evidencias_rows.append(
                    {
                        "campo": k,
                        "valor": v.get("valor"),
                        "confianza": v.get("confianza"),
                        "pagina": v.get("pagina"),
                    }
                )
            if evidencias_rows:
                st.write("Evidencias por campo")
                st.dataframe(evidencias_rows, use_container_width=True, hide_index=True)
                field_options = [r["campo"] for r in evidencias_rows]
                sel = st.selectbox("Ver detalle de evidencia", options=field_options)
                chosen_full = campos.get(sel) if isinstance(sel, str) else None
                if isinstance(chosen_full, dict):
                    evidencia = chosen_full.get("evidencia_textual")
                    if isinstance(evidencia, str) and evidencia.strip():
                        st.markdown("**Evidencia textual**")
                        st.markdown(f"<mark>{evidencia}</mark>", unsafe_allow_html=True)

        _render_json(label="Ver informe (JSON)", payload=report_json)

    if tab_eval is not None:
        with tab_eval:
            st.write("Comparación contra ground truth del corpus (si existe).")
            uploaded_names = st.session_state.get("uploaded_names") or []
            pdf_filename = uploaded_names[0] if uploaded_names else None
            render_ground_truth_evaluation(result=result, pdf_filename=pdf_filename, project_root=PROJECT_ROOT)

    with tab_json:
        st.write("Resultado completo (para depuración / trazabilidad).")
        st.download_button(
            "Descargar JSON",
            data=json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="docaudit_resultado.json",
            mime="application/json",
        )
        st.json(result)
