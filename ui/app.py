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
import sys
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.components import render_ground_truth_evaluation  # noqa: E402

try:  # noqa: E402
    from core.document_loader import extract_text_from_pdf_bytes, extract_text_from_scanned_pdf_bytes
except ImportError:  # noqa: E402
    from core.document_loader import extract_text_from_pdf_bytes

    extract_text_from_scanned_pdf_bytes = None
from core.orchestrator import run_expediente, run_pipeline  # noqa: E402
from core.schema_loader import load_schema  # noqa: E402
from core.normalizer import normalize_extracted  # noqa: E402


st.set_page_config(page_title="DocAudit Agent", layout="wide")
st.title("DocAudit Agent")

st.write("Pega texto (o el contenido extraído de un documento) y ejecuta el pipeline: extraer → normalizar → validar.")

use_vision = st.checkbox("Si el PDF no tiene texto, intentar con visión (Qwen2.5-VL)", value=True)

schema_choice = st.selectbox(
    "Caso de uso",
    options=["Auto", "credito_hipotecario", "auditoria_fiscal", "kyc_onboarding"],
    index=0,
)

uploaded_pdfs = st.file_uploader("Subir PDF(s)", type=["pdf"], accept_multiple_files=True)
if uploaded_pdfs:
    expediente_texts: list[str] = []
    expediente_pages: list[list[str] | None] = []
    expediente_doc_ids: list[str] = []
    uploaded_names: list[str] = []
    preview_lines: list[str] = []
    for i, up in enumerate(uploaded_pdfs):
        uploaded_names.append(up.name)
        pdf_bytes = up.read()
        doc_id = hashlib.sha256(pdf_bytes).hexdigest()
        extracted = extract_text_from_pdf_bytes(pdf_bytes)
        text = extracted.get("text") or ""
        pages = extracted.get("page_texts") if isinstance(extracted.get("page_texts"), list) else None
        if not text and use_vision:
            if extract_text_from_scanned_pdf_bytes is None:
                st.error(
                    "No está disponible la función de visión. Detén y vuelve a iniciar Streamlit "
                    "para recargar los módulos, y verifica que tu entorno tenga el código actualizado."
                )
            else:
                with st.spinner(f"Extrayendo texto con visión (Qwen2.5-VL) — {up.name}..."):
                    extracted_v = extract_text_from_scanned_pdf_bytes(pdf_bytes)
                text = extracted_v.get("text") or ""
                pages = extracted_v.get("page_texts") if isinstance(extracted_v.get("page_texts"), list) else pages
        expediente_texts.append(text)
        expediente_pages.append(pages)
        expediente_doc_ids.append(doc_id)
        preview_lines.append(f"- {up.name}: {len(text)} caracteres")

    st.session_state["expediente_texts"] = expediente_texts
    st.session_state["expediente_pages"] = expediente_pages
    st.session_state["expediente_doc_ids"] = expediente_doc_ids
    st.session_state["uploaded_names"] = uploaded_names
    if len(expediente_texts) == 1:
        st.session_state["input_pages"] = expediente_pages[0]
        st.session_state["doc_id"] = expediente_doc_ids[0]
    st.write("Documentos cargados")
    st.write("\n".join(preview_lines))
    st.session_state["input_text"] = "\n\n".join(
        f"=== Documento {i+1}: {up.name} ===\n{expediente_texts[i]}"
        for i, up in enumerate(uploaded_pdfs)
    )

col_left, col_right = st.columns([2, 1])

with col_right:
    st.subheader("Ejemplos")
    if st.button("Cargar ejemplo (hipotecario)"):
        st.session_state["input_text"] = (
            "NOTA SIMPLE INFORMATIVA. Registro de la Propiedad. "
            "Titular: María López Sánchez. NIE: X1234567T. "
            "Entidad financiera: Banco Ejemplo S.A. "
            "Préstamo hipotecario por importe de 245.000,50 EUR. "
            "Fecha de firma: 22/11/2023."
        )
    if st.button("Demo normalización (sin LLM)"):
        schema = load_schema(PROJECT_ROOT / "schemas" / "credito_hipotecario.yaml")
        raw = {
            "base_imponible_general": "30.000,00 EUR",
            "cuota_liquida_estatal": "5.489,11 EUR",
            "nif_emisor": "B12345678",
            "importe_total_iva": "121,00 EUR",
            "deuda_vigente": "9.000,00 EUR",
            "incidencias_activas": "false",
            "total_gastos_mensuales": "1.250,50 EUR",
        }
        st.session_state["demo_normalization"] = normalize_extracted(raw, schema)
    demo_normalization = st.session_state.get("demo_normalization")
    if demo_normalization:
        st.subheader("Demo (resultado)")
        st.write("Cambios aplicados")
        st.dataframe(demo_normalization.get("changes", []), use_container_width=True)
        st.write("Normalizado")
        st.json(demo_normalization.get("normalized", {}))

with col_left:
    text = st.text_area(
        "Texto de entrada",
        key="input_text",
        height=260,
        placeholder="Pega aquí el texto del documento...",
    )

run_clicked = st.button("Ejecutar", type="primary", disabled=not text.strip())

if run_clicked:
    try:
        with st.spinner("Ejecutando extracción con LLM local (Ollama)..."):
            expediente_texts = st.session_state.get("expediente_texts")
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
                )
    except Exception as e:
        st.error("Falló la extracción: el modelo devolvió una respuesta no válida o hubo un problema de conexión.")
        st.exception(e)
        st.stop()

    st.subheader("Resultado")
    tab_labels = ["Extracción", "Normalización", "Validación", "Auditoría"]
    show_gt = False
    if "documents" not in result:
        uploaded_names = st.session_state.get("uploaded_names")
        show_gt = isinstance(uploaded_names, list) and len(uploaded_names) == 1 and isinstance(uploaded_names[0], str)
    if show_gt:
        tab_labels.append("Evaluación")
    if "documents" in result:
        tab_labels.insert(0, "Expediente")
    tabs = st.tabs(tab_labels)
    idx = 0
    if "documents" in result:
        tab_expediente = tabs[idx]
        idx += 1
    tab_extracted, tab_normalization, tab_validation, tab_audit = tabs[idx], tabs[idx + 1], tabs[idx + 2], tabs[idx + 3]
    tab_eval = tabs[idx + 4] if show_gt else None

    if "documents" in result:
        with tab_expediente:
            st.write("Documentos procesados")
            st.json(result.get("documents", []))

    with tab_extracted:
        st.write("Schema")
        st.json(result.get("schema", {}))
        st.write("Extracted (raw)")
        st.json(result.get("extracted_raw", {}))
        st.write("Extracted (normalizado)")
        st.json(result.get("extracted", {}))

    with tab_normalization:
        normalization = result.get("normalization", {})
        changes = normalization.get("changes", [])
        st.write("Cambios aplicados")
        if changes:
            st.dataframe(changes, use_container_width=True)
        else:
            st.write("No se aplicaron cambios porque el LLM ya devolvió formatos consistentes.")

        st.write("Normalizado")
        st.json(normalization.get("normalized", {}))

    with tab_validation:
        validation = result.get("validation", {})
        st.write(f"Válido: {validation.get('valid')}")
        issues = validation.get("issues", [])
        if issues:
            st.dataframe(issues, use_container_width=True)
        else:
            st.write("Sin incidencias.")

    with tab_audit:
        report = result.get("report", {}) or {}
        report_json = report.get("json") or {}
        report_md = report.get("markdown") or ""
        if report_md:
            st.markdown(report_md)
        if report_json:
            st.write("Informe (JSON)")
            st.json(report_json)

    if tab_eval is not None:
        with tab_eval:
            st.write("Comparación contra ground truth del corpus (si existe).")
            uploaded_names = st.session_state.get("uploaded_names") or []
            pdf_filename = uploaded_names[0] if uploaded_names else None
            render_ground_truth_evaluation(result=result, pdf_filename=pdf_filename, project_root=PROJECT_ROOT)

    st.subheader("JSON completo")
    st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")
