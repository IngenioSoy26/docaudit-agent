from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.orchestrator import run_pipeline  # noqa: E402
from core.schema_loader import load_schema  # noqa: E402
from core.normalizer import normalize_extracted  # noqa: E402


st.set_page_config(page_title="DocAudit Agent", layout="wide")
st.title("DocAudit Agent")

st.write("Pega texto (o el contenido extraído de un documento) y ejecuta el pipeline: extraer → normalizar → validar.")

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
        schema = load_schema(PROJECT_ROOT / "schemas" / "hipotecario.yaml")
        raw = {
            "titular_nombre": "  María López Sánchez  ",
            "titular_identificacion": "X1234567T",
            "entidad_financiera": "Banco Ejemplo S.A.",
            "importe_prestamo": "245.000,50 EUR",
            "fecha_firma": "22/11/2023",
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
    with st.spinner("Ejecutando extracción con LLM local (Ollama)..."):
        result: dict[str, Any] = run_pipeline(text)

    st.subheader("Resultado")
    tab_extracted, tab_normalization, tab_validation = st.tabs(
        ["Extracción", "Normalización", "Validación"]
    )

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

    st.subheader("JSON completo")
    st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")
