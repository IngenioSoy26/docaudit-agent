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
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Streamlit ejecuta el script como módulo; esto asegura imports relativos al proyecto.
    sys.path.insert(0, str(PROJECT_ROOT))

from core.schema_models import DocSchema  # noqa: E402
import core.schema_loader as schema_loader  # noqa: E402
from ui.components import render_confidence_gauge, render_ground_truth_evaluation  # noqa: E402

try:  # noqa: E402
    from core.document_loader import extract_text_from_pdf_bytes, extract_text_from_scanned_pdf_bytes
except ImportError:  # noqa: E402
    from core.document_loader import extract_text_from_pdf_bytes

    extract_text_from_scanned_pdf_bytes = None
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
</style>
""",
    unsafe_allow_html=True,
)

st.title("DocAudit Agent")
st.caption(
    "Plataforma multi-agente para extracción, normalización, validación y auditoría documental. "
    "Diseñada para generar datos estructurados y trazables que apoyen el análisis y la toma de decisiones."
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


def _load_pdfs_from_folder(folder: str) -> list[dict[str, Any]]:
    p = Path(folder or "").expanduser()
    if not p.exists() or not p.is_dir():
        return []
    files = sorted([x for x in p.glob("*.pdf") if x.is_file()])
    out: list[dict[str, Any]] = []
    for f in files:
        out.append({"name": f.name, "bytes": f.read_bytes(), "source": "carpeta", "path": str(f)})
    return out


def _load_pdfs_from_uploads(uploaded: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for up in uploaded:
        out.append({"name": up.name, "bytes": up.read(), "source": "subida", "path": None})
    return out

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
    folder_path = st.text_input("Ruta de carpeta (PDFs)", value="")
    load_folder_clicked = st.button("Cargar PDFs desde carpeta")
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

uploaded_pdfs = st.file_uploader("Subir PDF(s)", type=["pdf"], accept_multiple_files=True)

pdf_items: list[dict[str, Any]] = []
if load_folder_clicked and folder_path.strip():
    pdf_items = _load_pdfs_from_folder(folder_path.strip())
elif uploaded_pdfs:
    pdf_items = _load_pdfs_from_uploads(list(uploaded_pdfs))

if pdf_items:
    expediente_texts: list[str] = []
    expediente_pages: list[list[str] | None] = []
    expediente_doc_ids: list[str] = []
    expediente_meta: list[dict[str, Any]] = []

    for i, item in enumerate(pdf_items):
        name = str(item.get("name") or f"documento_{i+1}.pdf")
        pdf_bytes = item.get("bytes") or b""
        source = str(item.get("source") or "-")
        path = item.get("path")

        doc_id = hashlib.sha256(pdf_bytes).hexdigest() if isinstance(pdf_bytes, (bytes, bytearray)) else ""
        extracted = extract_text_from_pdf_bytes(pdf_bytes)
        text = extracted.get("text") or ""
        pages = extracted.get("page_texts") if isinstance(extracted.get("page_texts"), list) else None
        pages_n = extracted.get("pages")
        method = extracted.get("method")

        if not text and use_vision:
            if extract_text_from_scanned_pdf_bytes is None:
                st.error(
                    "No está disponible la función de OCR. Detén y vuelve a iniciar Streamlit "
                    "para recargar los módulos, y verifica que tu entorno tenga el código actualizado."
                )
            else:
                with st.spinner(f"Extrayendo texto con OCR — {name}..."):
                    extracted_v = extract_text_from_scanned_pdf_bytes(pdf_bytes)
                text = extracted_v.get("text") or ""
                pages = extracted_v.get("page_texts") if isinstance(extracted_v.get("page_texts"), list) else pages
                pages_n = extracted_v.get("pages") or pages_n
                method = extracted_v.get("method") or "vision"

        expediente_texts.append(text)
        expediente_pages.append(pages)
        expediente_doc_ids.append(doc_id)
        expediente_meta.append(
            {
                "doc_index": i + 1,
                "documento": name,
                "origen": source,
                "folios": pages_n,
                "metodo": method,
                "caracteres": len(text or ""),
                "ruta": path,
            }
        )

    st.session_state["expediente_texts"] = expediente_texts
    st.session_state["expediente_pages"] = expediente_pages
    st.session_state["expediente_doc_ids"] = expediente_doc_ids
    st.session_state["expediente_meta"] = expediente_meta
    st.session_state["uploaded_names"] = [m["documento"] for m in expediente_meta]
    if len(expediente_texts) == 1:
        st.session_state["input_pages"] = expediente_pages[0]
        st.session_state["doc_id"] = expediente_doc_ids[0]

    st.subheader("Documentos cargados")
    st.dataframe(expediente_meta, use_container_width=True, hide_index=True)
    st.session_state["input_text"] = "\n\n".join(
        f"=== Documento {i+1}: {expediente_meta[i]['documento']} ===\n{expediente_texts[i]}"
        for i in range(len(expediente_texts))
    )

top_left, top_right = st.columns([2, 1])

with top_right:
    st.subheader("Ejemplos rápidos")
    if st.button("Ejemplo (hipotecario)"):
        st.session_state["input_text"] = (
            "Escritura de préstamo hipotecario. Comparecen D./Dña. Armida Falcón Trillo, con DNI 86473212N. "
            "Capital de 157.520,60 Euros al 3,6 % anual. a 2025-03-08."
        )
    if st.button("Ejemplo (factura)"):
        st.session_state["input_text"] = (
            "FACTURA FAC-2026-0001. Emisor: Proveedor Demo S.L. NIF: A1234567B. "
            "Fecha: 2026-01-15. Base imponible: 100,00 EUR. Tipo IVA: 21%. Cuota IVA: 21,00. Total: 121,00."
        )
    if st.button("Ejemplo (KYC)"):
        st.session_state["input_text"] = (
            "Expediente KYC. Nombre: Juan. Primer apellido: Pérez. Segundo apellido: García. "
            "Documento: 12345678Z. Fecha nacimiento: 1990-01-01. Fecha caducidad: 2099-01-01. "
            "Domicilio: Calle Ejemplo 1, Madrid, 28001."
        )
    if st.button("Demo normalización (sin LLM)"):
        schema = load_schema(PROJECT_ROOT / "schemas" / "credito_hipotecario.yaml")
        raw = {
            "id_documento": "HIP-0001",
            "nombre_cliente": "Armida Falcón Trillo",
            "dni_cliente": "86473212N",
            "monto_prestamo_eur": "157.520,60 EUR",
            "tasa_interes": "3,6",
            "fecha_emision": "2025-03-08",
        }
        st.session_state["demo_normalization"] = normalize_extracted(raw, schema)
    demo_normalization = st.session_state.get("demo_normalization")
    if demo_normalization:
        with st.expander("Ver demo de normalización", expanded=False):
            st.write("Cambios aplicados")
            st.dataframe(demo_normalization.get("changes", []), use_container_width=True, hide_index=True)
            st.write("Normalizado")
            st.json(demo_normalization.get("normalized", {}))

with top_left:
    st.subheader("Entrada")
    text = st.text_area(
        "Texto del documento",
        key="input_text",
        height=240,
        placeholder="Pega aquí el texto del documento o sube un PDF(s) arriba.",
        label_visibility="visible",
    )
    run_clicked = st.button("Ejecutar análisis", type="primary", disabled=not text.strip())

if run_clicked:
    try:
        with st.spinner("Ejecutando pipeline (extracción → normalización → validación → auditoría)..."):
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
                    schema_name=None if schema_choice == "Auto" else schema_choice,
                )
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
                    rows.append(
                        {
                            "doc_index": (idx_doc + 1) if isinstance(idx_doc, int) else idx_doc,
                            "document_type": d.get("document_type"),
                            "valido": validation.get("valid") if isinstance(validation, dict) else None,
                            "incidencias": len(validation.get("issues", []) or []) if isinstance(validation, dict) else None,
                            "score_confianza": report_json.get("score_confianza") if isinstance(report_json, dict) else None,
                            "reglas_ok": reglas_ok,
                            "reglas_total": reglas_total,
                        }
                    )
            st.dataframe(rows, use_container_width=True, hide_index=True)
            if isinstance(meta, list) and meta:
                st.write("Archivos y folios")
                st.dataframe(meta, use_container_width=True, hide_index=True)

            if isinstance(docs, list) and docs:
                opciones = [f"Documento {int(d.get('doc_index'))+1 if isinstance(d.get('doc_index'), int) else d.get('doc_index')}" for d in docs if isinstance(d, dict)]
                sel = st.selectbox("Ver detalle por documento", options=opciones)
                sel_idx = opciones.index(sel) if sel in opciones else 0
                chosen = docs[sel_idx] if sel_idx < len(docs) else None
                if isinstance(chosen, dict):
                    sub_tabs = st.tabs(["Extracción", "Normalización", "Validación", "Auditoría", "JSON"])
                    with sub_tabs[0]:
                        st.dataframe(
                            _to_table_rows(extracted=chosen.get("extracted") or {}, schema=None),
                            use_container_width=True,
                            hide_index=True,
                        )
                        _render_json(label="Ver extracted_raw", payload=chosen.get("extracted_raw"))
                    with sub_tabs[1]:
                        norm = chosen.get("normalization") if isinstance(chosen.get("normalization"), dict) else {}
                        changes = norm.get("changes") if isinstance(norm.get("changes"), list) else []
                        if changes:
                            st.dataframe(changes, use_container_width=True, hide_index=True)
                        _render_json(label="Ver normalizado (JSON)", payload=norm.get("normalized"))
                    with sub_tabs[2]:
                        val = chosen.get("validation") if isinstance(chosen.get("validation"), dict) else {}
                        issues = val.get("issues") if isinstance(val.get("issues"), list) else []
                        _render_issues_table(issues=issues)
                        _render_json(label="Ver validación (JSON)", payload=val)
                    with sub_tabs[3]:
                        rep = chosen.get("report") if isinstance(chosen.get("report"), dict) else {}
                        rep_json = rep.get("json") if isinstance(rep.get("json"), dict) else {}
                        rep_md = rep.get("markdown") if isinstance(rep.get("markdown"), str) else ""
                        score = rep_json.get("score_confianza")
                        if isinstance(score, (int, float)):
                            _render_score_gauge(score=float(score), key=f"score_confianza_doc_{sel_idx}")
                        if rep_md:
                            st.markdown(rep_md)
                        _render_json(label="Ver informe (JSON)", payload=rep_json)
                    with sub_tabs[4]:
                        st.json(chosen)

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
