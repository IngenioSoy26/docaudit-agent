from __future__ import annotations

"""Componentes auxiliares de UI (Streamlit).

Este módulo encapsula lógica reutilizable para:
- localizar archivos de ground truth del corpus,
- comparar extracción vs etiqueta,
- renderizar resultados en la interfaz.
"""

import json
import re
from pathlib import Path
from typing import Any

import streamlit as st


def _norm_text(v: Any) -> str:
    """Normaliza valores a texto comparable (lower + colapsa espacios)."""
    if v is None:
        return ""
    s = str(v).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _as_float(v: Any) -> float | None:
    """Convierte valores a float cuando sea posible (evita bool)."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    try:
        return float(str(v))
    except Exception:
        return None


def find_ground_truth_for_pdf(*, pdf_filename: str, project_root: Path) -> Path | None:
    """Busca el JSON de ground truth asociado a un PDF del corpus.

    La búsqueda se hace por convenciones de nombre dentro de `data/sample_docs/`.
    """
    name = Path(pdf_filename).name
    stem = Path(name).stem
    data_root = project_root / "data" / "sample_docs"
    if not data_root.exists():
        return None

    if stem.startswith("contrato_hipoteca_esp_"):
        m = re.search(r"(\d{4})$", stem)
        if m:
            gt_name = f"ground_truth_esp_{m.group(1)}.json"
            hits = list(data_root.rglob(gt_name))
            return hits[0] if hits else None

    for candidate in (f"{stem}.json",):
        hits = list(data_root.rglob(candidate))
        if hits:
            return hits[0]

    return None


def load_ground_truth(path: Path) -> dict[str, Any]:
    """Carga un ground truth en formato JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def compare_extracted_to_ground_truth(
    *,
    extracted: dict[str, Any],
    ground_truth: dict[str, Any],
    float_tol: float = 0.01,
) -> dict[str, Any]:
    """Compara extracción vs ground truth y devuelve filas + resumen.

    - Si ambos valores son numéricos, compara con tolerancia.
    - Si no, compara texto normalizado.
    """
    rows: list[dict[str, Any]] = []
    matched = 0
    total = 0
    tp = 0
    predicted_present = 0
    expected_present = 0

    for key, expected in ground_truth.items():
        if key in {"id_documento"}:
            continue
        total += 1
        got = extracted.get(key)
        ok = False
        exp_f = _as_float(expected)
        got_f = _as_float(got)
        if exp_f is not None and got_f is not None:
            ok = abs(exp_f - got_f) <= float_tol
        else:
            ok = _norm_text(expected) == _norm_text(got)
        if ok:
            matched += 1
        got_present = _norm_text(got) != ""
        exp_present = _norm_text(expected) != ""
        if got_present:
            predicted_present += 1
        if exp_present:
            expected_present += 1
        if ok and got_present and exp_present:
            tp += 1
        rows.append({"campo": key, "esperado": expected, "extraido": got, "match": ok})

    rate = (matched / total) if total else 0.0
    precision = (tp / predicted_present) if predicted_present else 0.0
    recall = (tp / expected_present) if expected_present else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "rows": rows,
        "summary": {
            "total": total,
            "matched": matched,
            "match_rate": rate,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
    }


def render_confidence_gauge(*, score: float | None) -> None:
    if score is None:
        return
    try:
        import plotly.graph_objects as go
    except Exception:
        st.progress(int(max(0.0, min(1.0, float(score))) * 100))
        return
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(score) * 100.0,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 50], "color": "#f2f2f2"},
                    {"range": [50, 80], "color": "#e6f2ff"},
                    {"range": [80, 100], "color": "#d9f2d9"},
                ],
            },
        )
    )
    fig.update_layout(height=220, margin={"l": 20, "r": 20, "t": 30, "b": 10})
    st.plotly_chart(fig, use_container_width=True)


def render_ground_truth_evaluation(
    *,
    result: dict[str, Any],
    pdf_filename: str | None,
    project_root: Path,
) -> None:
    """Renderiza en Streamlit la comparación contra ground truth (si existe)."""
    if not pdf_filename:
        st.write("No hay nombre de archivo para buscar ground truth.")
        return

    gt_path = find_ground_truth_for_pdf(pdf_filename=pdf_filename, project_root=project_root)
    if gt_path is None:
        st.write("No se encontró ground truth para este PDF dentro de data/sample_docs/.")
        return

    gt = load_ground_truth(gt_path)
    extracted = result.get("extracted") or {}
    if not isinstance(extracted, dict):
        st.write("El resultado extraído no es un dict, no se puede comparar.")
        return

    cmp = compare_extracted_to_ground_truth(extracted=extracted, ground_truth=gt)
    summary = cmp["summary"]
    st.write(f"Ground truth: {gt_path.as_posix()}")
    st.write(f"Match rate: {summary['match_rate']:.2%} ({summary['matched']}/{summary['total']})")
    st.write(
        "Precision/Recall/F1 (por presencia de campo): "
        f"{summary['precision']:.2%} / {summary['recall']:.2%} / {summary['f1']:.2%}"
    )
    st.dataframe(cmp["rows"], use_container_width=True)
