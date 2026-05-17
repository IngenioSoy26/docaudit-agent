from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import streamlit as st


def _norm_text(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _as_float(v: Any) -> float | None:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    try:
        return float(str(v))
    except Exception:
        return None


def find_ground_truth_for_pdf(*, pdf_filename: str, project_root: Path) -> Path | None:
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
    return json.loads(path.read_text(encoding="utf-8"))


def compare_extracted_to_ground_truth(
    *,
    extracted: dict[str, Any],
    ground_truth: dict[str, Any],
    float_tol: float = 0.01,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    matched = 0
    total = 0

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
        rows.append({"campo": key, "esperado": expected, "extraido": got, "match": ok})

    rate = (matched / total) if total else 0.0
    return {"rows": rows, "summary": {"total": total, "matched": matched, "match_rate": rate}}


def render_ground_truth_evaluation(
    *,
    result: dict[str, Any],
    pdf_filename: str | None,
    project_root: Path,
) -> None:
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
    st.dataframe(cmp["rows"], use_container_width=True)
