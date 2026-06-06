from __future__ import annotations

"""
Evaluación cuantitativa reproducible del sistema (corpus + ground truth).

Este script permite medir, por documento y de forma agregada:
- coincidencia exacta por campo (exact match / match_rate),
- precisión, recall y F1 a nivel de campo (micro, sobre presencia de pred/gt),
- latencia por ejecución (segundos),
- pico de memoria RSS del proceso (MB),
- desglose nativo vs degradado (simulación de degradación tipo OCR).

Backends:
- heuristic: baseline determinista (no llama al LLM). Útil para reproducibilidad y regresiones.
- llm: extracción con LLM (Ollama). Puede variar por hardware/modelo/temperatura.
- auto: intenta LLM y hace fallback a heurístico si el LLM no devuelve campos útiles.

Salida:
- CSV con métricas por ejecución.
- JSON resumen con agregados (overall, por caso, nativo vs degradado).
- Un JSON por documento con lista de mismatches (para análisis de errores).
"""

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.auditor import audit_document
from agents.extractor import extract_from_text
from core.document_loader import extract_text_from_pdf_bytes
from core.normalizer import normalize_extracted
from core.privacy import redact_pii
from core.schema_loader import load_schema
from core.settings import settings
from core.validator import validate_extracted


@dataclass(frozen=True)
class DocSample:
    case_id: str
    pdf_path: Path
    ground_truth_path: Path
    schema_name: str


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolve_under_project(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (_PROJECT_ROOT / p).resolve()


def _iter_samples(dataset_root: Path) -> list[DocSample]:
    case_map = {
        "caso_uso_1_auditoria_hipotecaria": "credito_hipotecario",
        "caso_uso_2_auditoria_fiscal": "auditoria_fiscal",
        "caso_uso_3_kyc_onboarding": "kyc_onboarding",
    }
    samples: list[DocSample] = []
    for case_dir_name, schema_name in case_map.items():
        case_dir = dataset_root / case_dir_name
        if not case_dir.exists():
            continue
        for pdf_path in sorted(case_dir.glob("*.pdf")):
            gt_path = _find_ground_truth_for_pdf(pdf_path.name, case_dir)
            if gt_path is None:
                continue
            samples.append(
                DocSample(
                    case_id=case_dir_name,
                    pdf_path=pdf_path,
                    ground_truth_path=gt_path,
                    schema_name=schema_name,
                )
            )
    return samples


def _find_ground_truth_for_pdf(pdf_filename: str, case_dir: Path) -> Path | None:
    name = pdf_filename.lower()
    if name.startswith("contrato_hipoteca_esp_") and name.endswith(".pdf"):
        suffix = name.removesuffix(".pdf").split("_")[-1]
        gt = case_dir / f"ground_truth_esp_{suffix}.json"
        return gt if gt.exists() else None
    if name.startswith("factura_fiscal_") and name.endswith(".pdf"):
        stem = Path(pdf_filename).stem
        gt = case_dir / f"{stem}.json"
        return gt if gt.exists() else None
    if name.startswith("expediente_kyc_") and name.endswith(".pdf"):
        stem = Path(pdf_filename).stem
        gt = case_dir / f"{stem}.json"
        return gt if gt.exists() else None
    return None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_for_compare(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value.strip()).casefold()
    return value


def _values_equal(a: Any, b: Any, float_tol: float) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return abs(float(a) - float(b)) <= float_tol
    if isinstance(a, bool) and isinstance(b, bool):
        return a is b
    return _normalize_for_compare(a) == _normalize_for_compare(b)


def _compute_field_metrics(
    extracted: dict[str, Any],
    ground_truth: dict[str, Any],
    float_tol: float,
) -> dict[str, Any]:
    keys = sorted(ground_truth.keys())
    total_fields = len(keys)
    gt_present = 0
    pred_present = 0
    exact_matches = 0
    mismatches: list[dict[str, Any]] = []

    for k in keys:
        gt_v = ground_truth.get(k)
        pred_v = extracted.get(k)

        gt_has = gt_v is not None and gt_v != ""
        pred_has = pred_v is not None and pred_v != ""
        if gt_has:
            gt_present += 1
        if pred_has:
            pred_present += 1

        ok = _values_equal(pred_v, gt_v, float_tol=float_tol)
        if ok:
            exact_matches += 1
        else:
            if gt_has and not pred_has:
                mismatch_type = "miss"
            elif (not gt_has) and pred_has:
                mismatch_type = "spurious"
            elif gt_has and pred_has:
                mismatch_type = "value"
            else:
                mismatch_type = "other"
            mismatches.append(
                {
                    "field": k,
                    "pred": pred_v,
                    "gt": gt_v,
                    "type": mismatch_type,
                    "gt_has": gt_has,
                    "pred_has": pred_has,
                }
            )

    precision = (exact_matches / pred_present) if pred_present else 0.0
    recall = (exact_matches / gt_present) if gt_present else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    match_rate = (exact_matches / total_fields) if total_fields else 0.0

    return {
        "fields_total": total_fields,
        "fields_gt_present": gt_present,
        "fields_pred_present": pred_present,
        "exact_matches": exact_matches,
        "match_rate": match_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mismatches": mismatches,
    }


def _apply_degradation(text: str, seed: int) -> str:
    rng = random.Random(seed)

    def maybe(p: float) -> bool:
        return rng.random() < p

    out = text

    if maybe(0.6):
        out = out.replace("€", "EUR")
    if maybe(0.6):
        out = re.sub(r"(\d)\.(\d{3})", r"\1\2", out)
    if maybe(0.6):
        out = re.sub(r"(\d),(\d{2})", r"\1.\2", out)
    if maybe(0.4):
        out = out.replace("O", "0").replace("I", "1")
    if maybe(0.5):
        out = re.sub(r"[–—−]", "-", out)
    if maybe(0.7):
        out = re.sub(r"\s+", " ", out)

    chars = list(out)
    for i in range(0, len(chars), 250):
        if i + 1 < len(chars) and maybe(0.15):
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        if maybe(0.10):
            chars[i] = ""
    out = "".join(chars)

    return out


def _parse_es_number(raw: str) -> float | None:
    s = (raw or "").strip()
    s = s.replace("€", "").replace("EUR", "").replace("Euros", "").replace("euros", "")
    s = re.sub(r"\s+", "", s)
    if not s:
        return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "")
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s and "." not in s:
        s = s.replace(",", ".")
    s = re.sub(r"[^0-9.\-+]", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def _heuristic_extract_fields(schema_name: str, text: str, pdf_name: str) -> dict[str, Any]:
    t = text or ""

    if schema_name == "credito_hipotecario":
        m = re.search(r"contrato_hipoteca_esp_(\d{4})", pdf_name.lower())
        doc_suffix = m.group(1) if m else None
        id_documento = f"HIP-{doc_suffix}" if doc_suffix else None

        fecha_emision = None
        m = re.search(r"\ba\s+(\d{4}-\d{2}-\d{2})\b", t)
        if m:
            fecha_emision = m.group(1)

        m = re.search(r"Comparecen\s+D\./Dña\.\s+(.+?),\s+con\s+DNI\s+([0-9]{8}[A-Z])", t)
        nombre_cliente = m.group(1).strip() if m else None
        dni_cliente = m.group(2).strip() if m else None

        monto_prestamo_eur = None
        m = re.search(r"capital\s+de\s+([\d\.,]+)\s+Euros", t, flags=re.IGNORECASE)
        if m:
            monto_prestamo_eur = _parse_es_number(m.group(1))

        tasa_interes = None
        m = re.search(r"\bal\s+([\d\.,]+)\s*%\s+anual\b", t, flags=re.IGNORECASE)
        if m:
            tasa_interes = _parse_es_number(m.group(1))

        return {
            "id_documento": id_documento,
            "nombre_cliente": nombre_cliente,
            "dni_cliente": dni_cliente,
            "monto_prestamo_eur": monto_prestamo_eur,
            "tasa_interes": tasa_interes,
            "fecha_emision": fecha_emision,
        }

    if schema_name == "auditoria_fiscal":
        razon_social_emisor = None
        m = re.search(r"(?:Emisor|Proveedor|Raz[oó]n social)\s*:\s*(.+)", t, flags=re.IGNORECASE)
        if m:
            razon_social_emisor = m.group(1).strip()

        nif_emisor = None
        m = re.search(r"\b([A-HJNP-SUW][0-9]{7}[0-9A-J])\b", t)
        if m:
            nif_emisor = m.group(1).strip()

        num_factura = None
        m = re.search(r"\b(FAC-\d{4}-\d{4})\b", t, flags=re.IGNORECASE)
        if m:
            num_factura = m.group(1).upper()

        fecha_expedicion = None
        m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", t)
        if m:
            fecha_expedicion = m.group(1)

        base_imponible = None
        m = re.search(r"(?:Base imponible|Base)\s*:\s*([\d\.,]+)", t, flags=re.IGNORECASE)
        if m:
            base_imponible = _parse_es_number(m.group(1))

        tipo_iva = None
        m = re.search(r"(?:Tipo IVA|IVA)\s*:\s*([0-9]{1,2})\s*%", t, flags=re.IGNORECASE)
        if m:
            tipo_iva = int(m.group(1))

        cuota_iva = None
        m = re.search(r"(?:Cuota IVA|IVA)\s*(?:repercutido)?\s*:\s*([\d\.,]+)", t, flags=re.IGNORECASE)
        if m:
            cuota_iva = _parse_es_number(m.group(1))

        importe_total = None
        m = re.search(r"(?:Total|Importe total)\s*:\s*([\d\.,]+)", t, flags=re.IGNORECASE)
        if m:
            importe_total = _parse_es_number(m.group(1))

        return {
            "razon_social_emisor": razon_social_emisor,
            "nif_emisor": nif_emisor,
            "num_factura": num_factura,
            "fecha_expedicion": fecha_expedicion,
            "base_imponible": base_imponible,
            "tipo_iva": tipo_iva,
            "cuota_iva": cuota_iva,
            "retencion_irpf": None,
            "importe_total": importe_total,
        }

    if schema_name == "kyc_onboarding":
        nombre_titular = None
        primer_apellido = None
        segundo_apellido = None
        m = re.search(r"\bNombre\s*:\s*([A-Za-zÁÉÍÓÚÑÜáéíóúñü]+)\b", t)
        if m:
            nombre_titular = m.group(1).strip()
        m = re.search(r"\bPrimer apellido\s*:\s*([A-Za-zÁÉÍÓÚÑÜáéíóúñü]+)\b", t, flags=re.IGNORECASE)
        if m:
            primer_apellido = m.group(1).strip()
        m = re.search(r"\bSegundo apellido\s*:\s*([A-Za-zÁÉÍÓÚÑÜáéíóúñü]+)\b", t, flags=re.IGNORECASE)
        if m:
            segundo_apellido = m.group(1).strip()

        num_documento = None
        m = re.search(r"\b([XYZ0-9][0-9]{7}[A-Z])\b", t)
        if m:
            num_documento = m.group(1).strip()

        fechas = re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", t)
        fecha_nacimiento = fechas[0] if len(fechas) >= 1 else None
        fecha_caducidad = fechas[1] if len(fechas) >= 2 else None

        domicilio_comprobante = None
        m = re.search(r"\bDomicilio\s*:\s*(.+)", t, flags=re.IGNORECASE)
        if m:
            domicilio_comprobante = m.group(1).strip()

        return {
            "nombre_titular": nombre_titular,
            "primer_apellido": primer_apellido,
            "segundo_apellido": segundo_apellido,
            "num_documento": num_documento,
            "fecha_nacimiento": fecha_nacimiento,
            "fecha_caducidad": fecha_caducidad,
            "domicilio_comprobante": domicilio_comprobante,
        }

    return {}


def _measure_peak_rss_mb(fn) -> tuple[Any, float]:
    process = psutil.Process()
    stop = threading.Event()
    peak = {"rss": process.memory_info().rss}

    def sampler() -> None:
        while not stop.is_set():
            try:
                rss = process.memory_info().rss
                if rss > peak["rss"]:
                    peak["rss"] = rss
            except Exception:
                pass
            time.sleep(0.05)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    try:
        result = fn()
    finally:
        stop.set()
        t.join(timeout=1.0)
    return result, peak["rss"] / (1024 * 1024)


def _run_single(
    pdf_bytes: bytes,
    schema_name: str,
    degraded: bool,
    float_tol: float,
    pdf_name: str,
    backend: str,
) -> dict[str, Any]:
    doc_id = _sha256_hex(pdf_bytes)
    extracted_doc = extract_text_from_pdf_bytes(pdf_bytes)
    text = extracted_doc.get("text") or ""
    pages = extracted_doc.get("page_texts") or None

    if settings.enable_pii_redaction:
        text, _ = redact_pii(text)
        if isinstance(pages, list):
            pages = [redact_pii(p)[0] for p in pages]

    if degraded:
        text = _apply_degradation(text, seed=int(doc_id[:8], 16))
        pages = None

    schema = load_schema(_PROJECT_ROOT / "schemas" / f"{schema_name}.yaml")

    extractor_backend = "llm" if backend in {"auto", "llm"} else "heuristic"
    extracted_fields: dict[str, Any] = {}
    field_details: dict[str, Any] = {}
    if backend in {"auto", "llm"}:
        try:
            extracted_payload = extract_from_text(text, schema, pages=pages, doc_id=doc_id)
            extracted_candidate = (
                extracted_payload.get("fields") if isinstance(extracted_payload, dict) else None
            )
            details_candidate = (
                extracted_payload.get("details") if isinstance(extracted_payload, dict) else None
            )
            if isinstance(extracted_candidate, dict):
                extracted_fields = extracted_candidate
            elif isinstance(extracted_payload, dict):
                extracted_fields = extracted_payload
            if isinstance(details_candidate, dict):
                field_details = details_candidate
        except Exception:
            extracted_fields = {}
            field_details = {}

    def _is_empty_pred(v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, str) and v.strip() == "":
            return True
        return False

    should_fallback = (
        backend == "heuristic"
        or not extracted_fields
        or all(_is_empty_pred(v) for v in extracted_fields.values())
    )
    if should_fallback:
        extractor_backend = "heuristic"
        extracted_fields = _heuristic_extract_fields(schema_name, text, pdf_name)
        field_details = {}

    normalization = normalize_extracted(extracted_fields, schema)
    normalized = normalization["normalized"]
    validation = validate_extracted(normalized, schema)
    report = audit_document(schema, normalized, validation, field_details=field_details or {})
    return {
        "doc_id": doc_id,
        "schema_name": schema_name,
        "text_method": extracted_doc.get("method"),
        "pages": extracted_doc.get("pages"),
        "degraded_text": degraded,
        "extractor_backend": extractor_backend,
        "extracted": normalized,
        "normalization": normalization,
        "validation": validation,
        "report": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluación cuantitativa sobre el corpus sample_docs.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(Path("data") / "sample_docs"),
        help="Directorio con el corpus (por defecto: data/sample_docs).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("reports") / "evaluation_results.csv"),
        help="CSV de salida (por defecto: reports/evaluation_results.csv).",
    )
    parser.add_argument(
        "--float-tol",
        type=float,
        default=0.01,
        help="Tolerancia absoluta para comparar números (por defecto: 0.01).",
    )
    parser.add_argument(
        "--include-degraded",
        action="store_true",
        help="Evalúa también una versión degradada del texto (simula OCR).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "llm", "heuristic"],
        help="Backend de extracción (auto intenta LLM y cae a heurístico).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limita el número de documentos evaluados (0 = sin límite).",
    )
    args = parser.parse_args()

    dataset_root = _resolve_under_project(args.dataset)
    samples = _iter_samples(dataset_root)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    out_path = _resolve_under_project(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    fieldnames = [
        "case_id",
        "pdf",
        "schema",
        "degraded_text",
        "text_method",
        "extractor_backend",
        "fields_total",
        "fields_gt_present",
        "fields_pred_present",
        "exact_matches",
        "match_rate",
        "precision",
        "recall",
        "f1",
        "latency_s",
        "ram_peak_mb",
        "mismatch_count",
    ]

    mismatches_out = out_path.with_suffix("")
    mismatches_dir = mismatches_out.parent / (mismatches_out.name + "_mismatches")
    mismatches_dir.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        f.flush()

        for sample in samples:
            pdf_bytes = sample.pdf_path.read_bytes()
            gt = _load_json(sample.ground_truth_path)

            for degraded in ([False, True] if args.include_degraded else [False]):
                started = time.perf_counter()

                def _run():
                    return _run_single(
                        pdf_bytes=pdf_bytes,
                        schema_name=sample.schema_name,
                        degraded=degraded,
                        float_tol=args.float_tol,
                        pdf_name=sample.pdf_path.name,
                        backend=args.backend,
                    )

                result, peak_rss_mb = _measure_peak_rss_mb(_run)
                elapsed_s = time.perf_counter() - started

                metrics = _compute_field_metrics(result["extracted"], gt, float_tol=args.float_tol)

                row = {
                    "case_id": sample.case_id,
                    "pdf": sample.pdf_path.name,
                    "schema": sample.schema_name,
                    "degraded_text": degraded,
                    "text_method": result.get("text_method"),
                    "extractor_backend": result.get("extractor_backend"),
                    "fields_total": metrics["fields_total"],
                    "fields_gt_present": metrics["fields_gt_present"],
                    "fields_pred_present": metrics["fields_pred_present"],
                    "exact_matches": metrics["exact_matches"],
                    "match_rate": metrics["match_rate"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "latency_s": elapsed_s,
                    "ram_peak_mb": peak_rss_mb,
                    "mismatch_count": len(metrics["mismatches"]),
                }
                rows.append(row)
                w.writerow(row)
                f.flush()

                mm_path = mismatches_dir / f"{sample.pdf_path.stem}{'_degraded' if degraded else ''}.json"
                mm_path.write_text(json.dumps(metrics["mismatches"], ensure_ascii=False, indent=2), encoding="utf-8")

    if not rows:
        print(f"ERROR: no se encontraron muestras en: {dataset_root}")
        print("Sugerencia: ejecuta desde el root del repositorio o pasa --dataset con ruta absoluta.")
        return 2

    summary = _summarize(rows)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"OK: {len(rows)} ejecuciones")
    print(f"CSV: {out_path}")
    print(f"Resumen: {summary_path}")
    return 0


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _avg(xs: list[float]) -> float:
        return float(statistics.mean(xs)) if xs else 0.0

    def _p95(xs: list[float]) -> float:
        return float(statistics.quantiles(xs, n=20)[-1]) if len(xs) >= 2 else float(xs[0]) if xs else 0.0

    def _micro(sel: list[dict[str, Any]]) -> dict[str, Any]:
        total_fields = int(sum(r["fields_total"] for r in sel))
        gt_present = int(sum(r["fields_gt_present"] for r in sel))
        pred_present = int(sum(r["fields_pred_present"] for r in sel))
        exact = int(sum(r["exact_matches"] for r in sel))
        precision = (exact / pred_present) if pred_present else 0.0
        recall = (exact / gt_present) if gt_present else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        match_rate = (exact / total_fields) if total_fields else 0.0
        return {
            "fields_total": total_fields,
            "fields_gt_present": gt_present,
            "fields_pred_present": pred_present,
            "exact_matches": exact,
            "micro_match_rate": match_rate,
            "micro_precision": precision,
            "micro_recall": recall,
            "micro_f1": f1,
        }

    def _aggregate(sel: list[dict[str, Any]]) -> dict[str, Any]:
        out_block = {
            "runs": len(sel),
            "avg_match_rate": _avg([r["match_rate"] for r in sel]),
            "avg_f1": _avg([r["f1"] for r in sel]),
            "avg_latency_s": _avg([r["latency_s"] for r in sel]),
            "p95_latency_s": _p95([r["latency_s"] for r in sel]),
            "avg_ram_peak_mb": _avg([r["ram_peak_mb"] for r in sel]),
        }
        out_block.update(_micro(sel))
        return out_block

    out: dict[str, Any] = {"overall": {}, "by_case": {}, "by_degraded": {}}
    out["overall"] = _aggregate(rows)

    by_case: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_case.setdefault(r["case_id"], []).append(r)

    for case_id, sel in by_case.items():
        out["by_case"][case_id] = _aggregate(sel)

    by_degraded: dict[str, list[dict[str, Any]]] = {"native": [], "degraded": []}
    for r in rows:
        by_degraded["degraded" if r.get("degraded_text") else "native"].append(r)
    out["by_degraded"]["native"] = _aggregate(by_degraded["native"])
    out["by_degraded"]["degraded"] = _aggregate(by_degraded["degraded"])

    return out


if __name__ == "__main__":
    raise SystemExit(main())
