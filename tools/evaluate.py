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
import os
import random
import re
import subprocess
import sys
import statistics
import threading
import tempfile
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
from core.document_loader import extract_text_from_pdf_bytes, extract_text_from_image_bytes, extraction_policy_for_doc
from core.normalizer import normalize_extracted
from core.privacy import redact_pii
from core.schema_loader import load_schema
from core.settings import settings
from core.validator import validate_extracted


def _trace_eval(message: str) -> None:
    trace_path = os.environ.get("DOCAUDIT_EVAL_TRACE_PATH", "").strip()
    if not trace_path:
        return
    try:
        with Path(trace_path).open("a", encoding="utf-8") as fh:
            fh.write(message + "\n")
    except Exception:
        pass


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


def _empty_extraction_result(schema: Any) -> dict[str, Any]:
    fields = {f.name: None for f in schema.fields}
    details = {
        name: {"nombre": name, "valor": None, "confianza": None, "evidencia_textual": "", "pagina": 1}
        for name in fields
    }
    return {"fields": fields, "details": details}


def _rule_key(rule: dict[str, Any]) -> str:
    if rule.get("id") is not None:
        return str(rule["id"])
    expr = rule.get("expresion")
    if isinstance(expr, str) and expr.strip():
        return expr.strip()
    desc = rule.get("descripcion")
    if isinstance(desc, str) and desc.strip():
        return desc.strip()
    return json.dumps(rule, ensure_ascii=False, sort_keys=True)


def _extract_rule_outcomes(report: dict[str, Any]) -> dict[str, bool | None]:
    rules = (
        report.get("json", {}).get("decision_rules")
        if isinstance(report.get("json"), dict)
        else report.get("decision_rules")
    )
    if not isinstance(rules, list):
        return {}
    out: dict[str, bool | None] = {}
    for r in rules:
        if not isinstance(r, dict):
            continue
        out[_rule_key(r)] = r.get("cumple") if isinstance(r.get("cumple"), (bool, type(None))) else None
    return out


def _compute_rule_metrics(pred_report: dict[str, Any], gt_report: dict[str, Any]) -> dict[str, Any]:
    pred = _extract_rule_outcomes(pred_report)
    gt = _extract_rule_outcomes(gt_report)
    keys = sorted(set(pred.keys()) | set(gt.keys()))
    total = len(keys)
    pred_evaluable = sum(1 for k in keys if pred.get(k) is not None)
    gt_evaluable = sum(1 for k in keys if gt.get(k) is not None)
    matches = sum(
        1
        for k in keys
        if pred.get(k) is not None and gt.get(k) is not None and pred.get(k) == gt.get(k)
    )
    accuracy = (matches / pred_evaluable) if pred_evaluable else 0.0
    coverage = (pred_evaluable / total) if total else 0.0
    return {
        "rules_total": total,
        "rules_gt_evaluable": gt_evaluable,
        "rules_pred_evaluable": pred_evaluable,
        "rules_matches": matches,
        "rules_accuracy": accuracy,
        "rules_coverage": coverage,
    }


def _extract_from_text_guarded(
    *,
    text: str,
    schema: Any,
    pages: list[str] | None,
    doc_id: str,
) -> dict[str, Any]:
    _trace_eval(f"guard:start doc_id={doc_id[:8]} text_len={len(text or '')}")
    if not bool(getattr(settings, "enable_extract_llm_subprocess", True)):
        _trace_eval("guard:subprocess_disabled")
        return extract_from_text(text, schema, pages=pages, doc_id=doc_id)

    timeout_s = max(int(getattr(settings, "extract_llm_timeout_s", 240) or 240), 1)
    worker_path = _PROJECT_ROOT / "core" / "extract_fields_worker.py"
    payload = {
        "schema_name": schema.name,
        "text": text,
        "pages": pages,
        "doc_id": doc_id,
    }

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8") as input_tmp:
        json.dump(payload, input_tmp, ensure_ascii=False)
        input_path = Path(input_tmp.name)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as output_tmp:
        output_path = Path(output_tmp.name)
    _trace_eval("guard:payload_written")

    env = dict(os.environ)
    try:
        _trace_eval("guard:before_run")
        completed = subprocess.run(
            [sys.executable, str(worker_path), "--input-json", str(input_path), "--output-json", str(output_path)],
            cwd=str(_PROJECT_ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
        )
        _trace_eval(f"guard:after_run rc={completed.returncode} exists={output_path.exists()}")
        if completed.returncode != 0 or not output_path.exists():
            _trace_eval("guard:empty_result_due_worker_failure")
            return _empty_extraction_result(schema)
        result = json.loads(output_path.read_text(encoding="utf-8"))
        _trace_eval("guard:after_read_json")
        return result if isinstance(result, dict) else _empty_extraction_result(schema)
    except Exception:
        _trace_eval("guard:exception")
        return _empty_extraction_result(schema)
    finally:
        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass


def _iter_samples(dataset_root: Path) -> list[DocSample]:
    case_map = {
        "caso_uso_1_auditoria_hipotecaria": "credito_hipotecario",
        "caso_uso_2_auditoria_fiscal": "auditoria_fiscal",
        "caso_uso_3_kyc_onboarding": "kyc_onboarding",
    }
    # Carpetas válidas que contienen los documentos estructurados
    valid_folders = {"native_pdf", "scanned_blurry_pdf", "image_photo", "image_handwritten"}
    samples: list[DocSample] = []
    for case_dir_name, schema_name in case_map.items():
        case_dir = dataset_root / case_dir_name
        if not case_dir.exists():
            continue
        # Buscar PDFs y imágenes en TODAS las subcarpetas
        for ext in ["*.pdf", "*.jpg", "*.jpeg", "*.png"]:
            for doc_path in sorted(case_dir.rglob(ext)):
                # Verificar que la ruta contenga una carpeta válida en algún nivel
                path_parts = doc_path.parent.relative_to(case_dir).parts
                has_valid_folder = any(part in valid_folders for part in path_parts)
                if not has_valid_folder:
                    continue
                # Primero buscar ground truth en la carpeta del documento
                gt_path = _find_ground_truth_for_pdf(doc_path.name, doc_path.parent)
                # Si no lo encuentra, buscar en la carpeta raíz del caso
                if gt_path is None:
                    gt_path = _find_ground_truth_for_pdf(doc_path.name, case_dir)
                if gt_path is None:
                    continue
                samples.append(
                    DocSample(
                        case_id=case_dir_name,
                        pdf_path=doc_path,  # Aunque sea imagen, usamos el mismo campo
                        ground_truth_path=gt_path,
                        schema_name=schema_name,
                    )
                )
    return samples


def _find_ground_truth_for_pdf(pdf_filename: str, case_dir: Path) -> Path | None:
    name = pdf_filename.lower()
    # Patrón para documentos en subcarpetas (ej: credito_hipotecario_native_pdf_conforme_001.pdf)
    if "_conforme_" in name or "_no_conforme_" in name:
        stem = Path(pdf_filename).stem
        gt = case_dir / f"{stem}_ground_truth.json"
        if gt.exists():
            return gt
        # Si no, intentar quitar el tipo de documento del nombre
        parts = stem.split("_")
        if len(parts) >= 3:
            # Ej: credito_hipotecario_native_pdf_conforme_001 -> native_pdf_conforme_001 no, quitar el primer y segundo elemento?
            # Mejor probar con el nombre completo + _ground_truth.json en la misma carpeta
            pass
    # Patrones originales
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
    # Último intento: buscar cualquier JSON que termine con el mismo sufijo numérico
    # Ej: credito_hipotecario_native_pdf_conforme_001.pdf -> *001*.json
    match = re.search(r'(\d+)\.pdf$', pdf_filename.lower())
    if match:
        suffix = match.group(1)
        for gt_candidate in case_dir.glob(f"*_{suffix}_ground_truth.json"):
            if gt_candidate.exists():
                return gt_candidate
        for gt_candidate in case_dir.glob(f"*{suffix}.json"):
            if gt_candidate.exists() and "ground_truth" in gt_candidate.name.lower():
                return gt_candidate
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

    # document_correct arranca en True y pasa a False ante el primer error
    # relevante (campo con GT presente mal extraído, o campo espurio).
    document_correct = True

    for k in keys:
        gt_v = ground_truth.get(k)
        pred_v = extracted.get(k)

        gt_has = gt_v is not None and gt_v != ""
        pred_has = pred_v is not None and pred_v != ""
        if gt_has:
            gt_present += 1
        if pred_has:
            pred_present += 1

        values_match = _values_equal(pred_v, gt_v, float_tol=float_tol)

        # CORRECCIÓN: un campo solo cuenta como acierto real si el GT tenía
        # contenido. Un GT-vacío que coincide con pred-vacío NO es una
        # extracción correcta (nadie esperaba nada) y no debe inflar
        # exact_matches/precision/recall. Antes esto permitía recall > 1.0.
        if values_match and gt_has:
            exact_matches += 1
        elif not values_match:
            if gt_has and not pred_has:
                mismatch_type = "miss"
                document_correct = False
            elif (not gt_has) and pred_has:
                mismatch_type = "spurious"
                document_correct = False
            elif gt_has and pred_has:
                mismatch_type = "value"
                document_correct = False
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
    # field_accuracy: aciertos sobre campos con GT presente (denominador honesto,
    # ya no total_fields que premiaba los vacíos como "gratis").
    field_accuracy = (exact_matches / gt_present) if gt_present else 0.0

    return {
        "fields_total": total_fields,
        "fields_gt_present": gt_present,
        "fields_pred_present": pred_present,
        "exact_matches": exact_matches,
        "field_accuracy": field_accuracy,
        "match_rate": field_accuracy,  # alias retrocompatible
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "document_correct": document_correct,
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


def _has_llm_extractable_signal(text: str) -> bool:
    normalized = " ".join((text or "").split())
    if not normalized:
        return False
    word_like = re.findall(r"[A-Za-z0-9ÁÉÍÓÚÜÑáéíóúüñ]{2,}", normalized)
    digits = sum(ch.isdigit() for ch in normalized)
    letters = sum(ch.isalpha() for ch in normalized)
    return len(word_like) >= 3 and (letters + digits) >= 12


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
    schema: Any,
    degraded: bool,
    float_tol: float,
    pdf_name: str,
    backend: str,
    *,
    force_reading_mode: str = "auto",
    run_seed: int = 0,
) -> dict[str, Any]:
    doc_id = _sha256_hex(pdf_bytes)
    is_image = pdf_name.lower().endswith((".jpg", ".jpeg", ".png"))
    with extraction_policy_for_doc(pdf_name):
        if is_image:
            extracted_doc = extract_text_from_image_bytes(pdf_bytes, force_reading_mode=force_reading_mode)
        else:
            extracted_doc = extract_text_from_pdf_bytes(pdf_bytes, force_reading_mode=force_reading_mode)
    text = extracted_doc.get("text") or ""
    pages = extracted_doc.get("page_texts") or None

    if settings.enable_pii_redaction:
        text, _ = redact_pii(text)
        if isinstance(pages, list):
            pages = [redact_pii(p)[0] for p in pages]

    if degraded:
        text = _apply_degradation(text, seed=int(doc_id[:8], 16) ^ (int(run_seed) & 0xFFFFFFFF))
        pages = None

    def _is_empty_pred(v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, str) and v.strip() == "":
            return True
        return False

    # ============================================================
    # FRENTE-E: backend llm_flat_pipeline (baseline 2-pasos minimalista)
    # SIN classify_text, SIN LangGraph run_pipeline(), SIN audit_document score_confianza
    # ============================================================
    if backend == "llm_flat_pipeline":
        extractor_backend = "llm_flat_pipeline"
        extracted_fields: dict[str, Any] = {}
        has_input_text = bool((text or "").strip())
        if has_input_text:
            try:
                extracted_payload = _extract_from_text_guarded(
                    text=text,
                    schema=schema,
                    pages=pages,
                    doc_id=doc_id,
                )
                extracted_candidate = (
                    extracted_payload.get("fields") if isinstance(extracted_payload, dict) else None
                )
                if isinstance(extracted_candidate, dict):
                    extracted_fields = extracted_candidate
                elif isinstance(extracted_payload, dict):
                    extracted_fields = {
                        k: v for k, v in extracted_payload.items()
                        if k not in {"details", "meta", "metadata"}
                    }
            except Exception:
                extracted_fields = {}

        llm_returned_empty = (
            not extracted_fields
            or all(_is_empty_pred(v) for v in extracted_fields.values())
        )

        normalization = normalize_extracted(extracted_fields, schema)
        normalized = normalization["normalized"]
        validation = validate_extracted(normalized, schema)
        report = {
            "decision_rules": [],
            "document_status": "OK",
            "status_reason": "flat_pipeline_no_audit",
            "confidence_score": None,
            "confidence_signals": {},
            "field_details": {},
            "severity_counts": {"critical": 0, "warning": 0, "pass": 0},
        }
        return {
            "doc_id": doc_id,
            "schema_name": schema_name,
            "text_method": extracted_doc.get("method"),
            "pages": extracted_doc.get("pages"),
            "degraded_text": degraded,
            "extractor_backend": extractor_backend,
            "llm_returned_empty": llm_returned_empty,
            "extracted": normalized,
            "normalization": normalization,
            "validation": validation,
            "report": report,
            "reading_mode_used": force_reading_mode,
            "run_seed": int(run_seed),
        }

    # ============================================================
    # Flujo normal: backend in {auto, llm, heuristic} (con LangGraph audit)
    # ============================================================
    extractor_backend = "llm" if backend in {"auto", "llm"} else "heuristic"
    extracted_fields = {}
    field_details: dict[str, Any] = {}
    has_input_text = bool((text or "").strip())
    has_llm_signal = _has_llm_extractable_signal(text)
    if backend in {"auto", "llm"} and has_input_text and has_llm_signal:
        try:
            extracted_payload = _extract_from_text_guarded(
                text=text,
                schema=schema,
                pages=pages,
                doc_id=doc_id,
            )
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

    llm_returned_empty = (
        backend in {"auto", "llm"}
        and (
            not extracted_fields
            or all(_is_empty_pred(v) for v in extracted_fields.values())
        )
    )

    if backend == "heuristic":
        should_fallback = True
    elif backend == "auto":
        should_fallback = (
            not extracted_fields
            or all(_is_empty_pred(v) for v in extracted_fields.values())
        )
    else:
        should_fallback = False

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
        "llm_returned_empty": llm_returned_empty,
        "extracted": normalized,
        "normalization": normalization,
        "validation": validation,
        "report": report,
        "reading_mode_used": force_reading_mode,
        "run_seed": int(run_seed),
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
        choices=["auto", "llm", "heuristic", "llm_flat_pipeline"],
        help=(
            "Backend de extracción: auto (LLM + fallback heurístico), llm (solo LLM), "
            "heuristic (solo heurístico), llm_flat_pipeline (2 pasos: extraer+validar simple, "
            "sin clasificador, sin score_confianza auditor, sin LangGraph — ablación FRENTE-E)."
        ),
    )
    parser.add_argument(
        "--reading-mode",
        type=str,
        default="auto",
        choices=["auto", "pypdf_only", "ocr_only", "vlm_only"],
        help=(
            "Modo de lectura forzado para ablación FRENTE-D: auto (cascada normal), "
            "pypdf_only (SOLO texto embebido), ocr_only (SOLO OCR local sin VLM), "
            "vlm_only (SOLO Qwen2.5-VL multimodal sin OCR ni PyPDF)."
        ),
    )
    parser.add_argument(
        "--run-seed",
        type=int,
        default=0,
        help="Semilla RNG independiente para esta corrida (para repetibilidad Frente-F).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limita el número de documentos evaluados (0 = sin límite).",
    )
    parser.add_argument(
        "--only-doc-type",
        type=str,
        default="",
        choices=["", "native_pdf", "scanned_blurry_pdf", "image_photo", "image_handwritten"],
        help="Filtra la evaluación a un tipo de documento (por defecto: todos).",
    )
    args = parser.parse_args()

    # Aplicamos semilla de corrida para reproducibilidad (RNG interno + ENVs de workers)
    random.seed(args.run_seed)
    os.environ["DOCAUDIT_EVAL_RUN_SEED"] = str(int(args.run_seed or 0))

    # Fijamos modo lectura forzado vía ENV para que todo el pipeline lo vea
    if args.reading_mode and args.reading_mode != "auto":
        os.environ["DOCAUDIT_FORCE_READING_MODE"] = args.reading_mode
        try:
            settings.force_reading_mode = args.reading_mode
        except (ValueError, AttributeError):
            pass
    else:
        os.environ.pop("DOCAUDIT_FORCE_READING_MODE", None)
        try:
            delattr(settings, "force_reading_mode")
        except (AttributeError, ValueError):
            pass

    dataset_root = _resolve_under_project(args.dataset)
    samples = _iter_samples(dataset_root)
    if args.only_doc_type:
        needle = args.only_doc_type.lower()
        samples = [s for s in samples if needle in s.pdf_path.name.lower()]
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    out_path = _resolve_under_project(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    fieldnames = [
        "run_seed",
        "reading_mode",
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
        "document_correct",
        "llm_returned_empty",
        "rules_total",
        "rules_gt_evaluable",
        "rules_pred_evaluable",
        "rules_matches",
        "rules_accuracy",
        "rules_coverage",
        "latency_s",
        "ram_peak_mb",
        "mismatch_count",
    ]

    mismatches_out = out_path.with_suffix("")
    mismatches_dir = mismatches_out.parent / (mismatches_out.name + "_mismatches")
    mismatches_dir.mkdir(parents=True, exist_ok=True)

    summary_extra_metadata = {
        "run_seed": int(args.run_seed or 0),
        "reading_mode": args.reading_mode,
        "backend": args.backend,
        "dataset": str(dataset_root),
        "include_degraded": bool(args.include_degraded),
        "float_tol": float(args.float_tol),
        "total_samples_requested": len(samples),
        "filter_only_doc_type": args.only_doc_type or None,
        "limit": int(args.limit or 0),
    }

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        f.flush()

        total_samples = len(samples)
        print(f"\n{'='*80}")
        print(f"INICIANDO EVALUACIÓN DE {total_samples} DOCUMENTOS")
        print(f"  run_seed       = {args.run_seed}")
        print(f"  reading_mode   = {args.reading_mode}")
        print(f"  backend        = {args.backend}")
        print(f"  only_doc_type  = {args.only_doc_type or '(todos)'}")
        print(f"  include_deg.   = {bool(args.include_degraded)}")
        print(f"{'='*80}\n")

        for idx, sample in enumerate(samples, 1):
            # Mostrar progreso
            print(f"[{idx}/{total_samples}] Procesando: {sample.pdf_path.name}")
            print(f"  Caso: {sample.case_id} | Esquema: {sample.schema_name}")

            pdf_bytes = sample.pdf_path.read_bytes()
            gt = _load_json(sample.ground_truth_path)
            schema = load_schema(_PROJECT_ROOT / "schemas" / f"{sample.schema_name}.yaml")
            gt_norm = normalize_extracted(gt, schema)["normalized"]
            gt_validation = validate_extracted(gt_norm, schema)
            gt_report = audit_document(schema, gt_norm, gt_validation, field_details={})

            for degraded in ([False, True] if args.include_degraded else [False]):
                started = time.perf_counter()

                def _run():
                    return _run_single(
                        pdf_bytes=pdf_bytes,
                        schema_name=sample.schema_name,
                        schema=schema,
                        degraded=degraded,
                        float_tol=args.float_tol,
                        pdf_name=sample.pdf_path.name,
                        backend=args.backend,
                        force_reading_mode=args.reading_mode,
                        run_seed=args.run_seed,
                    )

                result, peak_rss_mb = _measure_peak_rss_mb(_run)
                elapsed_s = time.perf_counter() - started

                metrics = _compute_field_metrics(result["extracted"], gt, float_tol=args.float_tol)
                rule_metrics = _compute_rule_metrics(result.get("report", {}), gt_report)

                row = {
                    "run_seed": int(result.get("run_seed") if result.get("run_seed") is not None else (args.run_seed or 0)),
                    "reading_mode": str(result.get("reading_mode_used") if result.get("reading_mode_used") else (args.reading_mode or "auto")),
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
                    "document_correct": metrics["document_correct"],
                    "llm_returned_empty": result.get("llm_returned_empty", False),
                    "rules_total": rule_metrics["rules_total"],
                    "rules_gt_evaluable": rule_metrics["rules_gt_evaluable"],
                    "rules_pred_evaluable": rule_metrics["rules_pred_evaluable"],
                    "rules_matches": rule_metrics["rules_matches"],
                    "rules_accuracy": rule_metrics["rules_accuracy"],
                    "rules_coverage": rule_metrics["rules_coverage"],
                    "latency_s": elapsed_s,
                    "ram_peak_mb": peak_rss_mb,
                    "mismatch_count": len(metrics["mismatches"]),
                }
                rows.append(row)
                w.writerow(row)
                f.flush()

                mm_path = mismatches_dir / f"{sample.pdf_path.stem}{'_degraded' if degraded else ''}.json"
                mm_path.write_text(json.dumps(metrics["mismatches"], ensure_ascii=False, indent=2), encoding="utf-8")

                # Mostrar métricas del documento
                print(f"  [OK] Hecho | F1: {metrics['f1']:.3f} | Match rate: {metrics['match_rate']:.3f}")
                print(f"    Latencia: {elapsed_s:.1f}s | RAM peak: {peak_rss_mb:.1f}MB | Backend: {result.get('extractor_backend', 'N/A')}")

                # Calcular y mostrar progreso acumulado
                if idx > 0:
                    current_avg_f1 = sum(r['f1'] for r in rows) / len(rows)
                    current_avg_latency = sum(r['latency_s'] for r in rows) / len(rows)
                    elapsed_total = sum(r['latency_s'] for r in rows)
                    estimated_remaining = (total_samples - idx) * current_avg_latency
                    print(f"  Acumulado: F1 avg={current_avg_f1:.3f} | Tiempo total: {elapsed_total/60:.1f}min | Restante ~{estimated_remaining/60:.1f}min")

                print()

    if not rows:
        print(f"ERROR: no se encontraron muestras en: {dataset_root}")
        print("Sugerencia: ejecuta desde el root del repositorio o pasa --dataset con ruta absoluta.")
        return 2

    summary = _summarize(rows)
    summary["experiment_metadata"] = summary_extra_metadata
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*80}")
    print(f"EVALUACIÓN COMPLETA!")
    print(f"{'='*80}")
    print(f"Total ejecuciones: {len(rows)}")
    print(f"Global F1: {summary['overall']['micro_f1']:.3f}")
    print(f"Global Match rate: {summary['overall']['micro_match_rate']:.3f}")
    print(f"Latencia media: {summary['overall']['avg_latency_s']:.1f}s")
    print(f"RAM peak media: {summary['overall']['avg_ram_peak_mb']:.1f}MB")
    
    # Mostrar desglose por tipo de documento
    if "by_doc_type" in summary and summary["by_doc_type"]:
        print(f"\n--- Desglose por tipo de documento ---")
        for doc_type, metrics in summary["by_doc_type"].items():
            print(f"{doc_type}:")
            print(f"  F1: {metrics['micro_f1']:.3f} | Match: {metrics['micro_match_rate']:.3f}")
            print(f"  Latencia: {metrics['avg_latency_s']:.1f}s | RAM: {metrics['avg_ram_peak_mb']:.1f}MB")
    
    print(f"\nCSV: {out_path}")
    print(f"Resumen JSON: {summary_path}")
    print(f"Mismatches: {mismatches_dir}")
    print(f"{'='*80}\n")
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
        # micro_match_rate ahora sobre gt_present (campos que SÍ debían extraerse),
        # coherente con la corrección de field_accuracy. Ya no premia vacíos.
        match_rate = (exact / gt_present) if gt_present else 0.0
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

    def _micro_rules(sel: list[dict[str, Any]]) -> dict[str, Any]:
        rules_total = int(sum(int(r.get("rules_total") or 0) for r in sel))
        rules_gt_evaluable = int(sum(int(r.get("rules_gt_evaluable") or 0) for r in sel))
        rules_pred_evaluable = int(sum(int(r.get("rules_pred_evaluable") or 0) for r in sel))
        rules_matches = int(sum(int(r.get("rules_matches") or 0) for r in sel))
        micro_rules_accuracy = (rules_matches / rules_pred_evaluable) if rules_pred_evaluable else 0.0
        micro_rules_coverage = (rules_pred_evaluable / rules_total) if rules_total else 0.0
        return {
            "rules_total": rules_total,
            "rules_gt_evaluable": rules_gt_evaluable,
            "rules_pred_evaluable": rules_pred_evaluable,
            "rules_matches": rules_matches,
            "micro_rules_accuracy": micro_rules_accuracy,
            "micro_rules_coverage": micro_rules_coverage,
        }

    def _aggregate(sel: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(sel)
        docs_correct = int(sum(1 for r in sel if r.get("document_correct")))
        runs_heuristic = int(sum(1 for r in sel if r.get("extractor_backend") == "heuristic"))
        runs_llm = int(sum(1 for r in sel if r.get("extractor_backend") == "llm"))
        llm_empty = int(sum(1 for r in sel if r.get("llm_returned_empty")))
        out_block = {
            "runs": n,
            "avg_field_accuracy": _avg([r["match_rate"] for r in sel]),
            "avg_f1": _avg([r["f1"] for r in sel]),
            "avg_latency_s": _avg([r["latency_s"] for r in sel]),
            "p95_latency_s": _p95([r["latency_s"] for r in sel]),
            "avg_ram_peak_mb": _avg([r["ram_peak_mb"] for r in sel]),
            # document_exact_match: fracción de DOCUMENTOS con todos los campos
            # correctos. Esta es la lectura estándar de "Exact Match Rate".
            "document_exact_match": (docs_correct / n) if n else 0.0,
            "documents_correct": docs_correct,
            # Trazabilidad del backend: cuántos casos resolvió el VLM vs heurístico
            # y en cuántos el VLM devolvió vacío. Imprescindible para declarar
            # honestamente el origen de las métricas.
            "runs_llm": runs_llm,
            "runs_heuristic": runs_heuristic,
            "llm_returned_empty_count": llm_empty,
            "avg_rules_accuracy": _avg([float(r.get("rules_accuracy") or 0.0) for r in sel]),
            "avg_rules_coverage": _avg([float(r.get("rules_coverage") or 0.0) for r in sel]),
        }
        out_block.update(_micro(sel))
        out_block.update(_micro_rules(sel))
        return out_block

    out: dict[str, Any] = {"overall": {}, "by_case": {}, "by_degraded": {}, "by_doc_type": {}}
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

    # Agregar desglose por tipo de documento
    by_doc_type: dict[str, list[dict[str, Any]]] = {
        "native_pdf": [],
        "scanned_blurry_pdf": [],
        "image_photo": [],
        "image_handwritten": [],
    }
    for r in rows:
        pdf_name = r.get("pdf", "").lower()
        if "native_pdf" in pdf_name:
            by_doc_type["native_pdf"].append(r)
        elif "scanned_blurry_pdf" in pdf_name:
            by_doc_type["scanned_blurry_pdf"].append(r)
        elif "image_photo" in pdf_name:
            by_doc_type["image_photo"].append(r)
        elif "image_handwritten" in pdf_name:
            by_doc_type["image_handwritten"].append(r)
    
    for doc_type, sel in by_doc_type.items():
        if sel:
            out["by_doc_type"][doc_type] = _aggregate(sel)

    return out


if __name__ == "__main__":
    raise SystemExit(main())
