
"""
Genera un informe PDF para la tesis a partir de los logs de ejecucion.

El informe incluye:
- Tabla real de resultados con Field-level F1, Exact Match Rate, latencia media y RAM peak.
- Numero de campos evaluados y coincidencias exactas.
- Errores mas frecuentes por campo.
- Comparacion entre PDF nativo y PDF degradado.
- Respuesta explicita y basada en evidencia para PI1 y PI2.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from reportlab.platypus.flowables import HRFlowable
except ImportError:
    SimpleDocTemplate = None


PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_FOLDER = PROJECT_ROOT / "test_data" / "execution_logs"
REPORTS_FOLDER = PROJECT_ROOT / "reports"
REPORTS_FOLDER.mkdir(parents=True, exist_ok=True)

GROUND_TRUTH_DIRS = {
    "credito_hipotecario": PROJECT_ROOT / "data" / "sample_docs" / "caso_uso_1_auditoria_hipotecaria",
    "auditoria_fiscal": PROJECT_ROOT / "data" / "sample_docs" / "caso_uso_2_auditoria_fiscal",
    "kyc_onboarding": PROJECT_ROOT / "data" / "sample_docs" / "caso_uso_3_kyc_onboarding",
}

SCHEMA_METADATA = {
    "credito_hipotecario": {
        "fields": [
            "id_documento",
            "nombre_cliente",
            "dni_cliente",
            "monto_prestamo_eur",
            "tasa_interes",
            "tae",
            "plazo_meses",
            "cuota_mensual_eur",
            "ingresos_mensuales_eur",
            "ratio_endeudamiento",
            "fein_entregada",
            "fiae_entregada",
            "fecha_emision",
        ],
        "rules": [
            "H01: Importe de prestamo positivo",
            "H02: Tipo de interes en rango razonable",
            "H03: Documento con identificador presente",
            "H04: TAE en rango razonable cuando esta informada",
            "H05: Plazo positivo cuando esta informado",
            "H06: Esfuerzo financiero razonable",
            "H07: Ratio de endeudamiento razonable",
            "H08: FEIN y FIAE, si se informan, deben constar como entregadas",
        ],
    },
    "auditoria_fiscal": {
        "fields": [
            "razon_social_emisor",
            "direccion_emisor",
            "nif_emisor",
            "nombre_destinatario",
            "nif_destinatario",
            "num_factura",
            "serie_factura",
            "fecha_expedicion",
            "fecha_operacion",
            "descripcion_operacion",
            "base_imponible",
            "tipo_iva",
            "cuota_iva",
            "retencion_irpf",
            "importe_total",
        ],
        "rules": [
            "RF01: Validacion matematica del IVA",
            "RF02: Consistencia del importe total facturado",
            "RF03: Tipos de IVA legales",
            "RF04: Fecha de expedicion no futura",
            "RF05: Fecha de operacion no posterior a la expedicion",
            "RF06: Descripcion de operacion presente cuando se informa",
            "RF07: Numero de factura informado",
        ],
    },
    "kyc_onboarding": {
        "fields": [
            "nombre_titular",
            "primer_apellido",
            "segundo_apellido",
            "num_documento",
            "nacionalidad",
            "pais_residencia",
            "fecha_nacimiento",
            "fecha_expedicion",
            "fecha_caducidad",
            "autoridad_emisora",
            "domicilio_comprobante",
            "actividad_economica",
            "proposito_relacion",
            "pep",
            "titular_real_identificado",
            "coincidencia_sanciones",
            "nivel_riesgo",
        ],
        "rules": [
            "RK01: Mayoria de edad",
            "RK02: Documento de identidad no caducado",
            "RK03: Documento no marcado como No encontrado",
            "RK04: Fecha de expedicion no futura cuando esta informada",
            "RK05: Sin coincidencia positiva en listas de sanciones",
            "RK06: Si es PEP, el riesgo no debe quedar en nivel bajo",
            "RK07: Titular real identificado cuando el dato se informa",
            "RK08: Proposito de la relacion informado cuando esta presente",
        ],
    },
}

USE_CASE_PROFILES = {
    "credito_hipotecario": {
        "titulo": "Caso de uso hipotecario y diseno del esquema YAML",
        "analisis": (
            "Este caso evalua la extraccion de identidad del prestatario y coherencias documentales "
            "minimas de un contrato o escritura hipotecaria. El YAML prioriza trazabilidad, validacion "
            "de formato, esfuerzo financiero y elementos precontractuales relevantes antes de la revision humana."
        ),
        "normativa": [
            ["RGPD art. 5", "Minimizacion de datos", "Extraccion limitada a campos necesarios del esquema y procesamiento local."],
            ["RGPD art. 25", "Proteccion de datos desde el diseno", "Procesamiento local y controlado; sin transferencia a terceros en el flujo definido."],
            ["RGPD art. 22", "No decision automatizada sin garantias", "El informe se plantea como apoyo a revision humana, no como decision final automatica."],
            ["AI Act art. 13-14 y Anexo III", "Transparencia, supervision y trazabilidad", "Score de confianza, evidencias por campo, logs y supervison humana."],
            ["Circular 4/2017 BdE", "Analisis de solvencia", "El YAML ya incorpora cuota, ingresos, plazo y ratio de endeudamiento como base de capacidad de pago."],
            ["Ley 5/2019 LCCI", "Informacion precontractual hipotecaria", "Se incorporan FEIN/FIAE, TAE, plazo y comision de apertura como soporte de trazabilidad."],
        ],
        "vacios": [
            "Sigue faltando informacion mas fina de solvencia, como estabilidad laboral, otras deudas, historial crediticio y fuentes externas.",
            "Aun no se cubren todos los artefactos LCCI, como FIAE detallada, cuadro de amortizacion, gastos distribuidos y advertencias personalizadas.",
            "La validacion de esfuerzo financiero sigue siendo simplificada y debe contrastarse con criterios de negocio reales.",
        ],
        "protocolo": "10 PDFs nativos del corpus y 10 versiones degradadas sinteticas del mismo conjunto para comparar robustez OCR/documental.",
    },
    "auditoria_fiscal": {
        "titulo": "Caso de uso de auditoria fiscal",
        "analisis": (
            "Este caso verifica facturas de proveedor mediante extraccion estructurada y reglas de coherencia "
            "fiscal. El YAML ahora cubre mas campos formales de facturacion y reglas de consistencia "
            "matematica y temporal para detectar errores en una fase temprana."
        ),
        "normativa": [
            ["RGPD art. 5", "Minimizacion de datos", "Se tratan solo datos necesarios para la verificacion fiscal documental."],
            ["RGPD art. 25", "Proteccion desde el diseno", "Procesamiento local y trazabilidad de ejecuciones."],
            ["RD 1619/2012", "Requisitos formales de facturacion", "El YAML cubre emisor, destinatario, numeracion, fechas, descripcion e importes principales."],
            ["Ley 37/1992 del IVA", "Tipos impositivos y coherencia del IVA", "Las reglas validan tipos de IVA, cuota y consistencia matematica de totales."],
            ["Control interno / auditoria", "Rastreo y repetibilidad", "El sistema genera logs, reglas y salida explicable para revision."],
        ],
        "vacios": [
            "Aunque mejora la cobertura formal, aun faltan validaciones de correlatividad global y conciliacion con series historicas.",
            "No se evalua aun correlatividad global entre facturas del mismo emisor ni conciliacion con libros registro.",
            "Las retenciones siguen modeladas de forma opcional y conviene especializarlas por tipo de proveedor o servicio.",
        ],
        "protocolo": "10 facturas nativas del corpus y 10 versiones degradadas sinteticas con ruido, compresion y perdida de contraste.",
    },
    "kyc_onboarding": {
        "titulo": "Caso de uso KYC/onboarding",
        "analisis": (
            "Este caso valida identidad documental y datos basicos de onboarding. El YAML se orienta a "
            "diligencia debida minima y reforzada: identificacion, vigencia documental, mayoria de edad, "
            "domicilio, riesgo AML y controles basicos de PEP/sanciones."
        ),
        "normativa": [
            ["RGPD art. 5", "Minimizacion de datos", "El esquema se restringe a identidad, vigencia y domicilio del expediente."],
            ["RGPD art. 25", "Proteccion desde el diseno", "Procesamiento local, logs y campos estrictamente necesarios."],
            ["RGPD art. 22", "Supervision humana", "El resultado se formula como apoyo al analista antes del alta."],
            ["Ley 10/2010 PBC/FT", "Diligencia debida e identificacion formal", "El YAML incorpora documento, vigencia, titular real, PEP y proposito de la relacion."],
            ["RD 304/2014", "Desarrollo reglamentario PBC/FT", "Se añade perfilado minimo de riesgo y control de coincidencia en sanciones, con cobertura aun parcial."],
        ],
        "vacios": [
            "Aun faltan comprobaciones externas, como listas reales de sanciones, verificacion documental avanzada y fuentes independientes.",
            "Conviene separar con mas detalle las piezas del expediente: identidad, domicilio, actividad y titularidad real.",
            "El nivel de riesgo sigue siendo declarativo; faltan reglas compuestas y scoring AML mas completo.",
        ],
        "protocolo": "10 expedientes KYC nativos del corpus y 10 versiones degradadas sinteticas para medir sensibilidad de OCR y validacion.",
    },
}


def load_logs() -> list[dict[str, Any]]:
    logs: list[dict[str, Any]] = []
    for log_file in sorted(LOGS_FOLDER.glob("log_*.json")):
        try:
            logs.append(json.loads(log_file.read_text(encoding="utf-8")))
        except Exception as exc:
            print(f"Error cargando {log_file.name}: {exc}")
    return logs


def normalize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 4)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        compact = " ".join(text.split())
        number_candidate = compact.replace(".", "").replace(",", ".", 1)
        if re.fullmatch(r"-?\d+(?:\.\d+)?", number_candidate):
            try:
                return round(float(number_candidate), 4)
            except ValueError:
                return compact.casefold()
        return compact.casefold()
    return str(value).strip().casefold()


def values_match(predicted: Any, expected: Any) -> bool:
    return normalize_value(predicted) == normalize_value(expected)


def classify_condition(log_data: dict[str, Any]) -> str:
    file_type = str(log_data.get("file_type") or "").lower()
    method = str(log_data.get("method") or "").lower()
    if file_type == "native_pdf" and method not in {"easyocr", "vision", "ollama_vision", "qwen25-vl"}:
        return "native"
    if file_type in {"scanned_pdf", "image"} or method in {"easyocr", "vision", "ollama_vision", "qwen25-vl"}:
        return "degraded"
    return "unknown"


def infer_schema_name(log_data: dict[str, Any]) -> str | None:
    full_result = log_data.get("full_result")
    if isinstance(full_result, dict):
        schema = full_result.get("schema")
        if isinstance(schema, dict):
            name = schema.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
    return None


def infer_ground_truth_path(log_data: dict[str, Any]) -> Path | None:
    file_name = str(log_data.get("file_name") or "")
    schema_name = infer_schema_name(log_data)

    if schema_name in GROUND_TRUTH_DIRS:
        base_dir = GROUND_TRUTH_DIRS[schema_name]
        if schema_name == "credito_hipotecario":
            match = re.search(r"(\d{4})", file_name)
            if match:
                candidate = base_dir / f"ground_truth_esp_{match.group(1)}.json"
                if candidate.exists():
                    return candidate
        else:
            candidate = base_dir / (Path(file_name).stem + ".json")
            if candidate.exists():
                return candidate

    stem = Path(file_name).stem
    for candidate in PROJECT_ROOT.glob("data/sample_docs/**/*.json"):
        if candidate.stem == stem:
            return candidate

    suffix_match = re.search(r"(\d{4}|\d+)", stem)
    if suffix_match:
        suffix = suffix_match.group(1)
        for candidate in PROJECT_ROOT.glob("data/sample_docs/**/*.json"):
            if suffix in candidate.stem:
                return candidate
    return None


def extract_predicted_fields(log_data: dict[str, Any]) -> dict[str, Any]:
    extracted_fields = log_data.get("extracted_fields")
    if isinstance(extracted_fields, dict) and extracted_fields:
        return extracted_fields

    full_result = log_data.get("full_result")
    if not isinstance(full_result, dict):
        return {}

    extracted = full_result.get("extracted")
    if isinstance(extracted, dict) and extracted:
        return extracted

    extracted_raw = full_result.get("extracted_raw")
    if isinstance(extracted_raw, dict):
        fields = extracted_raw.get("fields")
        if isinstance(fields, dict):
            return fields
    return {}


def evaluate_case(log_data: dict[str, Any]) -> dict[str, Any]:
    gt_path = infer_ground_truth_path(log_data)
    ground_truth = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path and gt_path.exists() else None
    predicted = extract_predicted_fields(log_data)
    condition = classify_condition(log_data)
    schema_name = infer_schema_name(log_data) or "-"

    case_result: dict[str, Any] = {
        "file_name": str(log_data.get("file_name") or "-"),
        "schema_name": schema_name,
        "condition": condition,
        "method": str(log_data.get("method") or "-"),
        "latency_seconds": float(log_data.get("processing_time_seconds") or 0.0),
        "ram_mb": float(log_data.get("ram_used_mb") or 0.0),
        "ground_truth_path": str(gt_path) if gt_path else None,
        "ground_truth_found": ground_truth is not None,
        "evaluated_fields": 0,
        "exact_field_matches": 0,
        "field_f1": None,
        "exact_match_document": None,
        "errors": [],
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }

    if not isinstance(ground_truth, dict):
        return case_result

    expected_fields = list(ground_truth.keys())
    case_result["evaluated_fields"] = len(expected_fields)
    predicted_keys = set(predicted.keys())
    exact_matches = 0
    tp = 0
    fp = 0
    fn = 0
    errors: list[dict[str, Any]] = []

    for field in expected_fields:
        expected_value = ground_truth.get(field)
        has_prediction = field in predicted_keys
        predicted_value = predicted.get(field)

        if has_prediction and values_match(predicted_value, expected_value):
            tp += 1
            exact_matches += 1
            continue

        fn += 1
        if has_prediction:
            fp += 1
            error_type = "value_mismatch"
        else:
            error_type = "missing_field"

        errors.append(
            {
                "field": field,
                "expected": expected_value,
                "predicted": predicted_value if has_prediction else None,
                "error_type": error_type,
            }
        )

    extra_fields = sorted(predicted_keys - set(expected_fields))
    for field in extra_fields:
        fp += 1
        errors.append(
            {
                "field": field,
                "expected": None,
                "predicted": predicted.get(field),
                "error_type": "extra_field",
            }
        )

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    field_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    case_result.update(
        {
            "exact_field_matches": exact_matches,
            "field_f1": field_f1,
            "exact_match_document": exact_matches == len(expected_fields) and not extra_fields,
            "errors": errors,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    )
    return case_result


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def load_schema_metadata(schema_name: str) -> dict[str, Any]:
    metadata = SCHEMA_METADATA.get(schema_name, {})
    corpus_dir = GROUND_TRUTH_DIRS.get(schema_name)
    pdf_count = len(list(corpus_dir.glob("*.pdf"))) if corpus_dir and corpus_dir.exists() else 0
    ground_truth_count = len(list(corpus_dir.glob("*.json"))) if corpus_dir and corpus_dir.exists() else 0

    return {
        "fields": metadata.get("fields", []),
        "rules": metadata.get("rules", []),
        "pdf_count": pdf_count,
        "ground_truth_count": ground_truth_count,
    }


def summarize_subset(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        return {
            "cases": 0,
            "field_f1": None,
            "exact_match_rate": None,
            "avg_latency_seconds": None,
            "ram_peak_mb": None,
            "total_fields": 0,
            "exact_field_matches": 0,
        }

    tp = sum(int(case.get("tp", 0)) for case in cases)
    fp = sum(int(case.get("fp", 0)) for case in cases)
    fn = sum(int(case.get("fn", 0)) for case in cases)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    field_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "cases": len(cases),
        "field_f1": field_f1,
        "exact_match_rate": sum(1 for case in cases if case.get("exact_match_document")) / len(cases),
        "avg_latency_seconds": safe_mean([float(case.get("latency_seconds", 0.0)) for case in cases]),
        "ram_peak_mb": max((float(case.get("ram_mb", 0.0)) for case in cases), default=0.0),
        "total_fields": sum(int(case.get("evaluated_fields", 0)) for case in cases),
        "exact_field_matches": sum(int(case.get("exact_field_matches", 0)) for case in cases),
    }


def summarize_by_use_case(case_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    evaluated = [case for case in case_results if case.get("ground_truth_found")]
    summary: dict[str, dict[str, Any]] = {}
    for schema_name in USE_CASE_PROFILES:
        subset = [case for case in evaluated if case.get("schema_name") == schema_name]
        summary[schema_name] = summarize_subset(subset)
    return summary


def summarize_cases(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [case for case in case_results if case.get("ground_truth_found")]
    tp = sum(int(case.get("tp", 0)) for case in evaluated)
    fp = sum(int(case.get("fp", 0)) for case in evaluated)
    fn = sum(int(case.get("fn", 0)) for case in evaluated)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    field_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    exact_match_rate = (
        sum(1 for case in evaluated if case.get("exact_match_document")) / len(evaluated) if evaluated else 0.0
    )
    total_fields = sum(int(case.get("evaluated_fields", 0)) for case in evaluated)
    exact_field_matches = sum(int(case.get("exact_field_matches", 0)) for case in evaluated)
    avg_latency = safe_mean([float(case.get("latency_seconds", 0.0)) for case in evaluated])
    ram_peak = max((float(case.get("ram_mb", 0.0)) for case in evaluated), default=0.0)

    error_counter = Counter()
    error_type_counter = Counter()
    for case in evaluated:
        for error in case.get("errors", []):
            error_counter[error["field"]] += 1
            error_type_counter[error["error_type"]] += 1

    by_condition: dict[str, dict[str, Any]] = {}
    for condition in ("native", "degraded"):
        subset = [case for case in evaluated if case.get("condition") == condition]
        s_tp = sum(int(case.get("tp", 0)) for case in subset)
        s_fp = sum(int(case.get("fp", 0)) for case in subset)
        s_fn = sum(int(case.get("fn", 0)) for case in subset)
        s_precision = s_tp / (s_tp + s_fp) if (s_tp + s_fp) else 0.0
        s_recall = s_tp / (s_tp + s_fn) if (s_tp + s_fn) else 0.0
        s_f1 = 2 * s_precision * s_recall / (s_precision + s_recall) if (s_precision + s_recall) else 0.0
        by_condition[condition] = {
            "cases": len(subset),
            "field_f1": s_f1 if subset else None,
            "exact_match_rate": (
                sum(1 for case in subset if case.get("exact_match_document")) / len(subset) if subset else None
            ),
            "avg_latency_seconds": safe_mean([float(case.get("latency_seconds", 0.0)) for case in subset]) if subset else None,
            "ram_peak_mb": max((float(case.get("ram_mb", 0.0)) for case in subset), default=0.0) if subset else None,
            "total_fields": sum(int(case.get("evaluated_fields", 0)) for case in subset),
            "exact_field_matches": sum(int(case.get("exact_field_matches", 0)) for case in subset),
        }

    return {
        "logs_loaded": len(case_results),
        "cases_evaluated": len(evaluated),
        "cases_without_ground_truth": len(case_results) - len(evaluated),
        "field_f1": field_f1,
        "exact_match_rate": exact_match_rate,
        "avg_latency_seconds": avg_latency,
        "ram_peak_mb": ram_peak,
        "total_fields": total_fields,
        "exact_field_matches": exact_field_matches,
        "error_counter": error_counter,
        "error_type_counter": error_type_counter,
        "by_condition": by_condition,
        "by_use_case": summarize_by_use_case(case_results),
    }


def fmt_metric(value: float | None, *, percent: bool = False, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if percent:
        return f"{value * 100:.2f}%"
    return f"{value:.{digits}f}"


def build_pi_answers(summary: dict[str, Any]) -> tuple[str, str]:
    pi1 = (
        "PI1: Con los experimentos evaluados, el sistema demuestra viabilidad empirica "
        f"para la extraccion estructurada. Sobre {summary['cases_evaluated']} caso(s) con ground truth, "
        f"obtiene Field-level F1 = {summary['field_f1']:.4f}, Exact Match Rate = {summary['exact_match_rate']:.2%}, "
        f"latencia media = {summary['avg_latency_seconds']:.2f} s y RAM peak = {summary['ram_peak_mb']:.2f} MB."
    )

    degraded = summary["by_condition"]["degraded"]
    native = summary["by_condition"]["native"]
    if degraded["cases"] == 0:
        pi2 = (
            "PI2: No existe evidencia empirica suficiente para responder esta pregunta con el experimento actual, "
            "porque no hay casos evaluados de PDF degradado. El informe deja esta ausencia explicitamente "
            "y evita concluir robustez sin datos observados."
        )
    else:
        delta_f1 = (native["field_f1"] or 0.0) - (degraded["field_f1"] or 0.0)
        pi2 = (
            "PI2: La comparacion nativo vs degradado se responde con evidencia observada. "
            f"Casos nativos = {native['cases']}, casos degradados = {degraded['cases']}. "
            f"Field-level F1 nativo = {fmt_metric(native['field_f1'])}, "
            f"Field-level F1 degradado = {fmt_metric(degraded['field_f1'])}, "
            f"diferencia absoluta = {delta_f1:.4f}."
        )
    return pi1, pi2


def plot_group_metrics(summary: dict[str, Any], output_path: Path) -> bool:
    if plt is None:
        return False

    native = summary["by_condition"]["native"]
    degraded = summary["by_condition"]["degraded"]
    labels = ["Field F1", "Exact Match", "Latencia", "RAM Peak"]
    native_values = [
        native["field_f1"] or 0.0,
        native["exact_match_rate"] or 0.0,
        native["avg_latency_seconds"] or 0.0,
        native["ram_peak_mb"] or 0.0,
    ]
    degraded_values = [
        degraded["field_f1"] or 0.0,
        degraded["exact_match_rate"] or 0.0,
        degraded["avg_latency_seconds"] or 0.0,
        degraded["ram_peak_mb"] or 0.0,
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    x_positions = list(range(len(labels)))
    width = 0.35
    ax.bar([x - width / 2 for x in x_positions], native_values, width, label="Nativo", color="#2e86de")
    ax.bar([x + width / 2 for x in x_positions], degraded_values, width, label="Degradado", color="#e67e22")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_title("Comparacion de metricas por condicion")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_top_errors(summary: dict[str, Any], output_path: Path) -> bool:
    if plt is None or not summary["error_counter"]:
        return False

    top_errors = summary["error_counter"].most_common(8)
    labels = [field for field, _ in top_errors]
    values = [count for _, count in top_errors]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color="#c0392b")
    ax.set_title("Errores mas frecuentes por campo")
    ax.set_ylabel("Frecuencia")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def create_plots(summary: dict[str, Any], plots_folder: Path) -> list[Path]:
    plots_folder.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    comparison_plot = plots_folder / "comparison_metrics.png"
    if plot_group_metrics(summary, comparison_plot):
        generated.append(comparison_plot)

    errors_plot = plots_folder / "top_errors.png"
    if plot_top_errors(summary, errors_plot):
        generated.append(errors_plot)

    return generated


def build_table(data: list[list[Any]], col_widths: list[float]) -> Table:
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def bullets_to_html(items: list[str]) -> str:
    return "<br/>".join(f"- {item}" for item in items)


def generate_pdf_report(
    summary: dict[str, Any],
    case_results: list[dict[str, Any]],
    plots: list[Path],
    output_pdf_path: Path,
) -> None:
    if SimpleDocTemplate is None:
        raise RuntimeError("reportlab no esta instalado. Instala 'reportlab' para generar el PDF.")

    doc = SimpleDocTemplate(
        str(output_pdf_path),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.7 * cm,
        bottomMargin=1.7 * cm,
    )
    story: list[Any] = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], fontSize=19, leading=24, spaceAfter=12)
    h1_style = ParagraphStyle("H1Custom", parent=styles["Heading1"], fontSize=15, leading=19, textColor=colors.darkblue)
    h2_style = ParagraphStyle("H2Custom", parent=styles["Heading2"], fontSize=12, leading=16, textColor=colors.HexColor("#1f4e79"))
    normal_style = ParagraphStyle("BodyCustom", parent=styles["Normal"], fontSize=9.5, leading=13)

    pi1_text, pi2_text = build_pi_answers(summary)

    story.append(Paragraph("Informe de Evaluacion Empirica - DocAudit Agent", title_style))
    story.append(Paragraph(f"Fecha de generacion: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("1. Evidencia experimental disponible", h1_style))
    story.append(
        Paragraph(
            (
                f"Se cargaron {summary['logs_loaded']} log(s) y se pudieron evaluar {summary['cases_evaluated']} caso(s) "
                f"contra ground truth. Casos sin ground truth emparejado: {summary['cases_without_ground_truth']}."
            ),
            normal_style,
        )
    )
    story.append(Spacer(1, 0.25 * cm))

    story.append(Paragraph("2. Tabla real de resultados globales", h1_style))
    overall_table = build_table(
        [
            ["Metrica", "Valor"],
            ["Field-level F1", fmt_metric(summary["field_f1"])],
            ["Exact Match Rate", fmt_metric(summary["exact_match_rate"], percent=True)],
            ["Latencia media", f"{summary['avg_latency_seconds']:.2f} s"],
            ["RAM peak", f"{summary['ram_peak_mb']:.2f} MB"],
            ["Numero de campos evaluados", str(summary["total_fields"])],
            ["Coincidencias exactas de campo", str(summary["exact_field_matches"])],
        ],
        [8.0 * cm, 7.0 * cm],
    )
    story.append(overall_table)
    story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph("3. Comparacion entre PDF nativo y PDF degradado", h1_style))
    native = summary["by_condition"]["native"]
    degraded = summary["by_condition"]["degraded"]
    comparison_table = build_table(
        [
            ["Criterio", "Nativo", "Degradado"],
            ["Casos evaluados", str(native["cases"]), str(degraded["cases"])],
            ["Field-level F1", fmt_metric(native["field_f1"]), fmt_metric(degraded["field_f1"])],
            ["Exact Match Rate", fmt_metric(native["exact_match_rate"], percent=True), fmt_metric(degraded["exact_match_rate"], percent=True)],
            ["Latencia media", f"{native['avg_latency_seconds']:.2f} s" if native["avg_latency_seconds"] is not None else "N/A", f"{degraded['avg_latency_seconds']:.2f} s" if degraded["avg_latency_seconds"] is not None else "N/A"],
            ["RAM peak", f"{native['ram_peak_mb']:.2f} MB" if native["ram_peak_mb"] is not None else "N/A", f"{degraded['ram_peak_mb']:.2f} MB" if degraded["ram_peak_mb"] is not None else "N/A"],
            ["Campos evaluados", str(native["total_fields"]), str(degraded["total_fields"])],
            ["Coincidencias exactas", str(native["exact_field_matches"]), str(degraded["exact_field_matches"])],
        ],
        [6.0 * cm, 4.0 * cm, 4.0 * cm],
    )
    story.append(comparison_table)
    story.append(Spacer(1, 0.35 * cm))

    if plots:
        story.append(Paragraph("4. Graficos", h1_style))
        for plot_path in plots:
            story.append(Image(str(plot_path), width=16 * cm, height=8 * cm))
            story.append(Spacer(1, 0.2 * cm))
    else:
        story.append(Paragraph("4. Graficos", h1_style))
        story.append(Paragraph("No se generaron graficos porque matplotlib no esta disponible o no habia datos suficientes.", normal_style))
    story.append(Spacer(1, 0.25 * cm))

    story.append(Paragraph("5. Errores mas frecuentes", h1_style))
    error_rows = [["Campo", "Frecuencia"]]
    for field, count in summary["error_counter"].most_common(10):
        error_rows.append([field, str(count)])
    if len(error_rows) == 1:
        error_rows.append(["Sin errores", "0"])
    story.append(build_table(error_rows, [9.0 * cm, 5.0 * cm]))
    story.append(Spacer(1, 0.25 * cm))

    story.append(Paragraph("6. Resultados por documento", h1_style))
    per_doc_rows = [["Archivo", "Condicion", "Campos", "Exactos", "F1", "EM", "Latencia", "RAM"]]
    for case in case_results:
        if not case["ground_truth_found"]:
            continue
        per_doc_rows.append(
            [
                case["file_name"][:28],
                case["condition"],
                str(case["evaluated_fields"]),
                str(case["exact_field_matches"]),
                fmt_metric(case["field_f1"]),
                "SI" if case["exact_match_document"] else "NO",
                f"{case['latency_seconds']:.2f} s",
                f"{case['ram_mb']:.2f}",
            ]
        )
    if len(per_doc_rows) == 1:
        per_doc_rows.append(["Sin casos evaluados", "-", "0", "0", "N/A", "N/A", "0.00 s", "0.00"])
    story.append(build_table(per_doc_rows, [4.5 * cm, 2.2 * cm, 1.6 * cm, 1.8 * cm, 1.5 * cm, 1.4 * cm, 2.0 * cm, 1.5 * cm]))
    story.append(PageBreak())

    story.append(Paragraph("7. Analisis de los tres casos de uso y diseno del esquema YAML", h1_style))
    for schema_name, profile in USE_CASE_PROFILES.items():
        schema_meta = load_schema_metadata(schema_name)
        schema_summary = summary["by_use_case"].get(schema_name, {})
        field_names = [str(field) for field in schema_meta["fields"][:12]]
        rule_names = [str(rule) for rule in schema_meta["rules"][:8]]

        story.append(Paragraph(profile["titulo"], h2_style))
        story.append(Paragraph(profile["analisis"], normal_style))
        story.append(Spacer(1, 0.15 * cm))
        story.append(
            build_table(
                [
                    ["Indicador", "Valor"],
                    ["PDFs del corpus", str(schema_meta["pdf_count"])],
                    ["Ground truth del corpus", str(schema_meta["ground_truth_count"])],
                    ["Casos evaluados en logs", str(schema_summary.get("cases", 0))],
                    ["Field-level F1 observado", fmt_metric(schema_summary.get("field_f1"))],
                    ["Exact Match Rate observado", fmt_metric(schema_summary.get("exact_match_rate"), percent=True)],
                    ["Latencia media observada", f"{schema_summary['avg_latency_seconds']:.2f} s" if schema_summary.get("avg_latency_seconds") is not None else "N/A"],
                    ["RAM peak observado", f"{schema_summary['ram_peak_mb']:.2f} MB" if schema_summary.get("ram_peak_mb") is not None else "N/A"],
                    ["Campos definidos en YAML", str(len(schema_meta["fields"]))],
                    ["Reglas definidas en YAML", str(len(schema_meta["rules"]))],
                ],
                [8.0 * cm, 7.0 * cm],
            )
        )
        story.append(Spacer(1, 0.15 * cm))
        story.append(Paragraph(f"<b>Campos principales del YAML:</b> {', '.join(field_names) if field_names else 'N/A'}", normal_style))
        story.append(Paragraph(f"<b>Reglas principales:</b> {'; '.join(rule_names) if rule_names else 'N/A'}", normal_style))
        story.append(Spacer(1, 0.1 * cm))
        story.append(build_table([["Norma y articulo", "Exigencia", "Decision de diseno en DocAudit Agent"]] + profile["normativa"], [4.0 * cm, 5.0 * cm, 7.0 * cm]))
        story.append(Spacer(1, 0.1 * cm))
        story.append(Paragraph(f"<b>Vacios o ampliaciones recomendadas:</b><br/>{bullets_to_html(profile['vacios'])}", normal_style))
        story.append(Spacer(1, 0.25 * cm))

    story.append(PageBreak())
    story.append(Paragraph("8. Protocolo experimental alineado con las normativas", h1_style))
    story.append(
        Paragraph(
            (
                "Se recomienda evaluar los tres casos de uso con el mismo diseno experimental: corpus nativo del proyecto, "
                "versiones degradadas sinteticas a partir del mismo ground truth, ejecucion local, logging completo, "
                "y revision humana de las salidas antes de cualquier conclusion operativa."
            ),
            normal_style,
        )
    )
    story.append(Spacer(1, 0.15 * cm))
    protocol_rows = [["Caso", "Diseno experimental recomendado", "Justificacion normativa"]]
    for _, profile in USE_CASE_PROFILES.items():
        protocol_rows.append(
            [
                profile["titulo"][:30],
                profile["protocolo"],
                "Trazabilidad, supervision humana, minimizacion y comparacion nativo/degradado con misma verdad terreno.",
            ]
        )
    story.append(build_table(protocol_rows, [4.0 * cm, 7.0 * cm, 5.0 * cm]))
    story.append(Spacer(1, 0.15 * cm))
    story.append(
        Paragraph(
            (
                "<b>Criterios minimos de reporte para tesis:</b><br/>"
                "- Reportar por cada caso de uso: Field-level F1, Exact Match Rate, latencia media y RAM peak.<br/>"
                "- Incluir numero de campos evaluados, coincidencias exactas y errores mas frecuentes.<br/>"
                "- Separar resultados de PDF nativo y PDF degradado sobre el mismo conjunto base.<br/>"
                "- Mantener supervision humana y no presentar la salida como decision automatizada final."
            ),
            normal_style,
        )
    )
    story.append(Spacer(1, 0.25 * cm))

    story.append(Paragraph("9. Respuesta explicita a PI1 y PI2", h1_style))
    story.append(Paragraph(pi1_text, normal_style))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(pi2_text, normal_style))
    story.append(Spacer(1, 0.25 * cm))

    story.append(Paragraph("10. Conclusiones basadas en datos", h1_style))
    if summary["cases_evaluated"] == 0:
        conclusion_text = (
            "No se puede cerrar una conclusion empirica porque no existen casos evaluados contra ground truth. "
            "Antes de redactar la tesis, es necesario ejecutar el experimento y regenerar este informe."
        )
    else:
        conclusion_text = (
            f"Las conclusiones de este informe se apoyan en {summary['cases_evaluated']} caso(s) evaluados y "
            f"{summary['total_fields']} campo(s) contrastados. El sistema alcanza {summary['field_f1']:.4f} de Field-level F1, "
            f"{summary['exact_match_rate']:.2%} de Exact Match Rate, una latencia media de {summary['avg_latency_seconds']:.2f} s "
            f"y un RAM peak observado de {summary['ram_peak_mb']:.2f} MB. "
            "Estos valores deben citarse en la tesis como evidencia empirica disponible, indicando de forma transparente "
            "el tamano muestral y la ausencia de datos degradados si aplica."
        )
    story.append(Paragraph(conclusion_text, normal_style))

    doc.build(story)


def main() -> None:
    print("=" * 80)
    print("Generando informe de tesis con metricas empiricas...")
    print("=" * 80)

    logs = load_logs()
    if not logs:
        print("No hay logs en test_data/execution_logs.")
        print("Procesa documentos en la app antes de ejecutar este informe.")
        sys.exit(0)

    case_results = [evaluate_case(log) for log in logs]
    summary = summarize_cases(case_results)

    plots_folder = PROJECT_ROOT / "temp_plots"
    plots = create_plots(summary, plots_folder)

    output_pdf_path = REPORTS_FOLDER / f"Informe_Tesis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    print(f"Logs cargados: {summary['logs_loaded']}")
    print(f"Casos evaluados con ground truth: {summary['cases_evaluated']}")

    try:
        generate_pdf_report(summary, case_results, plots, output_pdf_path)
        print(f"Informe PDF generado en: {output_pdf_path}")
    except RuntimeError as exc:
        print(str(exc))
        print("Instala dependencias con: py -m pip install reportlab matplotlib")

    print("=" * 80)


if __name__ == "__main__":
    main()
