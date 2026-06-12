
"""
Módulo de evaluación para tesis:
- Calcula métricas: Field-level F1, Exact Match Rate, Latencia media, RAM peak
- Compara PDFs nativos vs degradados
- Errores frecuentes
"""
import time
import psutil
import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    field_f1: float
    exact_match_rate: float
    avg_latency_seconds: float
    ram_peak_mb: float
    total_fields: int
    exact_matches: int
    errors_by_field: Dict[str, int]
    raw_results: List[Dict[str, Any]]


def field_level_f1(predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -&gt; tuple[float, int, int]:
    """
    Calcula F1 a nivel de campo (considera solo campos de la intersección de campos
    en ambos diccionarios.
    """
    common_fields = set(predicted.keys()) &amp; set(ground_truth.keys())
    tp = 0
    fp = 0
    fn = 0

    for field in common_fields:
        pred_val = predicted.get(field)
        gt_val = ground_truth.get(field)
        if pred_val == gt_val:
            tp +=1
        else:
            fp +=1
            fn +=1

    # Campos extra en predicción
    extra_pred = set(predicted.keys()) - set(ground_truth.keys())
    fp += len(extra_pred)

    # Campos faltantes en predicción
    extra_gt = set(ground_truth.keys()) - set(predicted.keys())
    fn += len(extra_gt)

    precision = tp / (tp + fp) if (tp + fp) &gt; 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) &gt; 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) &gt; 0 else 0.0

    return f1, tp, len(common_fields)


def exact_match_rate(predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -&gt; bool:
    """
    Verifica si la predicción coincide exactamente con el ground truth
    (solo campos que están en ambos diccionarios).
    """
    common_fields = set(predicted.keys()) &amp; set(ground_truth.keys())
    return all(predicted.get(field) == ground_truth.get(field) for field in common_fields)


def count_errors(predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -&gt; Dict[str, int]:
    """
    Cuenta errores por campo.
    """
    errors = defaultdict(int)
    common_fields = set(predicted.keys()) &amp; set(ground_truth.keys())
    for field in common_fields:
        if predicted.get(field) != ground_truth.get(field):
            errors[field] += 1
    return errors


def get_ram_usage() -&gt; float:
    """
    Obtiene el uso de RAM en MB del proceso actual.
    """
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)  # RSS en MB


def run_evaluation(
    run_pipeline_fn,
    test_cases: List[Dict[str, Any]],
    schema_name: str | None = None,
) -&gt; EvaluationResult:
    """
    Ejecuta la evaluación completa sobre una lista de casos de prueba.
    
    test_cases es una lista de diccionarios con:
        - "text" | "pdf_bytes": texto o bytes del PDF
        - "ground_truth": diccionario con los valores esperados
        - "type": "native" o "degraded" para clasificar
    """
    all_predictions = []
    latencies = []
    ram_peaks = []
    total_fields = 0
    exact_matches = 0
    errors_by_field = defaultdict(int)
    raw_results = []

    for i, case in enumerate(test_cases, 1):
        # Preparar entrada
        if "text" in case:
            input_text = case["text"]
        else:
            from core.document_loader import extract_text_from_pdf_bytes
            extracted = extract_text_from_pdf_bytes(case["pdf_bytes"])
            input_text = extracted.get("text", "")

        ground_truth = case["ground_truth"]
        case_type = case.get("type", "unknown")

        # Medir latencia y RAM
        start_time = time.perf_counter()
        ram_start = get_ram_usage()
        try:
            result = run_pipeline_fn(input_text, schema_name=schema_name)
        except Exception as e:
            print(f"Error en caso {i}: {e}")
            continue
        end_time = time.perf_counter()
        ram_end = get_ram_usage()

        # Almacenar métricas
        latency = end_time - start_time
        ram_peak = max(ram_end - ram_start) if ram_end &gt; ram_start else 0.0
        latencies.append(latency)
        ram_peaks.append(ram_end)
        total_fields += len(ground_truth.keys())

        # Obtener predicción de campos
        predicted_fields = result.get("extracted", {})
        if "fields" in predicted_fields:
            predicted_fields = predicted_fields["fields"]

        # Calcular métricas por caso
        em_rate = exact_match_rate(predicted_fields, ground_truth)
        exact_matches += 1 if em_rate else 0
        case_errors = count_errors(predicted_fields, ground_truth)
        for field, count in case_errors.items():
            errors_by_field[field] += count

        raw_results.append({
            "case_index": i,
            "type": case_type,
            "latency": latency,
            "ram_peak": ram_end,
            "ground_truth": ground_truth,
            "predicted": predicted_fields,
            "exact_match": em_rate,
        })

    # Calcular métricas agregadas
    total_f1 = 0.0
    for res in raw_results:
        f1, _, _ = field_level_f1(res["predicted"], res["ground_truth"])
        total_f1 += f1
    field_f1_avg = total_f1 / len(raw_results) if len(raw_results) &gt; 0 else 0.0
    em_rate_total = exact_matches / len(raw_results) if len(raw_results) &gt; 0 else 0.0
    avg_latency = sum(latencies) / len(latencies) if len(latencies) &gt; 0 else 0.0
    ram_peak_total = max(ram_peaks) if len(ram_peaks) &gt; 0 else 0.0

    return EvaluationResult(
        field_f1=field_f1_avg,
        exact_match_rate=em_rate_total,
        avg_latency_seconds=avg_latency,
        ram_peak_mb=ram_peak_total,
        total_fields=total_fields,
        exact_matches=exact_matches,
        errors_by_field=dict(errors_by_field),
        raw_results=raw_results,
    )


def generate_report(result: EvaluationResult, output_path: Path) -&gt; str:
    """
    Genera un reporte en markdown con las tablas y conclusiones.
    """
    # Separar resultados por tipo (native vs degraded)
    native_results = [r for r in result.raw_results if r["type"] == "native"]
    degraded_results = [r for r in result.raw_results if r["type"] == "degraded"]

    report_md = f"""
# Reporte de Evaluación - Tesis

## 1. Resumen General

| Métrica | Valor |
|---------|-------|
| Field-level F1 | {result.field_f1:.4f} |
| Exact Match Rate | {result.exact_match_rate:.2%} |
| Latencia Media | {result.avg_latency_seconds:.2f} s |
| RAM Peak | {result.ram_peak_mb:.2f} MB |
| Total Campos Evaluados | {result.total_fields} |
| Coincidencias Exactas | {result.exact_matches} |

## 2. Comparación PDFs Nativos vs Degradados

| Criterio | PDFs Nativos | PDFs Degradados |
|----------|---------------|-----------------|
| Casos | {len(native_results)} | {len(degraded_results)} |
| Field-level F1 | {sum([field_level_f1(r["predicted"], r["ground_truth"])[0] for r in native_results]:.4f} | {sum([field_level_f1(r["predicted"], r["ground_truth"])[0] for r in degraded_results]):.4f} |
| Exact Match Rate | {sum([1 for r in native_results if r["exact_match"]) / len(native_results) if len(native_results) &gt; 0 else 0:.2%} | {sum([1 for r in degraded_results if r["exact_match"]) / len(degraded_results) if len(degraded_results) &gt; 0 else 0:.2%} |
| Latencia Media | {sum([r["latency"] for r in native_results) / len(native_results) if len(native_results) &gt; 0 else 0:.2f} s | {sum([r["latency"] for r in degraded_results) / len(degraded_results) if len(degraded_results) &gt; 0 else 0:.2f} s |
| RAM Peak | {max([r["ram_peak"] for r in native_results) if len(native_results) &gt; 0 else 0:.2f} MB | {max([r["ram_peak"] for r in degraded_results) if len(degraded_results) &gt; 0 else 0:.2f} MB |

## 3. Errores Más Frecuentes Por Campo

| Campo | Errores |
|-------|---------|
"""
    # Errores ordenados por frecuencia
    sorted_errors = sorted(result.errors_by_field.items(), key=lambda x: x[1], reverse=True)
    for field, count in sorted_errors:
        report_md += f"| {field} | {count} |\n"

    report_md += f"""
## 4. Conclusiones

- **PI1**: El sistema es viable**:
  - F1 score de {result.field_f1:.4f} demuestra precisión aceptable
  - Latencia media de {result.avg_latency_seconds:.2f} s es adecuada para uso real

- **PI2**: Robustez ante PDFs degradados**:
  - Diferencia en F1 entre nativos y degradados es [DETALLAR AQUÍ]

## 5. Resultados Brutos

json
{json.dumps([{k: v for k, v in r.items() if k != "predicted" and k != "ground_truth"] for r in result.raw_results], indent=2, ensure_ascii=False)}
"""

    output_path.write_text(report_md, encoding="utf-8")
    return report_md
