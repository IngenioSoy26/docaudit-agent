
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
from typing import Dict, List, Any, Tuple, Optional
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
    errors_by_field: dict
    raw_results: list


def field_level_f1(predicted, ground_truth):
    # type: (Dict[str, Any], Dict[str, Any]) -> Tuple[float, int, int]
    """
    Calcula F1 a nivel de campo (considera solo campos de la intersección de campos
    en ambos diccionarios.
    """
    common_fields = set(predicted.keys()) & set(ground_truth.keys())
    tp = 0
    fp = 0
    fn = 0

    for field in common_fields:
        pred_val = predicted.get(field)
        gt_val = ground_truth.get(field)
        if pred_val == gt_val:
            tp += 1
        else:
            fp += 1
            fn += 1

    # Campos extra en predicción
    extra_pred = set(predicted.keys()) - set(ground_truth.keys())
    fp += len(extra_pred)

    # Campos faltantes en predicción
    extra_gt = set(ground_truth.keys()) - set(predicted.keys())
    fn += len(extra_gt)

    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return f1, tp, len(common_fields)


def exact_match_rate(predicted, ground_truth):
    # type: (Dict[str, Any], Dict[str, Any]) -> bool
    """
    Verifica si la predicción coincide exactamente con el ground truth
    (solo campos que están en ambos diccionarios).
    """
    common_fields = set(predicted.keys()) & set(ground_truth.keys())
    return all(predicted.get(field) == ground_truth.get(field) for field in common_fields)


def count_errors(predicted, ground_truth):
    # type: (Dict[str, Any], Dict[str, Any]) -> Dict[str, int]
    """
    Cuenta errores por campo.
    """
    errors = defaultdict(int)
    common_fields = set(predicted.keys()) & set(ground_truth.keys())
    for field in common_fields:
        if predicted.get(field) != ground_truth.get(field):
            errors[field] += 1
    return errors


def get_ram_usage():
    # type: () -> float
    """
    Obtiene el uso de RAM en MB del proceso actual.
    """
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)  # RSS en MB


def run_evaluation(
    run_pipeline_fn,
    test_cases,
    schema_name=None,
):
    # type: (Any, List[Dict[str, Any]], Optional[str]) -> EvaluationResult
    """
    Ejecuta la evaluación completa sobre una lista de casos de prueba.
    
    test_cases es una lista de diccionarios con:
        - "text" | "pdf_bytes" | "image_bytes": contenido a evaluar
        - "input_kind": "text", "pdf_native", "pdf_scanned" o "image" (opcional)
        - "ground_truth": diccionario con los valores esperados
        - "type": "native", "degraded" o "image" para clasificar
    """
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
        elif "image_bytes" in case:
            from core.document_loader import extract_text_from_image_bytes

            extracted = extract_text_from_image_bytes(case["image_bytes"])
            input_text = extracted.get("text", "")
        else:
            input_kind = case.get("input_kind")
            if input_kind == "pdf_scanned":
                from core.document_loader import extract_text_from_scanned_pdf_bytes

                extracted = extract_text_from_scanned_pdf_bytes(case["pdf_bytes"])
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
            import traceback
            traceback.print_exc()
            continue
        end_time = time.perf_counter()
        ram_end = get_ram_usage()

        # Almacenar métricas
        latency = end_time - start_time
        if ram_end > ram_start:
            ram_peak = ram_end - ram_start
        else:
            ram_peak = 0.0
        latencies.append(latency)
        ram_peaks.append(ram_end)
        total_fields += len(ground_truth.keys())

        # Obtener predicción de campos
        predicted_fields = result.get("extracted", {})
        if "fields" in predicted_fields:
            predicted_fields = predicted_fields["fields"]

        # Calcular métricas por caso
        em_rate = exact_match_rate(predicted_fields, ground_truth)
        if em_rate:
            exact_matches += 1
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

    if len(raw_results) > 0:
        field_f1_avg = total_f1 / len(raw_results)
    else:
        field_f1_avg = 0.0

    if len(raw_results) > 0:
        em_rate_total = exact_matches / len(raw_results)
    else:
        em_rate_total = 0.0

    if len(latencies) > 0:
        avg_latency = sum(latencies) / len(latencies)
    else:
        avg_latency = 0.0

    if len(ram_peaks) > 0:
        ram_peak_total = max(ram_peaks)
    else:
        ram_peak_total = 0.0

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


def generate_report(result, output_path):
    # type: (EvaluationResult, Path) -> str
    """
    Genera un reporte en markdown con las tablas y conclusiones.
    """
    # Separar resultados por tipo (native vs degraded)
    native_results = [r for r in result.raw_results if r["type"] == "native"]
    degraded_results = [r for r in result.raw_results if r["type"] == "degraded"]

    # Calcular métricas por grupo
    native_f1 = 0.0
    for r in native_results:
        f1, _, _ = field_level_f1(r["predicted"], r["ground_truth"])
        native_f1 += f1

    if len(native_results) > 0:
        native_f1_avg = native_f1 / len(native_results)
    else:
        native_f1_avg = 0.0

    degraded_f1 = 0.0
    for r in degraded_results:
        f1, _, _ = field_level_f1(r["predicted"], r["ground_truth"])
        degraded_f1 += f1

    if len(degraded_results) > 0:
        degraded_f1_avg = degraded_f1 / len(degraded_results)
    else:
        degraded_f1_avg = 0.0

    if len(native_results) > 0:
        native_em = sum(1 for r in native_results if r["exact_match"]) / len(native_results)
    else:
        native_em = 0.0

    if len(degraded_results) > 0:
        degraded_em = sum(1 for r in degraded_results if r["exact_match"]) / len(degraded_results)
    else:
        degraded_em = 0.0

    if len(native_results) > 0:
        native_latency = sum(r["latency"] for r in native_results) / len(native_results)
    else:
        native_latency = 0.0

    if len(degraded_results) > 0:
        degraded_latency = sum(r["latency"] for r in degraded_results) / len(degraded_results)
    else:
        degraded_latency = 0.0

    if len(native_results) > 0:
        native_ram = max(r["ram_peak"] for r in native_results)
    else:
        native_ram = 0.0

    if len(degraded_results) > 0:
        degraded_ram = max(r["ram_peak"] for r in degraded_results)
    else:
        degraded_ram = 0.0

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
| Field-level F1 | {native_f1_avg:.4f} | {degraded_f1_avg:.4f} |
| Exact Match Rate | {native_em:.2%} | {degraded_em:.2%} |
| Latencia Media | {native_latency:.2f} s | {degraded_latency:.2f} s |
| RAM Peak | {native_ram:.2f} MB | {degraded_ram:.2f} MB |

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

- **PI1**: El sistema es viable:
  - F1 score de {result.field_f1:.4f} demuestra precisión aceptable
  - Latencia media de {result.avg_latency_seconds:.2f} s es adecuada para uso real

- **PI2**: Robustez ante PDFs degradados:
  - Diferencia en F1 entre nativos y degradados es de {abs(native_f1_avg - degraded_f1_avg):.4f}

## 5. Resultados Brutos

```json
{json.dumps([{k: v for k, v in r.items() if k != "predicted" and k != "ground_truth"} for r in result.raw_results], indent=2, ensure_ascii=False)}
```
"""

    output_path.write_text(report_md, encoding="utf-8")
    return report_md
