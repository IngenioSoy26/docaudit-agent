
"""
Script para ejecutar experimentos de evaluación para la tesis.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
from core.orchestrator import run_pipeline
from core.evaluation import run_evaluation, generate_report, EvaluationResult


def main():
    print("=" * 80)
    print("INICIO DEL EXPERIMENTO - TESIS")
    print("=" * 80)
    
    # Cargar datos de prueba
    ground_truth_path = ROOT / "test_data" / "ground_truth.json"
    if not ground_truth_path.exists():
        print(f"Error: No se encontró {ground_truth_path}")
        print("Por favor, crea el archivo con tus casos de prueba.")
        return

    with open(ground_truth_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    test_cases = []
    
    for tc in gt_data["test_cases"]:
        pdf_path = ROOT / "test_data" / tc["pdf_path"]
        if not pdf_path.exists():
            print(f"Warning: No se encontró PDF {pdf_path}, saltando caso {tc['id']}")
            continue
        test_cases.append({
            "id": tc["id"],
            "name": tc["name"],
            "type": tc["type"],
            "pdf_bytes": pdf_path.read_bytes(),
            "ground_truth": tc["ground_truth"],
        })

    print(f"\nCargados {len(test_cases)} casos de prueba.")

    # Ejecutar evaluación
    print("\nEjecutando evaluación...")
    result: EvaluationResult = run_evaluation(run_pipeline, test_cases)

    # Generar reporte
    report_path = ROOT / "reporte_evaluacion.md"
    print(f"Generando reporte en {report_path}...")
    generate_report(result, report_path)

    print("\n" + "=" * 80)
    print("FIN DEL EXPERIMENTO")
    print("=" * 80)
    print("\nResumen rápido:")
    print(f"  Field-level F1: {result.field_f1:.4f}")
    print(f"  Exact Match Rate: {result.exact_match_rate:.2%}")
    print(f"  Latencia Media: {result.avg_latency_seconds:.2f} s")
    print(f"  RAM Peak: {result.ram_peak_mb:.2f} MB")
    print(f"\nReporte completo: {report_path}")


if __name__ == "__main__":
    main()
