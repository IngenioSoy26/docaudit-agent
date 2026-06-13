
"""
Script para ejecutar experimentos de evaluación para la tesis.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
from core.orchestrator import run_pipeline
from core.evaluation import run_evaluation, generate_report, EvaluationResult


def load_test_cases(manifest_path: Path) -> list[dict]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    test_cases = []
    for tc in gt_data.get("test_cases", []):
        relative_path = (
            tc.get("prepared_file_path")
            or tc.get("file_path")
            or tc.get("pdf_path")
            or tc.get("image_path")
        )
        input_kind = tc.get("input_kind")

        case = {
            "id": tc.get("id"),
            "name": tc.get("name"),
            "type": tc.get("type", "unknown"),
            "ground_truth": tc.get("ground_truth", {}),
            "schema_name": tc.get("schema_name"),
            "input_kind": input_kind,
        }

        if isinstance(tc.get("text"), str):
            case["text"] = tc["text"]
            test_cases.append(case)
            continue

        if not relative_path:
            print(f"Warning: Caso sin file_path/pdf_path/image_path: {tc.get('id')}, se omite.")
            continue

        file_path = ROOT / relative_path
        if not file_path.exists():
            print(f"Warning: No se encontró archivo {file_path}, saltando caso {tc.get('id')}")
            continue

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            case["pdf_bytes"] = file_path.read_bytes()
            case["input_kind"] = input_kind or ("pdf_scanned" if tc.get("type") == "degraded" else "pdf_native")
        elif suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            case["image_bytes"] = file_path.read_bytes()
            case["input_kind"] = input_kind or "image"
        else:
            print(f"Warning: Tipo de archivo no soportado en {file_path}, se omite.")
            continue

        test_cases.append(case)

    return test_cases


def main():
    parser = argparse.ArgumentParser(description="Ejecuta experimentos de evaluación para la tesis.")
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "test_data" / "ground_truth.json"),
        help="Ruta al manifiesto JSON con los casos de prueba.",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("INICIO DEL EXPERIMENTO - TESIS")
    print("=" * 80)
    
    # Cargar datos de prueba
    ground_truth_path = Path(args.manifest)
    if not ground_truth_path.is_absolute():
        ground_truth_path = (ROOT / ground_truth_path).resolve()
    if not ground_truth_path.exists():
        print(f"Error: No se encontró {ground_truth_path}")
        print("Por favor, crea el archivo con tus casos de prueba.")
        return

    test_cases = load_test_cases(ground_truth_path)

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
