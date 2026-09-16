#!/usr/bin/env python3
"""
Script de evaluación completa de DocAudit Agent.

Ejecuta:
1. Procesamiento de todos los 120 documentos del corpus
2. Guarda logs en test_data/execution_logs
3. Genera informes PDF (Informe_Tesis, Informe_Evaluacion)
4. Genera gráficos y métricas

Requisitos:
- Entorno virtual activado
- Ollama en ejecución con el modelo correspondiente
"""

import json
import sys
import time
import psutil
from pathlib import Path
from datetime import datetime

# Añadir el directorio del proyecto al PATH
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.orchestrator import run_pipeline
from core.document_loader import (
    extract_text_from_pdf_bytes,
    extract_text_from_image_bytes,
    extraction_policy_for_doc,
)
from core.settings import settings


def get_file_type(file_path: Path) -> str:
    """Inferir el tipo de documento a partir del nombre y contenido."""
    file_name_lower = file_path.name.lower()
    if "native_pdf" in file_name_lower:
        return "native_pdf"
    elif "scanned_blurry_pdf" in file_name_lower:
        return "scanned_pdf"
    elif "image_photo" in file_name_lower:
        return "image"
    elif "image_handwritten" in file_name_lower:
        return "image"
    elif file_path.suffix.lower() == ".pdf":
        return "native_pdf"
    else:
        return "image"


def get_schema_name_from_path(file_path: Path) -> str | None:
    """Inferir el schema a partir de la ruta del documento."""
    parts = file_path.parts
    for part in parts:
        if "caso_uso_1_auditoria_hipotecaria" in part:
            return "credito_hipotecario"
        elif "caso_uso_2_auditoria_fiscal" in part:
            return "auditoria_fiscal"
        elif "caso_uso_3_kyc_onboarding" in part:
            return "kyc_onboarding"
    return None


def measure_ram_usage():
    """Medir el uso de RAM actual del proceso Python."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)  # MB


def process_single_document(
    file_path: Path,
    schema_name: str,
) -> tuple[dict, float, float, str]:
    """
    Procesar un solo documento y devolver resultados, latencia y RAM.
    """
    print(f"  Procesando: {file_path.name}")
    start_time = time.perf_counter()
    start_ram = measure_ram_usage()

    file_bytes = file_path.read_bytes()
    file_type = get_file_type(file_path)

    # Extraer texto
    extracted = None
    text_method = ""
    with extraction_policy_for_doc(file_path.name):
        if file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            extracted = extract_text_from_image_bytes(file_bytes)
        else:
            extracted = extract_text_from_pdf_bytes(file_bytes)
    text_method = str(extracted.get("method") or "")

    text = extracted.get("text", "") or ""
    pages = extracted.get("pages")

    # Ejecutar pipeline completo
    try:
        result = run_pipeline(
            text=text,
            schemas_dir=PROJECT_ROOT / "schemas",
            pages=pages,
            schema_name=schema_name,
        )
    except Exception as e:
        print(f"    Error al procesar: {e}")
        result = {}

    end_ram = measure_ram_usage()
    end_time = time.perf_counter()

    latency = end_time - start_time
    ram_used = max(0, end_ram - start_ram)

    print(f"    Hecho! Latencia: {latency:.2f}s, RAM: {ram_used:.2f}MB")
    return result, latency, ram_used, text_method


def save_execution_log(
    file_name: str,
    file_type: str,
    method: str,
    processing_time: float,
    ram_used_mb: float,
    extracted_fields: dict,
    full_result: dict,
):
    """Guardar log de ejecución en test_data/execution_logs."""
    logs_dir = PROJECT_ROOT / "test_data" / "execution_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "file_name": file_name,
        "file_type": file_type,
        "method": method,
        "processing_time_seconds": round(processing_time, 4),
        "ram_used_mb": round(ram_used_mb, 2),
        "extracted_fields": extracted_fields,
        "full_result": full_result,
    }
    log_path = logs_dir / f"log_{timestamp}_{file_name.replace('.', '_')}.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)


def get_all_documents():
    """Obtener la lista de documentos válidos para evaluación."""
    data_dir = PROJECT_ROOT / "data" / "sample_docs"
    documents = []

    case_dirs = [
        data_dir / "caso_uso_1_auditoria_hipotecaria",
        data_dir / "caso_uso_2_auditoria_fiscal",
        data_dir / "caso_uso_3_kyc_onboarding",
    ]

    valid_folders = ["native_pdf", "scanned_blurry_pdf", "image_photo", "image_handwritten"]

    for case_dir in case_dirs:
        if not case_dir.exists():
            continue
        schema_name = get_schema_name_from_path(case_dir)
        if not schema_name:
            continue

        for valid_folder in valid_folders:
            folder_path = case_dir / valid_folder
            if not folder_path.exists():
                continue

            # Check both conform and non_conform subfolders
            for subfolder in ["conforme", "no_conforme"]:
                subfolder_path = folder_path / subfolder
                if not subfolder_path.exists():
                    continue

                for file in subfolder_path.iterdir():
                    if file.is_file() and file.suffix.lower() in [
                        ".pdf",
                        ".png",
                        ".jpg",
                        ".jpeg",
                    ]:
                        # Skip ground truth files
                        if file.name.endswith("_ground_truth.json"):
                            continue
                        documents.append((file, schema_name))

    return documents


def main():
    print("=" * 80)
    print("DocAudit Agent - Evaluación Completa")
    print("=" * 80)

    # Obtener lista de documentos
    print("\nObteniendo lista de documentos...")
    documents = get_all_documents()
    print(f"  {len(documents)} documentos encontrados")

    # Procesar cada documento
    print("\nComenzando procesamiento...")
    for i, (doc_path, schema_name) in enumerate(documents, 1):
        print(f"\n[{i}/{len(documents)}]")
        result, latency, ram_used, method = process_single_document(
            doc_path, schema_name
        )

        extracted_fields = result.get("extracted", {})
        save_execution_log(
            file_name=doc_path.name,
            file_type=get_file_type(doc_path),
            method=method,
            processing_time=latency,
            ram_used_mb=ram_used,
            extracted_fields=extracted_fields,
            full_result=result,
        )

    # Generar informes
    print("\n" + "=" * 80)
    print("Generando informes...")
    print("=" * 80)

    import subprocess

    # Generar informe de evaluación (con tools/evaluate.py)
    try:
        print("\nEjecutando tools/evaluate.py...")
        output_csv = PROJECT_ROOT / "reports" / f"evaluation_results_llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        subprocess.run([sys.executable, "-m", "tools.evaluate", "--backend", "llm", "--out", str(output_csv)], check=True, cwd=str(PROJECT_ROOT))
    except Exception as e:
        print(f"Error al ejecutar evaluación: {e}")

    # Generar informe de tesis una vez terminada la evaluación, para usar los
    # resultados recién creados y no un summary previo.
    try:
        print("\nEjecutando generate_thesis_report.py...")
        subprocess.run([sys.executable, str(PROJECT_ROOT / "generate_thesis_report.py")], check=True, cwd=str(PROJECT_ROOT))
    except Exception as e:
        print(f"Error al generar informe de tesis: {e}")

    print("\n" + "=" * 80)
    print("Evaluación completa!")
    print("=" * 80)


if __name__ == "__main__":
    main()
