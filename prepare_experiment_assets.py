from __future__ import annotations

"""
Prepara la estructura operativa del experimento de 120 pruebas.

Hace lo siguiente:
- Crea la arborescencia de carpetas bajo `test_data/experimentos_120`.
- Copia los casos nativos ya disponibles del corpus a esa estructura.
- Genera ficheros `.meta.json` para los casos pendientes (borrosos, fotos, manuscritos, no conformes).
- Escribe un manifiesto normalizado apuntando a la estructura experimental preparada.

Uso recomendado:
    py prepare_experiment_assets.py
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_target_file(case: dict[str, Any], base_dir: Path) -> Path:
    source_path = Path(str(case["file_path"]))
    extension = source_path.suffix or (".jpg" if case.get("input_kind") == "image" else ".pdf")
    return (
        base_dir
        / case["schema_name"]
        / case["variant"]
        / case["compliance_expected"]
        / f"{case['id']}{extension}"
    )


def build_target_ground_truth(case: dict[str, Any], base_dir: Path) -> Path:
    return (
        base_dir
        / case["schema_name"]
        / case["variant"]
        / case["compliance_expected"]
        / f"{case['id']}_ground_truth.json"
    )


def build_target_meta(case: dict[str, Any], base_dir: Path) -> Path:
    return (
        base_dir
        / case["schema_name"]
        / case["variant"]
        / case["compliance_expected"]
        / f"{case['id']}.meta.json"
    )


def prepare_case(case: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    normalized_case = dict(case)
    target_file = build_target_file(case, base_dir)
    target_gt = build_target_ground_truth(case, base_dir)
    target_meta = build_target_meta(case, base_dir)

    target_file.parent.mkdir(parents=True, exist_ok=True)

    normalized_case["prepared_file_path"] = str(target_file.relative_to(ROOT)).replace("\\", "/")
    normalized_case["prepared_ground_truth_path"] = str(target_gt.relative_to(ROOT)).replace("\\", "/")
    normalized_case["prepared_meta_path"] = str(target_meta.relative_to(ROOT)).replace("\\", "/")

    ready_source = ROOT / str(case.get("file_path", ""))
    source_gt = ROOT / str(case.get("source_ground_truth_path", ""))

    if case.get("acquisition_status") == "ready" and ready_source.exists():
        shutil.copy2(ready_source, target_file)
        if source_gt.exists():
            shutil.copy2(source_gt, target_gt)
        normalized_case["prepared_status"] = "copied"
        if target_meta.exists():
            target_meta.unlink()
    else:
        meta_payload = {
            "id": case["id"],
            "schema_name": case["schema_name"],
            "variant": case["variant"],
            "variant_description": case.get("variant_description"),
            "compliance_expected": case["compliance_expected"],
            "country": case.get("country", "Espana"),
            "legal_basis": case.get("legal_basis", []),
            "source_pdf_path": case.get("source_pdf_path"),
            "source_ground_truth_path": case.get("source_ground_truth_path"),
            "target_file_path": normalized_case["prepared_file_path"],
            "target_ground_truth_path": normalized_case["prepared_ground_truth_path"],
            "rule_break_plan": case.get("rule_break_plan", []),
            "generation_notes": case.get("generation_notes", []),
            "instructions": [
                "Crear o capturar el documento en el formato indicado.",
                "Guardar el fichero final exactamente en target_file_path.",
                "Si el caso es no conforme, ajustar tambien el ground truth final.",
                "Cuando el caso quede listo, actualizar el manifiesto y marcar acquisition_status como ready.",
            ],
        }
        dump_json(target_meta, meta_payload)
        normalized_case["prepared_status"] = "pending"

    return normalized_case


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total": len(cases),
        "copied": sum(1 for case in cases if case.get("prepared_status") == "copied"),
        "pending": sum(1 for case in cases if case.get("prepared_status") == "pending"),
        "by_use_case": {},
    }

    for case in cases:
        schema_name = case["schema_name"]
        bucket = summary["by_use_case"].setdefault(schema_name, {"copied": 0, "pending": 0})
        if case.get("prepared_status") == "copied":
            bucket["copied"] += 1
        else:
            bucket["pending"] += 1

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara la estructura del experimento de 120 pruebas.")
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "test_data" / "experiment_manifest_120_spain.json"),
        help="Ruta al manifiesto base.",
    )
    parser.add_argument(
        "--output-manifest",
        default=str(ROOT / "test_data" / "experiment_manifest_120_spain_prepared.json"),
        help="Ruta del manifiesto normalizado de salida.",
    )
    parser.add_argument(
        "--base-dir",
        default=str(ROOT / "test_data" / "experimentos_120"),
        help="Directorio base donde se preparan los activos del experimento.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (ROOT / manifest_path).resolve()

    output_manifest_path = Path(args.output_manifest)
    if not output_manifest_path.is_absolute():
        output_manifest_path = (ROOT / output_manifest_path).resolve()

    base_dir = Path(args.base_dir)
    if not base_dir.is_absolute():
        base_dir = (ROOT / base_dir).resolve()

    manifest = load_manifest(manifest_path)
    prepared_cases = [prepare_case(case, base_dir) for case in manifest.get("test_cases", [])]
    prepared_manifest = dict(manifest)
    prepared_manifest["test_cases"] = prepared_cases
    prepared_manifest["prepared_summary"] = summarize_cases(prepared_cases)

    dump_json(output_manifest_path, prepared_manifest)

    print("=" * 80)
    print("Estructura experimental preparada")
    print(f"Base experimental: {base_dir}")
    print(f"Manifiesto preparado: {output_manifest_path}")
    print(f"Casos copiados: {prepared_manifest['prepared_summary']['copied']}")
    print(f"Casos pendientes: {prepared_manifest['prepared_summary']['pending']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
