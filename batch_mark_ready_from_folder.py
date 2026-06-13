from __future__ import annotations

"""
Marca casos como listos en lote, detectando documento y ground truth dentro de la
estructura experimental preparada.

Uso:
    py batch_mark_ready_from_folder.py
    py batch_mark_ready_from_folder.py --schema credito_hipotecario
    py batch_mark_ready_from_folder.py --variant image_photo --limit 10
    py batch_mark_ready_from_folder.py --dry-run
"""

import argparse
from pathlib import Path
from typing import Any

from mark_experiment_case_ready import ROOT, load_manifest, mark_case_ready, save_manifest


def find_matching_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    resolved = ROOT / path_str
    return resolved if resolved.exists() else None


def should_consider_case(case: dict[str, Any], schema: str | None, variant: str | None) -> bool:
    if case.get("acquisition_status") == "ready" and case.get("ground_truth_status") == "ready":
        return False
    if schema and case.get("schema_name") != schema:
        return False
    if variant and case.get("variant") != variant:
        return False
    return True


def collect_ready_candidates(
    manifest: dict[str, Any],
    *,
    schema: str | None,
    variant: str | None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for case in manifest.get("test_cases", []):
        if not should_consider_case(case, schema, variant):
            continue

        document_path = find_matching_path(case.get("prepared_file_path"))
        ground_truth_path = find_matching_path(case.get("prepared_ground_truth_path"))
        if document_path and ground_truth_path:
            candidates.append(
                {
                    "id": case["id"],
                    "document_path": document_path,
                    "ground_truth_path": ground_truth_path,
                    "schema_name": case.get("schema_name"),
                    "variant": case.get("variant"),
                    "compliance_expected": case.get("compliance_expected"),
                }
            )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Marca en lote los casos listos detectados en la carpeta experimental.")
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "test_data" / "experiment_manifest_120_spain_generated.json"),
        help="Ruta al manifiesto generado.",
    )
    parser.add_argument("--schema", default=None, help="Filtra por schema_name.")
    parser.add_argument("--variant", default=None, help="Filtra por variant.")
    parser.add_argument("--limit", type=int, default=0, help="Maximo de casos a marcar. 0 = sin limite.")
    parser.add_argument("--dry-run", action="store_true", help="Solo muestra lo que se marcaria.")
    parser.add_argument("--keep-meta", action="store_true", help="No elimina los .meta.json al marcar los casos.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (ROOT / manifest_path).resolve()

    manifest = load_manifest(manifest_path)
    candidates = collect_ready_candidates(manifest, schema=args.schema, variant=args.variant)
    if args.limit > 0:
        candidates = candidates[: args.limit]

    if not candidates:
        print("No se encontraron casos listos para marcar con los filtros indicados.")
        return

    print("=" * 80)
    print("CASOS DETECTADOS PARA MARCADO EN LOTE")
    print("=" * 80)
    for item in candidates:
        print(f"{item['id']} | {item['schema_name']} | {item['variant']} | {item['compliance_expected']}")
        print(f"  documento: {item['document_path'].relative_to(ROOT)}")
        print(f"  ground truth: {item['ground_truth_path'].relative_to(ROOT)}")

    if args.dry_run:
        print("=" * 80)
        print(f"Dry run: {len(candidates)} casos detectados.")
        print("=" * 80)
        return

    updated = 0
    for item in candidates:
        mark_case_ready(
            manifest,
            case_id=item["id"],
            document_path=str(item["document_path"]),
            ground_truth_path=str(item["ground_truth_path"]),
            keep_meta=args.keep_meta,
        )
        updated += 1

    save_manifest(manifest_path, manifest)

    print("=" * 80)
    print(f"Marcado en lote completado. Casos actualizados: {updated}")
    print(f"Manifiesto actualizado: {manifest_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
