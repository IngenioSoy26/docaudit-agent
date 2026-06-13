from __future__ import annotations

"""
Lista y prioriza los casos no conformes pendientes del experimento.

Permite:
- Filtrar por caso de uso y variante.
- Mostrar un resumen por bloques.
- Ver una checklist operativa con rutas, base normativa y plan de ruptura de reglas.
- Exportar el resultado a JSON si se necesita compartir o seguir trabajando despues.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def filter_cases(
    manifest: dict[str, Any],
    *,
    schema_name: str | None,
    variant: str | None,
) -> list[dict[str, Any]]:
    cases = []
    for case in manifest.get("test_cases", []):
        if case.get("compliance_expected") != "no_conforme":
            continue
        if case.get("prepared_status") not in {"pending", "draft_generated"}:
            continue
        if schema_name and case.get("schema_name") != schema_name:
            continue
        if variant and case.get("variant") != variant:
            continue
        cases.append(case)
    return cases


def priority_for_variant(variant_name: str) -> int:
    order = {
        "native_pdf": 1,
        "scanned_blurry_pdf": 2,
        "image_photo": 3,
        "image_handwritten": 4,
    }
    return order.get(variant_name, 99)


def sort_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        cases,
        key=lambda case: (
            str(case.get("schema_name", "")),
            priority_for_variant(str(case.get("variant", ""))),
            str(case.get("id", "")),
        ),
    )


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for case in cases:
        grouped[str(case["schema_name"])][str(case["variant"])] += 1

    return {
        "total_pending_non_compliant": len(cases),
        "by_use_case": {schema: dict(variants) for schema, variants in grouped.items()},
    }


def build_checklist(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checklist: list[dict[str, Any]] = []
    for case in sort_cases(cases):
        checklist.append(
            {
                "id": case.get("id"),
                "schema_name": case.get("schema_name"),
                "variant": case.get("variant"),
                "country": case.get("country"),
                "legal_basis": case.get("legal_basis", []),
                "source_pdf_path": case.get("source_pdf_path"),
                "target_file_path": case.get("prepared_file_path") or case.get("file_path"),
                "target_ground_truth_path": case.get("prepared_ground_truth_path"),
                "prepared_status": case.get("prepared_status"),
                "ground_truth_status": case.get("ground_truth_status"),
                "rule_break_plan": case.get("rule_break_plan", []),
                "generation_notes": case.get("generation_notes", []),
                "next_steps": [
                    "Crear o editar el documento final para que incumpla alguna regla relevante.",
                    "Guardar el documento exactamente en target_file_path.",
                    "Ajustar el ground truth final para que represente fielmente el documento editado.",
                    "Marcar acquisition_status y ground_truth_status como ready en el manifiesto.",
                ],
            }
        )
    return checklist


def print_summary(summary: dict[str, Any]) -> None:
    print("=" * 80)
    print("COLA DE CASOS NO CONFORMES PENDIENTES")
    print("=" * 80)
    print(f"Total pendientes: {summary['total_pending_non_compliant']}")
    print("")
    for schema_name, variants in summary["by_use_case"].items():
        print(f"[{schema_name}]")
        for variant, count in variants.items():
            print(f"  - {variant}: {count}")
        print("")


def print_cases(cases: list[dict[str, Any]], limit: int) -> None:
    if not cases:
        print("No hay casos no conformes pendientes con el filtro indicado.")
        return

    selected = sort_cases(cases)[:limit] if limit > 0 else sort_cases(cases)
    for case in selected:
        print("-" * 80)
        print(f"ID: {case.get('id')}")
        print(f"Caso de uso: {case.get('schema_name')}")
        print(f"Variante: {case.get('variant')}")
        print(f"Estado: {case.get('prepared_status')} / GT: {case.get('ground_truth_status')}")
        print(f"Fuente: {case.get('source_pdf_path')}")
        print(f"Destino documento: {case.get('prepared_file_path') or case.get('file_path')}")
        print(f"Destino ground truth: {case.get('prepared_ground_truth_path')}")
        print("Base normativa:")
        for item in case.get("legal_basis", []):
            print(f"  * {item}")
        print("Que romper:")
        for item in case.get("rule_break_plan", []):
            print(f"  * {item}")
        print("Notas:")
        for item in case.get("generation_notes", []):
            print(f"  * {item}")
    print("-" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Revisa la cola de casos no conformes pendientes.")
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "test_data" / "experiment_manifest_120_spain_generated.json"),
        help="Ruta al manifiesto generado.",
    )
    parser.add_argument("--schema", default=None, help="Filtra por schema_name.")
    parser.add_argument("--variant", default=None, help="Filtra por variant.")
    parser.add_argument(
        "--limit",
        type=int,
        default=12,
        help="Numero maximo de casos a mostrar. Usa 0 para mostrar todos.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Si se indica, exporta la checklist filtrada a un JSON.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (ROOT / manifest_path).resolve()

    manifest = load_manifest(manifest_path)
    cases = filter_cases(manifest, schema_name=args.schema, variant=args.variant)
    summary = summarize(cases)
    checklist = build_checklist(cases)

    print_summary(summary)
    print_cases(cases, limit=args.limit)

    if args.output_json:
        output_path = Path(args.output_json)
        if not output_path.is_absolute():
            output_path = (ROOT / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"summary": summary, "checklist": checklist}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Checklist exportada en: {output_path}")


if __name__ == "__main__":
    main()
