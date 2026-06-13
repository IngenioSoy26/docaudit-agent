from __future__ import annotations

"""
Marca un caso experimental como listo dentro del manifiesto.

Valida que existan:
- el documento final
- el ground truth final

Y actualiza:
- acquisition_status
- ground_truth_status
- prepared_status
- variant_generation_status (si aplica)

Uso:
    py mark_experiment_case_ready.py --case-id credito_hipotecario_native_pdf_no_conforme_006
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_prepared(cases: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total": len(cases),
        "copied": 0,
        "pending": 0,
        "by_use_case": {},
    }
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: {"copied": 0, "pending": 0})

    for case in cases:
        ready_like = case.get("prepared_status") in {
            "copied",
            "generated",
            "ready_manual",
            "draft_generated_ready",
        }
        schema_name = str(case.get("schema_name", "unknown"))
        if ready_like:
            summary["copied"] += 1
            grouped[schema_name]["copied"] += 1
        else:
            summary["pending"] += 1
            grouped[schema_name]["pending"] += 1

    summary["by_use_case"] = dict(grouped)
    return summary


def summarize_variants(cases: list[dict[str, Any]]) -> dict[str, int]:
    summary = {
        "generated": 0,
        "draft_generated_requires_semantic_edit": 0,
        "pending_semantic_edit": 0,
        "pending_pdf_rasterizer": 0,
        "failed": 0,
        "ready_manual": 0,
    }
    for case in cases:
        status = case.get("variant_generation_status")
        if status in summary:
            summary[status] += 1
    return summary


def resolve_case_paths(
    target_case: dict[str, Any],
    *,
    document_path: str | None = None,
    ground_truth_path: str | None = None,
) -> tuple[Path, Path]:
    resolved_document = Path(document_path) if document_path else ROOT / str(target_case.get("prepared_file_path"))
    resolved_ground_truth = (
        Path(ground_truth_path) if ground_truth_path else ROOT / str(target_case.get("prepared_ground_truth_path"))
    )

    if not resolved_document.is_absolute():
        resolved_document = (ROOT / resolved_document).resolve()
    if not resolved_ground_truth.is_absolute():
        resolved_ground_truth = (ROOT / resolved_ground_truth).resolve()

    return resolved_document, resolved_ground_truth


def mark_case_ready(
    manifest: dict[str, Any],
    *,
    case_id: str,
    document_path: str | None = None,
    ground_truth_path: str | None = None,
    keep_meta: bool = False,
) -> dict[str, Any]:
    cases = manifest.get("test_cases", [])

    target_case = None
    for case in cases:
        if case.get("id") == case_id:
            target_case = case
            break

    if target_case is None:
        raise ValueError(f"No se encontro el case_id: {case_id}")

    resolved_document, resolved_ground_truth = resolve_case_paths(
        target_case,
        document_path=document_path,
        ground_truth_path=ground_truth_path,
    )

    if not resolved_document.exists():
        raise FileNotFoundError(f"No existe el documento final: {resolved_document}")
    if not resolved_ground_truth.exists():
        raise FileNotFoundError(f"No existe el ground truth final: {resolved_ground_truth}")

    target_case["prepared_file_path"] = str(resolved_document.relative_to(ROOT)).replace("\\", "/")
    target_case["prepared_ground_truth_path"] = str(resolved_ground_truth.relative_to(ROOT)).replace("\\", "/")
    target_case["acquisition_status"] = "ready"
    target_case["ground_truth_status"] = "ready"

    if target_case.get("compliance_expected") == "no_conforme":
        target_case["prepared_status"] = "ready_manual"
        target_case["variant_generation_status"] = "ready_manual"
    else:
        target_case["prepared_status"] = target_case.get("prepared_status") or "ready_manual"
        if not target_case.get("variant_generation_status"):
            target_case["variant_generation_status"] = "ready_manual"

    meta_path_str = target_case.get("prepared_meta_path")
    if meta_path_str and not keep_meta:
        meta_path = ROOT / str(meta_path_str)
        if meta_path.exists():
            meta_path.unlink()

    manifest["prepared_summary"] = summarize_prepared(cases)
    manifest["variant_generation_summary"] = summarize_variants(cases)
    return target_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Marca un caso experimental como listo.")
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "test_data" / "experiment_manifest_120_spain_generated.json"),
        help="Ruta al manifiesto generado.",
    )
    parser.add_argument("--case-id", required=True, help="Identificador exacto del caso a actualizar.")
    parser.add_argument(
        "--document-path",
        default=None,
        help="Ruta alternativa del documento final. Si no se indica, usa prepared_file_path.",
    )
    parser.add_argument(
        "--ground-truth-path",
        default=None,
        help="Ruta alternativa del ground truth final. Si no se indica, usa prepared_ground_truth_path.",
    )
    parser.add_argument(
        "--keep-meta",
        action="store_true",
        help="Si se indica, no elimina el .meta.json asociado.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (ROOT / manifest_path).resolve()

    manifest = load_manifest(manifest_path)
    try:
        target_case = mark_case_ready(
            manifest,
            case_id=args.case_id,
            document_path=args.document_path,
            ground_truth_path=args.ground_truth_path,
            keep_meta=args.keep_meta,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc))
    save_manifest(manifest_path, manifest)

    print("=" * 80)
    print("Caso marcado como listo")
    print(f"Case ID: {args.case_id}")
    print(f"Documento: {target_case['prepared_file_path']}")
    print(f"Ground truth: {target_case['prepared_ground_truth_path']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
