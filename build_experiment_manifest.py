from __future__ import annotations

"""
Genera un manifiesto base de 120 pruebas para la tesis.

Diseno:
- 3 casos de uso (hipotecario, fiscal, KYC)
- 4 familias de formato por caso:
  - native_pdf
  - scanned_blurry_pdf
  - image_photo
  - image_handwritten
- 10 pruebas por familia y caso:
  - 5 conformes (cumplen reglas / parametros)
  - 5 no conformes (deben disparar reglas o incidencias)

Total: 3 * 4 * 10 = 120 pruebas.

El script no fabrica todos los documentos; crea un manifiesto reproducible con:
- los nativos disponibles del corpus
- rutas esperadas para variantes futuras
- base legal de Espana por caso de uso
- plan de generacion para documentos no conformes y degradados
"""

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


FORMAT_VARIANTS = [
    {
        "variant": "native_pdf",
        "input_kind": "pdf_native",
        "type": "native",
        "description": "PDF nativo con texto embebido",
        "extension": ".pdf",
    },
    {
        "variant": "scanned_blurry_pdf",
        "input_kind": "pdf_scanned",
        "type": "degraded",
        "description": "PDF escaneado, borroso, con ruido o compresion",
        "extension": ".pdf",
    },
    {
        "variant": "image_photo",
        "input_kind": "image",
        "type": "image",
        "description": "Foto o captura de documento en imagen",
        "extension": ".jpg",
    },
    {
        "variant": "image_handwritten",
        "input_kind": "image",
        "type": "image",
        "description": "Documento con notas manuscritas o contenido a mano alzada",
        "extension": ".jpg",
    },
]


USE_CASES: dict[str, dict[str, Any]] = {
    "credito_hipotecario": {
        "label": "Hipotecario",
        "source_dir": ROOT / "data" / "sample_docs" / "caso_uso_1_auditoria_hipotecaria",
        "pdf_pattern": "contrato_hipoteca_esp_*.pdf",
        "ground_truth_from_pdf": lambda pdf_path: pdf_path.parent / f"ground_truth_esp_{pdf_path.stem.split('_')[-1]}.json",
        "legal_basis": [
            "Espana - RGPD art. 5",
            "Espana - RGPD art. 25",
            "Espana - RGPD art. 22",
            "Espana - Ley 5/2019 LCCI",
            "Espana - Circular 4/2017 Banco de Espana",
        ],
        "non_compliant_plan": [
            "Eliminar identificador documental o FEIN/FIAE",
            "Introducir TAE incoherente o plazo imposible",
            "Introducir ratio de endeudamiento superior al umbral",
        ],
    },
    "auditoria_fiscal": {
        "label": "Auditoria fiscal",
        "source_dir": ROOT / "data" / "sample_docs" / "caso_uso_2_auditoria_fiscal",
        "pdf_pattern": "factura_fiscal_*.pdf",
        "ground_truth_from_pdf": lambda pdf_path: pdf_path.with_suffix(".json"),
        "legal_basis": [
            "Espana - RGPD art. 5",
            "Espana - RGPD art. 25",
            "Espana - Real Decreto 1619/2012",
            "Espana - Ley 37/1992 del IVA",
        ],
        "non_compliant_plan": [
            "Modificar tipo de IVA a un valor no permitido",
            "Descuadrar cuota IVA o importe total",
            "Eliminar numero de factura o fecha obligatoria",
        ],
    },
    "kyc_onboarding": {
        "label": "KYC / onboarding",
        "source_dir": ROOT / "data" / "sample_docs" / "caso_uso_3_kyc_onboarding",
        "pdf_pattern": "expediente_kyc_*.pdf",
        "ground_truth_from_pdf": lambda pdf_path: pdf_path.with_suffix(".json"),
        "legal_basis": [
            "Espana - RGPD art. 5",
            "Espana - RGPD art. 25",
            "Espana - RGPD art. 22",
            "Espana - Ley 10/2010 PBC/FT",
            "Espana - RD 304/2014",
        ],
        "non_compliant_plan": [
            "Documento caducado o numero no legible",
            "Marcar coincidencia positiva en sanciones",
            "Asignar PEP con nivel de riesgo bajo o sin titular real",
        ],
    },
}


def load_ground_truth(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_pending_output_path(schema_name: str, variant: str, source_stem: str, compliance: str, extension: str) -> str:
    target = (
        Path("test_data")
        / "experimentos_120"
        / schema_name
        / variant
        / compliance
        / f"{source_stem}_{variant}_{compliance}{extension}"
    )
    return str(target).replace("\\", "/")


def build_case_entry(
    *,
    schema_name: str,
    case_index: int,
    variant: dict[str, str],
    compliance: str,
    source_pdf: Path,
    source_ground_truth: Path,
    ground_truth: dict[str, Any],
    ready: bool,
) -> dict[str, Any]:
    use_case = USE_CASES[schema_name]
    case_id = f"{schema_name}_{variant['variant']}_{compliance}_{case_index:03d}"
    file_path = (
        str(source_pdf.relative_to(ROOT)).replace("\\", "/")
        if ready
        else build_pending_output_path(schema_name, variant["variant"], source_pdf.stem, compliance, variant["extension"])
    )

    entry: dict[str, Any] = {
        "id": case_id,
        "name": f"{use_case['label']} - {variant['variant']} - {compliance} - {case_index:03d}",
        "country": "Espana",
        "schema_name": schema_name,
        "type": variant["type"],
        "input_kind": variant["input_kind"],
        "variant": variant["variant"],
        "variant_description": variant["description"],
        "compliance_expected": compliance,
        "legal_basis": use_case["legal_basis"],
        "file_path": file_path,
        "source_pdf_path": str(source_pdf.relative_to(ROOT)).replace("\\", "/"),
        "source_ground_truth_path": str(source_ground_truth.relative_to(ROOT)).replace("\\", "/"),
        "ground_truth": ground_truth if compliance == "conforme" else {},
        "ground_truth_status": "ready" if compliance == "conforme" and ready else "pending_adjustment",
        "acquisition_status": "ready" if ready else "pending_generation",
        "rule_break_plan": [] if compliance == "conforme" else use_case["non_compliant_plan"],
        "generation_notes": [],
    }

    if not ready:
        if variant["variant"] == "scanned_blurry_pdf":
            entry["generation_notes"] = [
                "Imprimir o rasterizar el documento y volver a escanearlo.",
                "Aplicar desenfoque, compresion JPEG, ruido o bajo contraste.",
                "Mantener correspondencia con el ground truth si el documento sigue siendo conforme.",
            ]
        elif variant["variant"] == "image_photo":
            entry["generation_notes"] = [
                "Capturar foto movil con perspectiva realista.",
                "Incluir variaciones de iluminacion y fondo moderado.",
            ]
        elif variant["variant"] == "image_handwritten":
            entry["generation_notes"] = [
                "Anadir notas a mano alzada, marcas o texto manuscrito sobre el documento.",
                "Puede ser documento fotografiado o escaneado como imagen.",
            ]
        elif variant["variant"] == "native_pdf":
            entry["generation_notes"] = [
                "Crear una version digital que incumpla reglas o parametros conservando estructura nativa.",
            ]

    return entry


def build_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "country": "Espana",
        "design": {
            "total_tests": 120,
            "use_cases": 3,
            "variants_per_use_case": 4,
            "tests_per_variant": 10,
            "compliance_split": {"conforme": 5, "no_conforme": 5},
        },
        "test_cases": [],
    }

    for schema_name, cfg in USE_CASES.items():
        pdfs = sorted(cfg["source_dir"].glob(cfg["pdf_pattern"]))
        if len(pdfs) < 10:
            raise RuntimeError(f"No hay suficientes PDFs base en {cfg['source_dir']} para construir la matriz de 40 casos.")

        compliant_sources = pdfs[:5]
        non_compliant_sources = pdfs[5:10]

        for variant in FORMAT_VARIANTS:
            for idx, pdf_path in enumerate(compliant_sources, start=1):
                gt_path = cfg["ground_truth_from_pdf"](pdf_path)
                gt = load_ground_truth(gt_path)
                ready = variant["variant"] == "native_pdf"
                manifest["test_cases"].append(
                    build_case_entry(
                        schema_name=schema_name,
                        case_index=idx,
                        variant=variant,
                        compliance="conforme",
                        source_pdf=pdf_path,
                        source_ground_truth=gt_path,
                        ground_truth=gt,
                        ready=ready,
                    )
                )

            for idx, pdf_path in enumerate(non_compliant_sources, start=6):
                gt_path = cfg["ground_truth_from_pdf"](pdf_path)
                gt = load_ground_truth(gt_path)
                manifest["test_cases"].append(
                    build_case_entry(
                        schema_name=schema_name,
                        case_index=idx,
                        variant=variant,
                        compliance="no_conforme",
                        source_pdf=pdf_path,
                        source_ground_truth=gt_path,
                        ground_truth=gt,
                        ready=False,
                    )
                )

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera el manifiesto base de 120 pruebas para la tesis.")
    parser.add_argument(
        "--output",
        default=str(ROOT / "test_data" / "experiment_manifest_120_spain.json"),
        help="Ruta del JSON de salida.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (ROOT / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest()
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("Manifiesto base generado correctamente")
    print(f"Salida: {output_path}")
    print(f"Casos totales: {len(manifest['test_cases'])}")
    print("=" * 80)


if __name__ == "__main__":
    main()
