from __future__ import annotations

"""
Genera variantes visuales para el experimento de 120 pruebas.

Capacidades:
- Rasteriza PDFs si PyMuPDF (`fitz`) esta disponible.
- Acepta tambien imagenes fuente.
- Genera automaticamente variantes conformes:
  - scanned_blurry_pdf
  - image_photo
  - image_handwritten
- Para casos no conformes, crea un borrador visual opcional y deja marcada la necesidad
  de ajuste semantico y de ground truth antes de evaluarlos.

Uso recomendado:
    py generate_experiment_variants.py
    py generate_experiment_variants.py --include-non-compliant-drafts
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parent
RANDOM = random.Random(120)

try:
    import fitz  # type: ignore
except ImportError:
    fitz = None


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_image_from_source(source_path: Path) -> Image.Image:
    suffix = source_path.suffix.lower()
    if suffix == ".pdf":
        if fitz is None:
            raise RuntimeError("missing_pymupdf")
        doc = fitz.open(source_path)
        try:
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            return image
        finally:
            doc.close()

    image = Image.open(source_path)
    return image.convert("RGB")


def add_noise(image: Image.Image, amount: int = 18) -> Image.Image:
    noise = Image.effect_noise(image.size, amount).convert("L")
    noise_rgb = Image.merge("RGB", (noise, noise, noise))
    return ImageChops.add_modulo(image, noise_rgb)


def simulate_scan_blur(image: Image.Image) -> Image.Image:
    img = image.convert("L").convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(0.78)
    img = ImageEnhance.Brightness(img).enhance(1.03)
    img = img.filter(ImageFilter.GaussianBlur(radius=1.4))
    img = add_noise(img, amount=10)
    return img


def simulate_photo(image: Image.Image) -> Image.Image:
    img = ImageOps.expand(image, border=30, fill=(245, 245, 240))
    img = img.rotate(-1.2, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(245, 245, 240))
    img = ImageEnhance.Contrast(img).enhance(0.92)
    img = ImageEnhance.Color(img).enhance(0.95)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
    shadow = Image.new("RGB", img.size, (12, 12, 12))
    mask = Image.linear_gradient("L").resize(img.size)
    shadow = Image.composite(img, shadow, mask)
    return Image.blend(img, shadow, 0.12)


def simulate_handwritten(image: Image.Image, label: str) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size
    ink = (30, 60, 150)

    for index in range(3):
        y = int(height * (0.12 + 0.12 * index))
        x1 = int(width * 0.62)
        x2 = min(width - 40, x1 + 180)
        draw.line((x1, y, x2, y + RANDOM.randint(-12, 12)), fill=ink, width=3)

    draw.line((width * 0.12, height * 0.88, width * 0.38, height * 0.91), fill=(150, 30, 30), width=4)
    draw.text((int(width * 0.58), int(height * 0.08)), f"Nota: {label}", fill=ink)
    return img


def save_variant(image: Image.Image, target_path: Path, variant: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.suffix.lower() == ".pdf":
        image.convert("RGB").save(target_path, "PDF", resolution=150.0)
        return

    quality = 88 if variant == "image_photo" else 92
    image.convert("RGB").save(target_path, quality=quality)


def build_variant_image(source_image: Image.Image, variant: str, label: str) -> Image.Image:
    if variant == "scanned_blurry_pdf":
        return simulate_scan_blur(source_image)
    if variant == "image_photo":
        return simulate_photo(source_image)
    if variant == "image_handwritten":
        return simulate_handwritten(source_image, label)
    return source_image.copy()


def copy_ground_truth_if_missing(case: dict[str, Any]) -> None:
    target_gt = ROOT / case["prepared_ground_truth_path"]
    if target_gt.exists():
        return

    if case.get("ground_truth"):
        target_gt.parent.mkdir(parents=True, exist_ok=True)
        target_gt.write_text(json.dumps(case["ground_truth"], ensure_ascii=False, indent=2), encoding="utf-8")


def process_case(case: dict[str, Any], include_non_compliant_drafts: bool) -> dict[str, Any]:
    prepared_status = case.get("prepared_status")
    if prepared_status == "copied":
        return case

    compliance = case.get("compliance_expected")
    if compliance == "no_conforme" and not include_non_compliant_drafts:
        case["variant_generation_status"] = "pending_semantic_edit"
        return case

    source_path = ROOT / str(case.get("source_pdf_path", ""))
    target_path = ROOT / str(case.get("prepared_file_path", ""))
    label = case["id"]

    try:
        source_image = load_image_from_source(source_path)
    except Exception as exc:
        if str(exc) == "missing_pymupdf":
            case["variant_generation_status"] = "pending_pdf_rasterizer"
            case["variant_generation_error"] = "Falta PyMuPDF para rasterizar PDF a imagen. Instala: py -m pip install pymupdf"
        else:
            case["variant_generation_status"] = "failed"
            case["variant_generation_error"] = str(exc)
        return case

    variant = str(case.get("variant"))
    generated = build_variant_image(source_image, variant, label)
    save_variant(generated, target_path, variant)

    if compliance == "conforme":
        copy_ground_truth_if_missing(case)
        case["ground_truth_status"] = "ready"
        case["acquisition_status"] = "ready"
        case["prepared_status"] = "generated"
        case["variant_generation_status"] = "generated"
    else:
        case["prepared_status"] = "draft_generated"
        case["variant_generation_status"] = "draft_generated_requires_semantic_edit"
        case["ground_truth_status"] = "pending_adjustment"
        case["acquisition_status"] = "pending_semantic_edit"

    return case


def summarize(cases: list[dict[str, Any]]) -> dict[str, int]:
    summary = {
        "generated": 0,
        "draft_generated_requires_semantic_edit": 0,
        "pending_semantic_edit": 0,
        "pending_pdf_rasterizer": 0,
        "failed": 0,
    }
    for case in cases:
        status = case.get("variant_generation_status")
        if status in summary:
            summary[status] += 1
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera variantes visuales del experimento de 120 pruebas.")
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "test_data" / "experiment_manifest_120_spain_prepared.json"),
        help="Manifiesto preparado de entrada.",
    )
    parser.add_argument(
        "--output-manifest",
        default=str(ROOT / "test_data" / "experiment_manifest_120_spain_generated.json"),
        help="Manifiesto de salida con estado de variantes.",
    )
    parser.add_argument(
        "--include-non-compliant-drafts",
        action="store_true",
        help="Genera tambien borradores visuales para los casos no conformes.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (ROOT / manifest_path).resolve()

    output_manifest_path = Path(args.output_manifest)
    if not output_manifest_path.is_absolute():
        output_manifest_path = (ROOT / output_manifest_path).resolve()

    manifest = load_manifest(manifest_path)
    updated_cases = [
        process_case(case, include_non_compliant_drafts=args.include_non_compliant_drafts)
        for case in manifest.get("test_cases", [])
    ]
    manifest["test_cases"] = updated_cases
    manifest["variant_generation_summary"] = summarize(updated_cases)

    save_manifest(output_manifest_path, manifest)

    print("=" * 80)
    print("Generacion de variantes completada")
    print(f"Salida: {output_manifest_path}")
    for key, value in manifest["variant_generation_summary"].items():
        print(f"{key}: {value}")
    print("=" * 80)


if __name__ == "__main__":
    main()
