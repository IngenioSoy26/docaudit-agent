from __future__ import annotations

"""
Genera un corpus experimental aislado de alta validez para la tesis.

Objetivos:
- No tocar el sistema productivo ni los documentos base.
- Crear 120 documentos reales organizados por caso de uso, variante y conformidad.
- Mantener coherencia normativa entre "conforme" y "no_conforme".
- Dejar un manifiesto nuevo listo para usar con run_experiment.py.

Salida principal:
- test_data/experimentos_120_high_validity/
- test_data/experiment_manifest_120_spain_high_validity.json
"""

import json
import math
import random
import shutil
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import fitz  # type: ignore
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "test_data" / "experimentos_120_high_validity"
MANIFEST_PATH = ROOT / "test_data" / "experiment_manifest_120_spain_high_validity.json"
RNG = random.Random(12042)
TODAY = date(2026, 6, 12)


LEGAL_BASIS = {
    "credito_hipotecario": [
        "Espana - RGPD art. 5",
        "Espana - RGPD art. 25",
        "Espana - RGPD art. 22",
        "Espana - Ley 5/2019 LCCI",
        "Espana - Circular 4/2017 Banco de Espana",
    ],
    "auditoria_fiscal": [
        "Espana - RGPD art. 5",
        "Espana - RGPD art. 25",
        "Espana - Real Decreto 1619/2012",
        "Espana - Ley 37/1992 del IVA",
    ],
    "kyc_onboarding": [
        "Espana - RGPD art. 5",
        "Espana - RGPD art. 25",
        "Espana - RGPD art. 22",
        "Espana - Ley 10/2010 PBC/FT",
        "Espana - RD 304/2014",
    ],
}


@dataclass
class SemanticCase:
    schema_name: str
    semantic_index: int
    compliance_expected: str
    source_pdf_path: Path
    source_ground_truth_path: Path
    ground_truth: dict[str, Any]
    expected_rule_failures: list[str]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def money(value: float) -> str:
    return f"{value:,.2f} EUR".replace(",", "_").replace(".", ",").replace("_", ".")


def bool_text(value: bool) -> str:
    return "Si" if value else "No"


def derive_hypotecario(record: dict[str, Any], idx: int, compliant: bool) -> tuple[dict[str, Any], list[str]]:
    amount = float(record["monto_prestamo_eur"])
    rate = float(record["tasa_interes"])
    issue_date = record["fecha_emision"]
    if compliant:
        tae = round(rate + 0.75 + (idx * 0.09), 2)
        plazo = 240 + idx * 12
        ingresos = round(max(amount / 70, 3400 + idx * 190), 2)
        cuota = round(min(ingresos * 0.29, amount / plazo * 1.22), 2)
        gastos = round(650 + idx * 55, 2)
        ratio = round(min((cuota + gastos) / ingresos, 0.39), 2)
        payload = {
            "id_documento": record["id_documento"],
            "nombre_cliente": record["nombre_cliente"],
            "dni_cliente": record["dni_cliente"],
            "monto_prestamo_eur": amount,
            "tasa_interes": rate,
            "tae": tae,
            "plazo_meses": plazo,
            "cuota_mensual_eur": cuota,
            "ingresos_mensuales_eur": ingresos,
            "gastos_mensuales_eur": gastos,
            "ratio_endeudamiento": ratio,
            "comision_apertura_eur": round(amount * 0.0045, 2),
            "fein_entregada": True,
            "fiae_entregada": True,
            "sistema_amortizacion": "frances",
            "fecha_emision": issue_date,
        }
        return payload, []

    ingresos = round(max(amount / 120, 2100 + idx * 80), 2)
    cuota = round(ingresos * 0.54, 2)
    gastos = round(980 + idx * 65, 2)
    payload = {
        "id_documento": record["id_documento"],
        "nombre_cliente": record["nombre_cliente"],
        "dni_cliente": record["dni_cliente"],
        "monto_prestamo_eur": amount,
        "tasa_interes": rate,
        "tae": 29.5,
        "plazo_meses": 0,
        "cuota_mensual_eur": cuota,
        "ingresos_mensuales_eur": ingresos,
        "gastos_mensuales_eur": gastos,
        "ratio_endeudamiento": 0.58,
        "comision_apertura_eur": round(amount * 0.011, 2),
        "fein_entregada": False,
        "fiae_entregada": False,
        "sistema_amortizacion": "frances",
        "fecha_emision": issue_date,
    }
    return payload, ["H04", "H05", "H06", "H07", "H08"]


def derive_fiscal(record: dict[str, Any], idx: int, compliant: bool) -> tuple[dict[str, Any], list[str]]:
    base = round(float(record["base_imponible"]), 2)
    issue = date.fromisoformat(record["fecha_expedicion"])
    issue = min(issue, TODAY - timedelta(days=10 + idx))
    if compliant:
        vat = int(record["tipo_iva"])
        quota = round(base * vat / 100, 2)
        withholding = round(base * 0.07, 2) if idx % 2 == 0 else 0.0
        total = round(base + quota - withholding, 2)
        payload = {
            "razon_social_emisor": record["razon_social_emisor"],
            "direccion_emisor": f"Calle Control Interno {40 + idx}, Madrid",
            "nif_emisor": record["nif_emisor"],
            "nombre_destinatario": f"Cliente Auditoria {idx:02d} S.L.",
            "nif_destinatario": f"B{7000000 + idx:07d}{chr(65 + idx)}",
            "num_factura": record["num_factura"],
            "serie_factura": "SER-A",
            "fecha_expedicion": issue.isoformat(),
            "fecha_operacion": (issue - timedelta(days=2)).isoformat(),
            "descripcion_operacion": f"Servicios profesionales de auditoria documental lote {idx:02d}",
            "base_imponible": base,
            "tipo_iva": vat,
            "cuota_iva": quota,
            "retencion_irpf": withholding if withholding > 0 else None,
            "importe_total": total,
        }
        return payload, []

    future_issue = TODAY + timedelta(days=25 + idx)
    payload = {
        "razon_social_emisor": record["razon_social_emisor"],
        "direccion_emisor": f"Avenida Revision Fiscal {80 + idx}, Valencia",
        "nif_emisor": record["nif_emisor"],
        "nombre_destinatario": f"Cliente Riesgo {idx:02d} S.L.",
        "nif_destinatario": f"B{7100000 + idx:07d}{chr(70 + idx)}",
        "num_factura": "",
        "serie_factura": "SER-X",
        "fecha_expedicion": future_issue.isoformat(),
        "fecha_operacion": (future_issue + timedelta(days=4)).isoformat(),
        "descripcion_operacion": f"Servicios con incidencia de control fiscal {idx:02d}",
        "base_imponible": base,
        "tipo_iva": 15,
        "cuota_iva": round(base * 0.12, 2),
        "retencion_irpf": None,
        "importe_total": round(base + (base * 0.12) + 45.5, 2),
    }
    return payload, ["RF01", "RF02", "RF03", "RF04", "RF05", "RF07"]


def derive_kyc(record: dict[str, Any], idx: int, compliant: bool) -> tuple[dict[str, Any], list[str]]:
    birth = date.fromisoformat(record["fecha_nacimiento"])
    if compliant:
        expiry = TODAY + timedelta(days=900 + idx * 12)
        issue = TODAY - timedelta(days=1000 + idx * 30)
        payload = {
            "nombre_titular": record["nombre_titular"],
            "primer_apellido": record["primer_apellido"],
            "segundo_apellido": record["segundo_apellido"],
            "num_documento": record["num_documento"],
            "nacionalidad": "Espana",
            "pais_residencia": "Espana",
            "fecha_nacimiento": birth.isoformat(),
            "fecha_expedicion": issue.isoformat(),
            "fecha_caducidad": expiry.isoformat(),
            "autoridad_emisora": "Direccion General de la Policia",
            "domicilio_comprobante": record["domicilio_comprobante"],
            "actividad_economica": "Profesional por cuenta ajena",
            "proposito_relacion": "Apertura de cuenta y verificacion de identidad",
            "pep": False,
            "titular_real_identificado": True,
            "coincidencia_sanciones": False,
            "nivel_riesgo": "bajo",
        }
        return payload, []

    underage_birth = TODAY - timedelta(days=365 * 16 + idx * 7)
    expiry = TODAY - timedelta(days=40 + idx)
    issue = TODAY + timedelta(days=10 + idx)
    payload = {
        "nombre_titular": record["nombre_titular"],
        "primer_apellido": record["primer_apellido"],
        "segundo_apellido": record["segundo_apellido"],
        "num_documento": record["num_documento"],
        "nacionalidad": "Espana",
        "pais_residencia": "Espana",
        "fecha_nacimiento": underage_birth.isoformat(),
        "fecha_expedicion": issue.isoformat(),
        "fecha_caducidad": expiry.isoformat(),
        "autoridad_emisora": "Direccion General de la Policia",
        "domicilio_comprobante": record["domicilio_comprobante"],
        "actividad_economica": "Actividad no verificada",
        "proposito_relacion": "",
        "pep": True,
        "titular_real_identificado": False,
        "coincidencia_sanciones": True,
        "nivel_riesgo": "bajo",
    }
    return payload, ["RK01", "RK02", "RK04", "RK05", "RK06", "RK07", "RK08"]


def load_semantic_cases() -> list[SemanticCase]:
    semantic_cases: list[SemanticCase] = []

    hipotecario_dir = ROOT / "data" / "sample_docs" / "caso_uso_1_auditoria_hipotecaria"
    for idx in range(1, 11):
        gt_path = hipotecario_dir / f"ground_truth_esp_{idx:04d}.json"
        pdf_path = hipotecario_dir / f"contrato_hipoteca_esp_{idx:04d}.pdf"
        compliant = idx <= 5
        payload, failures = derive_hypotecario(load_json(gt_path), idx, compliant)
        semantic_cases.append(
            SemanticCase(
                schema_name="credito_hipotecario",
                semantic_index=idx,
                compliance_expected="conforme" if compliant else "no_conforme",
                source_pdf_path=pdf_path,
                source_ground_truth_path=gt_path,
                ground_truth=payload,
                expected_rule_failures=failures,
            )
        )

    fiscal_dir = ROOT / "data" / "sample_docs" / "caso_uso_2_auditoria_fiscal"
    for idx in range(1, 11):
        gt_path = fiscal_dir / f"factura_fiscal_{idx}.json"
        pdf_path = fiscal_dir / f"factura_fiscal_{idx}.pdf"
        compliant = idx <= 5
        payload, failures = derive_fiscal(load_json(gt_path), idx, compliant)
        semantic_cases.append(
            SemanticCase(
                schema_name="auditoria_fiscal",
                semantic_index=idx,
                compliance_expected="conforme" if compliant else "no_conforme",
                source_pdf_path=pdf_path,
                source_ground_truth_path=gt_path,
                ground_truth=payload,
                expected_rule_failures=failures,
            )
        )

    kyc_dir = ROOT / "data" / "sample_docs" / "caso_uso_3_kyc_onboarding"
    for idx in range(1, 11):
        gt_path = kyc_dir / f"expediente_kyc_{idx}.json"
        pdf_path = kyc_dir / f"expediente_kyc_{idx}.pdf"
        compliant = idx <= 5
        payload, failures = derive_kyc(load_json(gt_path), idx, compliant)
        semantic_cases.append(
            SemanticCase(
                schema_name="kyc_onboarding",
                semantic_index=idx,
                compliance_expected="conforme" if compliant else "no_conforme",
                source_pdf_path=pdf_path,
                source_ground_truth_path=gt_path,
                ground_truth=payload,
                expected_rule_failures=failures,
            )
        )

    return semantic_cases


def draw_header(pdf: canvas.Canvas, title: str, subtitle: str) -> float:
    width, height = A4
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(20 * mm, height - 20 * mm, title)
    pdf.setFont("Helvetica", 10)
    pdf.drawString(20 * mm, height - 27 * mm, subtitle)
    pdf.line(20 * mm, height - 30 * mm, width - 20 * mm, height - 30 * mm)
    return height - 40 * mm


def draw_pairs(pdf: canvas.Canvas, pairs: list[tuple[str, str]], start_y: float) -> None:
    width, _ = A4
    y = start_y
    left_x = 20 * mm
    right_x = width / 2 + 5 * mm
    pdf.setFont("Helvetica", 10)
    for index in range(0, len(pairs), 2):
        label_left, value_left = pairs[index]
        pdf.drawString(left_x, y, f"{label_left}: {value_left}")
        if index + 1 < len(pairs):
            label_right, value_right = pairs[index + 1]
            pdf.drawString(right_x, y, f"{label_right}: {value_right}")
        y -= 8 * mm


def draw_paragraph(pdf: canvas.Canvas, lines: list[str], start_y: float) -> None:
    text = pdf.beginText(20 * mm, start_y)
    text.setFont("Helvetica", 10)
    for line in lines:
        text.textLine(line)
    pdf.drawText(text)


def render_hypotecario(pdf_path: Path, data: dict[str, Any], compliant: bool) -> None:
    pdf = canvas.Canvas(str(pdf_path), pagesize=A4)
    y = draw_header(
        pdf,
        "Contrato de prestamo hipotecario - corpus de validacion",
        f"Estado esperado: {'conforme' if compliant else 'no conforme'}",
    )
    pairs = [
        ("ID documento", str(data["id_documento"])),
        ("Fecha emision", str(data["fecha_emision"])),
        ("Cliente", str(data["nombre_cliente"])),
        ("DNI/NIE", str(data["dni_cliente"])),
        ("Monto", money(float(data["monto_prestamo_eur"]))),
        ("Interes", f"{float(data['tasa_interes']):.2f}%"),
        ("TAE", f"{float(data['tae']):.2f}%"),
        ("Plazo", f"{int(data['plazo_meses'])} meses"),
        ("Cuota", money(float(data["cuota_mensual_eur"]))),
        ("Ingresos", money(float(data["ingresos_mensuales_eur"]))),
        ("Gastos", money(float(data["gastos_mensuales_eur"]))),
        ("Ratio deuda", f"{float(data['ratio_endeudamiento']):.2f}"),
        ("Comision apertura", money(float(data["comision_apertura_eur"]))),
        ("Sistema", str(data["sistema_amortizacion"])),
        ("FEIN entregada", bool_text(bool(data["fein_entregada"]))),
        ("FIAE entregada", bool_text(bool(data["fiae_entregada"]))),
    ]
    draw_pairs(pdf, pairs, y)
    draw_paragraph(
        pdf,
        [
            "Documento sintetico para evaluacion controlada del caso de uso hipotecario.",
            "Incluye datos economicos y de transparencia precontractual para validar",
            "extraccion, reglas de decision y robustez ante degradaciones visuales.",
        ],
        80 * mm,
    )
    pdf.save()


def render_fiscal(pdf_path: Path, data: dict[str, Any], compliant: bool) -> None:
    pdf = canvas.Canvas(str(pdf_path), pagesize=A4)
    y = draw_header(
        pdf,
        "Factura de proveedor - corpus de validacion fiscal",
        f"Estado esperado: {'conforme' if compliant else 'no conforme'}",
    )
    pairs = [
        ("Emisor", str(data["razon_social_emisor"])),
        ("NIF emisor", str(data["nif_emisor"])),
        ("Direccion", str(data["direccion_emisor"])),
        ("Destinatario", str(data["nombre_destinatario"])),
        ("NIF destinatario", str(data["nif_destinatario"])),
        ("Numero factura", str(data["num_factura"])),
        ("Serie", str(data["serie_factura"])),
        ("Fecha expedicion", str(data["fecha_expedicion"])),
        ("Fecha operacion", str(data["fecha_operacion"])),
        ("Tipo IVA", f"{int(data['tipo_iva'])}%"),
        ("Base imponible", money(float(data["base_imponible"]))),
        ("Cuota IVA", money(float(data["cuota_iva"]))),
        ("Retencion", money(float(data["retencion_irpf"] or 0.0))),
        ("Importe total", money(float(data["importe_total"]))),
    ]
    draw_pairs(pdf, pairs, y)
    draw_paragraph(
        pdf,
        [
            "Descripcion operacion:",
            str(data["descripcion_operacion"]),
            "",
            "Documento sintetico para validar consistencia matematica del IVA,",
            "coherencia de fechas y numeracion obligatoria de la factura.",
        ],
        82 * mm,
    )
    pdf.save()


def render_kyc(pdf_path: Path, data: dict[str, Any], compliant: bool) -> None:
    pdf = canvas.Canvas(str(pdf_path), pagesize=A4)
    y = draw_header(
        pdf,
        "Expediente KYC / onboarding - corpus de validacion",
        f"Estado esperado: {'conforme' if compliant else 'no conforme'}",
    )
    pairs = [
        ("Titular", f"{data['nombre_titular']} {data['primer_apellido']} {data['segundo_apellido']}"),
        ("Documento", str(data["num_documento"])),
        ("Nacimiento", str(data["fecha_nacimiento"])),
        ("Expedicion", str(data["fecha_expedicion"])),
        ("Caducidad", str(data["fecha_caducidad"])),
        ("Nacionalidad", str(data["nacionalidad"])),
        ("Pais residencia", str(data["pais_residencia"])),
        ("Autoridad", str(data["autoridad_emisora"])),
        ("Actividad", str(data["actividad_economica"])),
        ("Proposito", str(data["proposito_relacion"] or " ")),
        ("PEP", bool_text(bool(data["pep"]))),
        ("Titular real", bool_text(bool(data["titular_real_identificado"]))),
        ("Sanciones", bool_text(bool(data["coincidencia_sanciones"]))),
        ("Nivel riesgo", str(data["nivel_riesgo"])),
    ]
    draw_pairs(pdf, pairs, y)
    draw_paragraph(
        pdf,
        [
            "Domicilio comprobante:",
            str(data["domicilio_comprobante"]),
            "",
            "Documento sintetico para validar mayoria de edad, vigencia documental,",
            "riesgo AML/KYC y diligencia reforzada cuando aplica.",
        ],
        82 * mm,
    )
    pdf.save()


def render_native_pdf(pdf_path: Path, schema_name: str, data: dict[str, Any], compliant: bool) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if schema_name == "credito_hipotecario":
        render_hypotecario(pdf_path, data, compliant)
        return
    if schema_name == "auditoria_fiscal":
        render_fiscal(pdf_path, data, compliant)
        return
    render_kyc(pdf_path, data, compliant)


def pdf_to_image(pdf_path: Path) -> Image.Image:
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    finally:
        doc.close()


def add_noise(image: Image.Image, amount: int) -> Image.Image:
    noise = Image.effect_noise(image.size, amount).convert("L")
    noise_rgb = Image.merge("RGB", (noise, noise, noise))
    return ImageChops.add_modulo(image, noise_rgb)


def simulate_scan(image: Image.Image) -> Image.Image:
    img = image.convert("L").convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(0.8)
    img = ImageEnhance.Brightness(img).enhance(1.02)
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    return add_noise(img, 9)


def simulate_photo(image: Image.Image) -> Image.Image:
    img = ImageOps.expand(image, border=24, fill=(247, 245, 240))
    img = img.rotate(-1.4, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(247, 245, 240))
    img = ImageEnhance.Contrast(img).enhance(0.93)
    img = ImageEnhance.Color(img).enhance(0.95)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    shadow = Image.new("RGB", img.size, (16, 16, 16))
    mask = Image.linear_gradient("L").resize(img.size)
    shadow = Image.composite(img, shadow, mask)
    return Image.blend(img, shadow, 0.1)


def simulate_handwritten(image: Image.Image, label: str) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size
    blue = (35, 70, 150)
    red = (150, 35, 35)
    for index in range(3):
        y = int(height * (0.16 + index * 0.10))
        start_x = int(width * 0.60)
        end_x = min(width - 50, start_x + 170)
        draw.line((start_x, y, end_x, y + RNG.randint(-10, 10)), fill=blue, width=3)
    draw.text((int(width * 0.58), int(height * 0.08)), f"Rev. {label}", fill=blue)
    draw.line((int(width * 0.12), int(height * 0.90), int(width * 0.34), int(height * 0.93)), fill=red, width=4)
    return img


def save_variant_image(image: Image.Image, path: Path, variant: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".pdf":
        image.convert("RGB").save(path, "PDF", resolution=150.0)
        return
    quality = 90 if variant == "image_photo" else 92
    image.convert("RGB").save(path, quality=quality)


def variant_extension(variant: str) -> str:
    if variant in {"native_pdf", "scanned_blurry_pdf"}:
        return ".pdf"
    return ".jpg"


def variant_input_kind(variant: str) -> str:
    if variant == "native_pdf":
        return "pdf_native"
    if variant == "scanned_blurry_pdf":
        return "pdf_scanned"
    return "image"


def variant_type(variant: str) -> str:
    if variant == "native_pdf":
        return "native"
    if variant == "scanned_blurry_pdf":
        return "degraded"
    return "image"


def render_variant(native_pdf_path: Path, output_path: Path, variant: str, label: str) -> None:
    if variant == "native_pdf":
        if native_pdf_path != output_path:
            shutil.copy2(native_pdf_path, output_path)
        return
    base_image = pdf_to_image(native_pdf_path)
    if variant == "scanned_blurry_pdf":
        save_variant_image(simulate_scan(base_image), output_path, variant)
        return
    if variant == "image_photo":
        save_variant_image(simulate_photo(base_image), output_path, variant)
        return
    save_variant_image(simulate_handwritten(base_image, label), output_path, variant)


def build_case_id(schema_name: str, variant: str, compliance: str, seq: int) -> str:
    return f"{schema_name}_{variant}_{compliance}_{seq:03d}"


def generate_corpus() -> dict[str, Any]:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    semantic_cases = load_semantic_cases()
    manifest_cases: list[dict[str, Any]] = []
    sequence = 1

    for schema_name in ["credito_hipotecario", "auditoria_fiscal", "kyc_onboarding"]:
        schema_cases = [case for case in semantic_cases if case.schema_name == schema_name]
        for variant in ["native_pdf", "scanned_blurry_pdf", "image_photo", "image_handwritten"]:
            for semantic_case in schema_cases:
                case_id = build_case_id(schema_name, variant, semantic_case.compliance_expected, sequence)
                compliance_dir = OUTPUT_DIR / schema_name / variant / semantic_case.compliance_expected
                final_doc_path = compliance_dir / f"{case_id}{variant_extension(variant)}"
                gt_path = compliance_dir / f"{case_id}_ground_truth.json"
                native_pdf_path = (
                    final_doc_path if variant == "native_pdf" else compliance_dir / f"{case_id}_native_source.pdf"
                )

                render_native_pdf(
                    native_pdf_path,
                    schema_name,
                    semantic_case.ground_truth,
                    semantic_case.compliance_expected == "conforme",
                )
                render_variant(native_pdf_path, final_doc_path, variant, case_id)
                if variant != "native_pdf" and native_pdf_path.exists():
                    native_pdf_path.unlink()

                write_json(gt_path, semantic_case.ground_truth)

                manifest_cases.append(
                    {
                        "id": case_id,
                        "name": f"{schema_name} - {variant} - {semantic_case.compliance_expected} - {semantic_case.semantic_index:02d}",
                        "country": "Espana",
                        "schema_name": schema_name,
                        "type": variant_type(variant),
                        "input_kind": variant_input_kind(variant),
                        "variant": variant,
                        "variant_description": variant,
                        "compliance_expected": semantic_case.compliance_expected,
                        "legal_basis": LEGAL_BASIS[schema_name],
                        "source_pdf_path": str(semantic_case.source_pdf_path.relative_to(ROOT)).replace("\\", "/"),
                        "source_ground_truth_path": str(semantic_case.source_ground_truth_path.relative_to(ROOT)).replace("\\", "/"),
                        "ground_truth": semantic_case.ground_truth,
                        "ground_truth_status": "ready",
                        "acquisition_status": "ready",
                        "prepared_file_path": str(final_doc_path.relative_to(ROOT)).replace("\\", "/"),
                        "prepared_ground_truth_path": str(gt_path.relative_to(ROOT)).replace("\\", "/"),
                        "prepared_status": "generated",
                        "variant_generation_status": "generated",
                        "expected_rule_failures": semantic_case.expected_rule_failures,
                        "semantic_group": f"{schema_name}_seed_{semantic_case.semantic_index:02d}",
                    }
                )
                sequence += 1

    manifest = {
        "country": "Espana",
        "design": {
            "objective": "Corpus experimental aislado para validacion alta de la herramienta",
            "total_tests": len(manifest_cases),
            "use_cases": 3,
            "semantic_cases_per_use_case": 10,
            "variants_per_semantic_case": 4,
            "compliance_split_per_use_case": {"conforme": 5, "no_conforme": 5},
            "visual_variants": ["native_pdf", "scanned_blurry_pdf", "image_photo", "image_handwritten"],
            "notes": [
                "Los documentos se generan en una carpeta separada para no tocar el sistema productivo.",
                "Cada caso semantico se replica en cuatro variantes visuales.",
                "Los no conformes violan reglas normativas de forma controlada.",
            ],
        },
        "test_cases": manifest_cases,
    }
    for helper_pdf in OUTPUT_DIR.rglob("*_native_source.pdf"):
        helper_pdf.unlink()
    write_json(MANIFEST_PATH, manifest)
    return manifest


def main() -> None:
    manifest = generate_corpus()
    print("=" * 80)
    print("Corpus experimental de alta validez generado")
    print(f"Salida: {OUTPUT_DIR}")
    print(f"Manifiesto: {MANIFEST_PATH}")
    print(f"Documentos generados: {len(manifest['test_cases'])}")
    print("=" * 80)


if __name__ == "__main__":
    main()
