from core.schema_loader import load_schema
from core.validator import validate_extracted


def test_validate_extracted_ok():
    schema = load_schema("schemas/auditoria_fiscal.yaml")
    extracted = {
        "razon_social_emisor": "Proveedor Demo S.L.",
        "nif_emisor": "A1234567B",
        "num_factura": "FAC-2026-0001",
        "fecha_expedicion": "2026-03-25",
        "base_imponible": 2445.37,
        "tipo_iva": 4,
        "cuota_iva": 97.81,
        "retencion_irpf": None,
        "importe_total": 2543.18,
    }
    result = validate_extracted(extracted, schema)
    assert result["valid"] is True
    assert result["issues"] == []


def test_validate_extracted_required_and_min():
    schema = load_schema("schemas/auditoria_fiscal.yaml")
    extracted = {
        "razon_social_emisor": "",
        "nif_emisor": "A1234567B",
        "num_factura": "FAC-2026-0001",
        "fecha_expedicion": "2026-03-25",
        "base_imponible": -1,
        "tipo_iva": 4,
        "cuota_iva": 97.81,
        "retencion_irpf": None,
        "importe_total": 2543.18,
    }
    result = validate_extracted(extracted, schema)
    assert result["valid"] is False
    codes = {(i["field"], i["code"]) for i in result["issues"]}
    assert ("razon_social_emisor", "required") in codes
    assert ("base_imponible", "min") in codes
