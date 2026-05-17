from core.normalizer import normalize_extracted
from core.schema_loader import load_schema


def test_normalize_date_and_number():
    schema = load_schema("schemas/auditoria_fiscal.yaml")
    extracted = {
        "razon_social_emisor": "  Proveedor Demo S.L. ",
        "nif_emisor": "A1234567B",
        "num_factura": "FAC-2026-0001",
        "fecha_expedicion": "25/03/2026",
        "base_imponible": "2.445,37 EUR",
        "tipo_iva": "4",
        "cuota_iva": "97,81",
        "importe_total": "2.543,18",
    }
    result = normalize_extracted(extracted, schema)
    normalized = result["normalized"]
    assert normalized["razon_social_emisor"] == "Proveedor Demo S.L."
    assert normalized["base_imponible"] == 2445.37
    assert normalized["tipo_iva"] == 4
    assert normalized["fecha_expedicion"] == "2026-03-25"


def test_normalize_enum_tipo_documento_from_noisy_string():
    schema = load_schema("schemas/auditoria_fiscal.yaml")
    extracted = {"tipo_iva": "21"}
    result = normalize_extracted(extracted, schema)
    assert result["normalized"]["tipo_iva"] == 21


def test_normalize_null_string_to_none():
    schema = load_schema("schemas/kyc_onboarding.yaml")
    extracted = {"fecha_caducidad": "null"}
    result = normalize_extracted(extracted, schema)
    assert result["normalized"]["fecha_caducidad"] is None


def test_infer_tipo_documento_from_numero_documento():
    schema = load_schema("schemas/kyc_onboarding.yaml")
    extracted = {"fecha_nacimiento": "08/12/1992"}
    result = normalize_extracted(extracted, schema)
    assert result["normalized"]["fecha_nacimiento"] == "1992-12-08"
