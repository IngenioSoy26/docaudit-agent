from core.normalizer import normalize_extracted
from core.schema_loader import load_schema


def test_normalize_date_and_number():
    schema = load_schema("schemas/hipotecario.yaml")
    extracted = {
        "titular_nombre": " María López Sánchez ",
        "titular_identificacion": "X1234567T",
        "entidad_financiera": "Banco Ejemplo S.A.",
        "importe_prestamo": "245.000,50 EUR",
        "fecha_firma": "22/11/2023",
    }
    result = normalize_extracted(extracted, schema)
    normalized = result["normalized"]
    assert normalized["titular_nombre"] == "María López Sánchez"
    assert normalized["importe_prestamo"] == 245000.5
    assert normalized["fecha_firma"] == "2023-11-22"


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
