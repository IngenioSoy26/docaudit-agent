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
