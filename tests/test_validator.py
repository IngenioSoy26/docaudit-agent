from core.schema_loader import load_schema
from core.validator import validate_extracted


def test_validate_extracted_ok():
    schema = load_schema("schemas/hipotecario.yaml")
    extracted = {
        "titular_nombre": "María López Sánchez",
        "titular_identificacion": "X1234567T",
        "entidad_financiera": "Banco Ejemplo S.A.",
        "importe_prestamo": 245000,
        "fecha_firma": "2023-11-22",
    }
    result = validate_extracted(extracted, schema)
    assert result["valid"] is True
    assert result["issues"] == []


def test_validate_extracted_required_and_min():
    schema = load_schema("schemas/hipotecario.yaml")
    extracted = {
        "titular_nombre": None,
        "titular_identificacion": "",
        "entidad_financiera": "Banco X",
        "importe_prestamo": -1,
        "fecha_firma": "2023-11-22",
    }
    result = validate_extracted(extracted, schema)
    assert result["valid"] is False
    codes = {(i["field"], i["code"]) for i in result["issues"]}
    assert ("titular_nombre", "required") in codes
    assert ("titular_identificacion", "required") in codes
    assert ("importe_prestamo", "min") in codes
