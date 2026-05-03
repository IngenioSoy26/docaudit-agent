from core.schema_loader import load_schema


def test_load_schema_hipotecario():
    schema = load_schema("schemas/hipotecario.yaml")
    assert schema.name == "hipotecario"
    assert schema.version == "0.1.0"
    assert len(schema.fields) > 0
