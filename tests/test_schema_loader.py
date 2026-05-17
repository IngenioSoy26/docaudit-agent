from core.schema_loader import load_schema


def test_load_schema_credito_hipotecario():
    schema = load_schema("schemas/credito_hipotecario.yaml")
    assert schema.name == "credito_hipotecario"
    assert schema.version == "1.0"
    assert len(schema.fields) > 0


def test_load_schema_new_format_sets_document_type_on_fields():
    schema = load_schema("schemas/kyc_onboarding.yaml")
    assert schema.name == "kyc_onboarding"
    assert schema.version == "1.0"
    assert len(schema.fields) > 0
    assert any(f.document_type for f in schema.fields)
