from __future__ import annotations

from core.pattern_extractor import extract_fields_by_schema_patterns
from core.schema_loader import load_schema


def test_pattern_extractor_can_match_anchored_dni_token():
    schema = load_schema("schemas/credito_hipotecario.yaml")
    text = "Cliente: Juan Perez\nDNI: 22073473S\n"
    fields, details = extract_fields_by_schema_patterns(schema=schema, text=text, pages=[text])
    assert fields["dni_cliente"] == "22073473S"
    assert details["dni_cliente"]["evidencia_textual"] == "22073473S"
    assert details["dni_cliente"]["pagina"] == 1

