from agents.auditor import audit_document
from core.schema_loader import load_schema
from core.validator import validate_extracted


def test_load_schema_new_format_has_rules_and_report():
    schema = load_schema("schemas/kyc_onboarding.yaml")
    assert schema.name == "kyc_onboarding"
    assert schema.version == "1.0"
    assert len(schema.fields) > 0
    assert len(schema.decision_rules) > 0
    assert schema.report is not None


def test_auditor_evaluates_decision_rule():
    schema = load_schema("schemas/kyc_onboarding.yaml")
    extracted = {
        "tipo_documento": "DNI",
        "numero_documento": "12345678Z",
        "fecha_caducidad": "2099-01-01",
        "codigo_postal": "28001",
    }
    validation = validate_extracted(extracted, schema)
    report = audit_document(schema, extracted, validation)
    rules = report["json"]["decision_rules"]
    assert rules[0]["cumple"] is True


def test_orchestrator_runs_without_llm(monkeypatch):
    import core.orchestrator as orchestrator

    orchestrator._build_graph.cache_clear()
    monkeypatch.setattr(orchestrator, "classify_text", lambda _text: "auditoria_fiscal")
    monkeypatch.setattr(
        orchestrator,
        "extract_from_text",
        lambda _text, _schema, pages=None, doc_id=None: {
            "numero_factura": "F-001",
            "base_imponible": 100.0,
            "tipo_iva": 21.0,
            "cuota_iva": 21.0,
            "base_iva_devengado": 100.0,
            "cuota_iva_devengado": 21.0,
        },
    )

    result = orchestrator.run_pipeline("FACTURA ...")
    assert "report" in result
    assert "json" in result["report"]
    assert "markdown" in result["report"]


def test_safe_json_parse_fixes_malformed_number_multiple_dots():
    from agents.extractor import _safe_json_parse

    raw = '[{"nombre":"importe","valor": 5.489.11, "confianza":0.9, "evidencia_textual":"", "pagina":1}]'
    parsed = _safe_json_parse(raw)
    assert isinstance(parsed, list)
    assert parsed[0]["valor"] == 5489.11


def test_heuristic_total_gastos_mensuales_from_transferencias():
    from agents.extractor import _heuristic_total_gastos_mensuales

    text = "\n".join(
        [
            "22/07/2025",
            "Transferencia enviada",
            "423,00",
            "4.621,73",
            "07/07/2025",
            "Transferencia enviada",
            "258,21",
            "4.363,52",
        ]
    )
    assert _heuristic_total_gastos_mensuales(text) == 681.21


def test_credito_hipotecario_rules_evaluable_with_complete_context():
    schema = load_schema("schemas/credito_hipotecario.yaml")
    extracted = {
        "base_imponible_general": 30000.0,
        "cuota_liquida_estatal": 2500.0,
        "nif_emisor": "B12345678",
        "importe_total_iva": 121.0,
        "deuda_vigente": 9000.0,
        "incidencias_activas": False,
        "total_gastos_mensuales": 700.0,
    }
    validation = validate_extracted(extracted, schema)
    report = audit_document(schema, extracted, validation)
    rules = {r["id"]: r for r in report["json"]["decision_rules"]}
    assert rules["R01"]["cumple"] is True
    assert rules["R02"]["cumple"] is True
    assert rules["R03"]["cumple"] is True
