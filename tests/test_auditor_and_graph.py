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
        lambda _text, _schema, pages=None: {
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
