"""Tests de auditor, extractor y orquestador.

Estos tests validan:
- carga de esquemas en formato extendido,
- evaluación de reglas de decisión,
- ejecución del orquestador sin depender de un LLM real (monkeypatch),
- robustez del parser JSON.
"""

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
        "nombre_titular": "Juan",
        "primer_apellido": "Pérez",
        "segundo_apellido": "García",
        "num_documento": "12345678Z",
        "fecha_nacimiento": "1990-01-01",
        "fecha_caducidad": "2099-01-01",
        "domicilio_comprobante": "Calle Ejemplo 1, Madrid, 28001",
    }
    validation = validate_extracted(extracted, schema)
    report = audit_document(schema, extracted, validation)
    rules = report["json"]["decision_rules"]
    assert all(r["cumple"] is True for r in rules)


def test_orchestrator_runs_without_llm(monkeypatch):
    import core.orchestrator as orchestrator

    orchestrator._build_graph.cache_clear()
    monkeypatch.setattr(orchestrator, "classify_text", lambda _text: "auditoria_fiscal")
    monkeypatch.setattr(
        orchestrator,
        "extract_from_text",
        lambda _text, _schema, pages=None, doc_id=None: {
            "razon_social_emisor": "Proveedor Demo S.L.",
            "nif_emisor": "A1234567B",
            "num_factura": "F-001",
            "fecha_expedicion": "2026-01-15",
            "base_imponible": 100.0,
            "tipo_iva": 21,
            "cuota_iva": 21.0,
            "retencion_irpf": None,
            "importe_total": 121.0,
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


def test_run_expediente_merges_multi_document_into_one_decision(monkeypatch):
    import core.orchestrator as orchestrator

    def fake_extract(text, schema, pages=None, doc_id=None):
        values_by_field = {
            "base_imponible_general": 30000.0,
            "cuota_liquida_estatal": 2500.0,
            "nif_emisor": "B12345678",
            "importe_total_iva": 121.0,
            "deuda_vigente": 9000.0,
            "incidencias_activas": False,
            "total_gastos_mensuales": 700.0,
        }
        fields = {f.name: values_by_field.get(f.name) for f in schema.fields}
        details = {
            f.name: {
                "nombre": f.name,
                "valor": fields[f.name],
                "confianza": 0.9 if fields[f.name] is not None else None,
                "evidencia_textual": "",
                "pagina": 1,
            }
            for f in schema.fields
        }
        return {"fields": fields, "details": details}

    monkeypatch.setattr(orchestrator, "extract_from_text", fake_extract)

    result = orchestrator.run_expediente(
        [
            "IRPF modelo 100 casilla 435",
            "Factura con IVA",
            "CIRBE incidencias y deuda vigente",
        ],
        schema_name="credito_hipotecario",
    )

    assert result["validation"]["valid"] is True
    rules = {r["id"]: r for r in result["report"]["json"]["decision_rules"]}
    assert rules["R01"]["cumple"] is True
    assert rules["R02"]["cumple"] is True
    assert rules["R03"]["cumple"] is True
