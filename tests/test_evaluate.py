import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.evaluate as evaluate
from core.schema_loader import load_schema


def test_extract_from_text_guarded_returns_null_fields_on_timeout(monkeypatch):
    schema = load_schema("schemas/credito_hipotecario.yaml")
    monkeypatch.setattr(evaluate.settings, "enable_extract_llm_subprocess", True)

    def _fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="python extract_fields_worker", timeout=30)

    monkeypatch.setattr(evaluate.subprocess, "run", _fake_run)

    result = evaluate._extract_from_text_guarded(
        text="Contrato de prestamo hipotecario",
        schema=schema,
        pages=["Contrato de prestamo hipotecario"],
        doc_id="doc-test",
    )

    assert set(result["fields"]) == {f.name for f in schema.fields}
    assert all(value is None for value in result["fields"].values())


def test_extract_from_text_guarded_reads_worker_payload(monkeypatch):
    schema = load_schema("schemas/credito_hipotecario.yaml")
    monkeypatch.setattr(evaluate.settings, "enable_extract_llm_subprocess", True)

    def _fake_run(cmd, cwd=None, env=None, stdout=None, stderr=None, timeout=None, **kwargs):
        output_arg = Path(cmd[cmd.index("--output-json") + 1])
        output_arg.write_text(
            json.dumps(
                {
                    "fields": {"nombre_cliente": "Armida Falcon Trillo"},
                    "details": {"nombre_cliente": {"nombre": "nombre_cliente", "pagina": 1}},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Completed()

    monkeypatch.setattr(evaluate.subprocess, "run", _fake_run)

    result = evaluate._extract_from_text_guarded(
        text="Nombre del cliente: Armida Falcon Trillo",
        schema=schema,
        pages=["Nombre del cliente: Armida Falcon Trillo"],
        doc_id="doc-test",
    )

    assert result["fields"]["nombre_cliente"] == "Armida Falcon Trillo"
