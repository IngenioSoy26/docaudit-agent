from __future__ import annotations

import json
from typing import Any

from core.llm import get_text_llm
from core.schema_models import DocSchema


def _schema_instructions(schema: DocSchema) -> str:
    lines: list[str] = []
    lines.append("Devuelve SOLO un objeto JSON válido.")
    lines.append("El JSON debe contener exactamente estas claves:")
    for f in schema.fields:
        lines.append(f'- "{f.name}" ({f.type}) requerido={f.required}')
    lines.append("Si no encuentras un valor, usa null.")
    return "\n".join(lines)


def _safe_json_loads(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def extract_from_text(text: str, schema: DocSchema) -> dict[str, Any]:
    llm = get_text_llm()
    prompt = (
        f"{_schema_instructions(schema)}\n\n"
        "Texto de entrada:\n"
        f"{text}\n"
    )
    response = llm.invoke(prompt)
    raw = response.content if isinstance(response.content, str) else str(response.content)
    data = _safe_json_loads(raw)

    allowed = {f.name for f in schema.fields}
    cleaned: dict[str, Any] = {k: v for k, v in data.items() if k in allowed}
    for key in allowed:
        cleaned.setdefault(key, None)
    return cleaned
