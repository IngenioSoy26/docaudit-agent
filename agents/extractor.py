from __future__ import annotations

import ast
import json
import re
from typing import Any

from core.llm import get_text_llm
from core.schema_models import DocSchema


def _schema_instructions(schema: DocSchema) -> str:
    lines: list[str] = []
    lines.append("Devuelve SOLO JSON válido (RFC 8259).")
    lines.append("Usa comillas dobles para claves y strings. No uses comas finales ni comentarios.")
    lines.append("No incluyas comentarios (// o /* */) ni texto fuera del JSON.")
    lines.append("Devuelve una lista JSON de objetos CampoExtraido, uno por campo del esquema.")
    lines.append("CampoExtraido = {")
    lines.append('  "nombre": string,')
    lines.append('  "valor": string|number|boolean|null,')
    lines.append('  "confianza": number (0.0-1.0),')
    lines.append('  "evidencia_textual": string,')
    lines.append('  "pagina": integer')
    lines.append("}")
    lines.append("Los nombres deben coincidir exactamente con estos campos:")
    for f in schema.fields:
        lines.append(f'- "{f.name}" ({f.type}) requerido={f.required}')
    lines.append("Si no encuentras un valor, usa null.")
    lines.append("Si no puedes determinar la página, usa 1.")
    return "\n".join(lines)


def _safe_json_parse(text: str) -> Any:
    raw = (text or "").strip()

    def _strip_js_style_comments(s: str) -> str:
        out: list[str] = []
        i = 0
        in_string = False
        string_quote = ""
        escape = False
        while i < len(s):
            ch = s[i]
            if in_string:
                out.append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == string_quote:
                    in_string = False
                i += 1
                continue

            if ch in {'"', "'"}:
                in_string = True
                string_quote = ch
                out.append(ch)
                i += 1
                continue

            if ch == "/" and i + 1 < len(s) and s[i + 1] == "/":
                i += 2
                while i < len(s) and s[i] not in "\r\n":
                    i += 1
                continue

            if ch == "/" and i + 1 < len(s) and s[i + 1] == "*":
                i += 2
                while i + 1 < len(s) and not (s[i] == "*" and s[i + 1] == "/"):
                    i += 1
                i = i + 2 if i + 1 < len(s) else len(s)
                continue

            out.append(ch)
            i += 1
        return "".join(out)

    def _remove_trailing_commas(s: str) -> str:
        prev = None
        cur = s
        while prev != cur:
            prev = cur
            cur = re.sub(r",(\s*[}\]])", r"\1", cur)
        return cur

    fence = re.search(
        r"```(?:json)?\s*([\s\S]*?)\s*```", raw, flags=re.IGNORECASE
    )
    if fence:
        raw = fence.group(1).strip()
    raw = _strip_js_style_comments(raw).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        obj_start = raw.find("{")
        obj_end = raw.rfind("}")
        arr_start = raw.find("[")
        arr_end = raw.rfind("]")
        candidates: list[str] = []
        if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
            candidates.append(raw[arr_start : arr_end + 1].strip())
        if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
            candidates.append(raw[obj_start : obj_end + 1].strip())
        candidates.append(raw)

        last_err: Exception | None = None
        for cand in candidates:
            cand = _strip_js_style_comments(cand).strip()
            for attempt in (cand, _remove_trailing_commas(cand)):
                try:
                    return json.loads(attempt)
                except Exception as e:
                    last_err = e

            py_like = _remove_trailing_commas(cand)
            py_like = re.sub(r"\bnull\b", "None", py_like, flags=re.IGNORECASE)
            py_like = re.sub(r"\btrue\b", "True", py_like, flags=re.IGNORECASE)
            py_like = re.sub(r"\bfalse\b", "False", py_like, flags=re.IGNORECASE)
            try:
                return ast.literal_eval(py_like)
            except Exception as e:
                last_err = e

        if last_err:
            raise last_err
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
    allowed = {f.name for f in schema.fields}
    parsed = _safe_json_parse(raw)

    if isinstance(parsed, dict):
        cleaned: dict[str, Any] = {k: v for k, v in parsed.items() if k in allowed}
        for key in allowed:
            cleaned.setdefault(key, None)
        return cleaned

    fields: dict[str, Any] = {}
    details: dict[str, Any] = {}
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            name = item.get("nombre")
            if not isinstance(name, str) or name not in allowed:
                continue
            value = item.get("valor")
            fields[name] = value
            details[name] = {
                "nombre": name,
                "valor": value,
                "confianza": item.get("confianza"),
                "evidencia_textual": item.get("evidencia_textual"),
                "pagina": item.get("pagina"),
            }

    for key in allowed:
        fields.setdefault(key, None)
        details.setdefault(
            key,
            {"nombre": key, "valor": fields[key], "confianza": None, "evidencia_textual": "", "pagina": 1},
        )

    return {"fields": fields, "details": details}
