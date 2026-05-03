from __future__ import annotations

import ast
import json
import re
from typing import Any

from core.llm import get_text_llm
from core.schema_models import DocSchema


def _schema_instructions(schema: DocSchema) -> str:
    lines: list[str] = []
    lines.append("Devuelve SOLO un objeto JSON válido (RFC 8259).")
    lines.append("Usa comillas dobles para claves y strings. No uses comas finales.")
    lines.append("No incluyas comentarios (// o /* */) ni texto fuera del JSON.")
    lines.append("El JSON debe contener exactamente estas claves:")
    for f in schema.fields:
        lines.append(f'- "{f.name}" ({f.type}) requerido={f.required}')
    lines.append("Si no encuentras un valor, usa null.")
    return "\n".join(lines)


def _safe_json_loads(text: str) -> dict[str, Any]:
    raw = (text or "").strip()

    def _ensure_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return {str(k): v for k, v in value.items()}
        raise json.JSONDecodeError("Expected JSON object", raw, 0)

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

    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw, flags=re.IGNORECASE)
    if fence:
        raw = fence.group(1).strip()
    raw = _strip_js_style_comments(raw).strip()

    try:
        return _ensure_dict(json.loads(raw))
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        candidates: list[str] = []
        if start != -1 and end != -1 and end > start:
            candidates.append(raw[start : end + 1].strip())
        candidates.append(raw)

        last_err: Exception | None = None
        for cand in candidates:
            cand = _strip_js_style_comments(cand).strip()
            for attempt in (cand, _remove_trailing_commas(cand)):
                try:
                    return _ensure_dict(json.loads(attempt))
                except Exception as e:
                    last_err = e

            py_like = _remove_trailing_commas(cand)
            py_like = re.sub(r"\bnull\b", "None", py_like, flags=re.IGNORECASE)
            py_like = re.sub(r"\btrue\b", "True", py_like, flags=re.IGNORECASE)
            py_like = re.sub(r"\bfalse\b", "False", py_like, flags=re.IGNORECASE)
            try:
                value = ast.literal_eval(py_like)
                return _ensure_dict(value)
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
    data = _safe_json_loads(raw)

    allowed = {f.name for f in schema.fields}
    cleaned: dict[str, Any] = {k: v for k, v in data.items() if k in allowed}
    for key in allowed:
        cleaned.setdefault(key, None)
    return cleaned
