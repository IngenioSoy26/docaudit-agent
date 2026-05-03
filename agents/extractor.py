from __future__ import annotations

import ast
import json
import re
from typing import Any

from core.llm import get_text_llm
from core.schema_models import DocSchema
from core.rag import (
    build_chunks_from_pages,
    build_chunks_from_text,
    retrieve_best_evidence_batch,
)


def _schema_instructions(schema: DocSchema) -> str:
    lines: list[str] = []
    lines.append("Devuelve SOLO JSON válido (RFC 8259).")
    lines.append("Usa comillas dobles para claves y strings. No uses comas finales ni comentarios.")
    lines.append("No incluyas comentarios (// o /* */) ni texto fuera del JSON.")
    lines.append("Para números: NO uses separadores de miles (ni '.' ni ','). Usa '.' solo como separador decimal.")
    lines.append("Si el número aparece en formato local (ej: 5.489,11), envíalo como string.")
    lines.append("Devuelve una lista JSON de objetos CampoExtraido, uno por campo del esquema.")
    lines.append("CampoExtraido = {")
    lines.append('  "nombre": string,')
    lines.append('  "valor": string|number|boolean|null,')
    lines.append('  "confianza": number (0.0-1.0),')
    lines.append('  "evidencia_textual": string (puede ser "" y debe ser corto),')
    lines.append('  "pagina": integer')
    lines.append("}")
    lines.append('Para "evidencia_textual", usa "" (será completado con RAG).')
    lines.append("Los nombres deben coincidir exactamente con estos campos:")
    for f in schema.fields:
        lines.append(f'- "{f.name}" ({f.type}) requerido={f.required}')
    lines.append("Si no encuentras un valor, usa null.")
    lines.append("Si no puedes determinar la página, usa 1.")
    return "\n".join(lines)


def _parse_number_str(raw: str) -> float | None:
    s = raw.strip()
    s = s.replace("€", "").replace("$", "")
    s = re.sub(r"\s+", "", s)
    lower = s.lower()
    lower = (
        lower.replace("eur", "")
        .replace("euros", "")
        .replace("euro", "")
        .replace("usd", "")
        .replace("cop", "")
    )
    s = re.sub(r"[^0-9,.\-+]", "", lower)

    if re.fullmatch(r"[-+]?\d+([.,]\d+)?", s) is None and re.fullmatch(
        r"[-+]?\d{1,3}([.,]\d{3})+([.,]\d+)?", s
    ) is None:
        return None

    has_dot = "." in s
    has_comma = "," in s

    if has_dot and has_comma:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "")
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    elif has_comma and not has_dot:
        s = s.replace(",", ".")

    try:
        return float(s)
    except ValueError:
        return None


def _heuristic_total_gastos_mensuales(text: str) -> float | None:
    t = (text or "").strip()
    if not t:
        return None

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    triggers = [
        "transferencia enviada",
        "transferencia emitida",
        "pago",
        "cargo",
        "adeudo",
        "retirada",
        "compra",
    ]
    num_like = re.compile(r"[-+]?\d[\d\.,]*\d")
    amounts: list[float] = []
    for i, ln in enumerate(lines):
        low = ln.lower()
        if not any(tr in low for tr in triggers):
            continue
        for j in range(i + 1, min(i + 5, len(lines))):
            m = num_like.fullmatch(lines[j])
            if not m:
                continue
            val = _parse_number_str(lines[j])
            if isinstance(val, float):
                amounts.append(abs(val))
                break

    if not amounts:
        return None
    return round(sum(amounts), 2)


def _safe_json_parse(text: str) -> Any:
    raw = (text or "").strip()

    def _fix_malformed_numbers(s: str) -> str:
        def _fix_token(token: str) -> str:
            t = token.strip()

            if "," in t and "." in t:
                if re.fullmatch(r"-?\d{1,3}(?:\.\d{3})+,\d{1,2}", t):
                    left, dec = t.split(",")
                    left = left.replace(".", "")
                    return f"{left}.{dec}"
                if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+\.\d{1,2}", t):
                    return t.replace(",", "")
                return t

            if "," in t:
                if re.fullmatch(r"-?\d+,\d{1,2}", t):
                    return t.replace(",", ".")
                if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+", t):
                    return t.replace(",", "")
                return t

            if "." in t and t.count(".") >= 2 and re.fullmatch(r"-?\d+(?:\.\d+){2,}", t):
                parts = t.split(".")
                last = parts[-1]
                if len(last) in {1, 2}:
                    return f"{''.join(parts[:-1])}.{last}"
                return "".join(parts)

            return t

        def _repl(match: re.Match[str]) -> str:
            prefix = match.group(1)
            token = match.group(2)
            return prefix + _fix_token(token)

        pattern = r'(:\s*)(-?\d[\d\.,]*\d)(?=\s*[,}\]])'
        return re.sub(pattern, _repl, s)

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
            for attempt in (cand, _remove_trailing_commas(cand), _fix_malformed_numbers(_remove_trailing_commas(cand))):
                try:
                    return json.loads(attempt)
                except Exception as e:
                    last_err = e

            py_like = _remove_trailing_commas(cand)
            py_like = _fix_malformed_numbers(py_like)
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


def extract_from_text(
    text: str,
    schema: DocSchema,
    pages: list[str] | None = None,
    doc_id: str | None = None,
) -> dict[str, Any]:
    if schema.name == "credito_hipotecario" and len(schema.fields) == 1:
        only = schema.fields[0]
        if only.name == "total_gastos_mensuales":
            total = _heuristic_total_gastos_mensuales(text)
            if total is not None:
                fields: dict[str, Any] = {"total_gastos_mensuales": total}
                details: dict[str, Any] = {
                    "total_gastos_mensuales": {
                        "nombre": "total_gastos_mensuales",
                        "valor": total,
                        "confianza": 0.7,
                        "evidencia_textual": "",
                        "pagina": 1,
                    }
                }
                return {"fields": fields, "details": details}

    llm = get_text_llm()
    prompt = (
        f"{_schema_instructions(schema)}\n\n"
        "Texto de entrada:\n"
        f"{text}\n"
    )
    response = llm.invoke(prompt, stream=False)
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

    chunks = build_chunks_from_pages(pages) if pages else build_chunks_from_text(text)
    names: list[str] = []
    queries: list[str] = []
    values: list[Any] = []
    for f in schema.fields:
        name = f.name
        label = f.description or name
        value = fields.get(name)
        q = f"{label}"
        if isinstance(value, str) and value.strip():
            q = f"{label}: {value}"
        names.append(name)
        queries.append(q)
        values.append(value)

    hits_by_query = retrieve_best_evidence_batch(queries, chunks, top_k=1, doc_id=doc_id)
    for idx, name in enumerate(names):
        hits = hits_by_query[idx] if idx < len(hits_by_query) else []
        if not hits:
            continue
        hit = hits[0]
        meta = hit.get("metadata") or {}
        if details.get(name) is None or not isinstance(details.get(name), dict):
            details[name] = {"nombre": name, "valor": values[idx], "confianza": None, "evidencia_textual": "", "pagina": 1}
        details[name]["evidencia_textual"] = hit.get("text") or details[name].get("evidencia_textual") or ""
        page = meta.get("page")
        if isinstance(page, int) and page > 0:
            details[name]["pagina"] = page

    return {"fields": fields, "details": details}
