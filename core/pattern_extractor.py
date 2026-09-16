from __future__ import annotations

import re
from typing import Any

from core.schema_models import DocSchema, SchemaField


def _get_regex_patterns(field: SchemaField) -> list[str]:
    patterns: list[str] = []
    for rule in field.rules:
        if rule.kind != "regex":
            continue
        pattern = rule.params.get("pattern")
        if isinstance(pattern, str) and pattern.strip():
            patterns.append(pattern.strip())
    return patterns


def _as_token_pattern(pattern: str) -> str | None:
    if pattern.startswith("^") and pattern.endswith("$") and len(pattern) >= 2:
        return pattern[1:-1]
    return None


def _try_parse_number(raw: str) -> float | None:
    s = raw.strip()
    if not s:
        return None
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


def _coerce_value(field: SchemaField, raw: str) -> Any:
    raw = raw.strip()
    if field.type == "string":
        return raw if raw else None
    if field.type == "boolean":
        return True
    if field.type == "integer":
        number = _try_parse_number(raw)
        if number is None:
            return None
        return int(round(number))
    if field.type == "number":
        return _try_parse_number(raw)
    if field.type == "date":
        return raw if raw else None
    if field.type == "datetime":
        return raw if raw else None
    return raw if raw else None


def extract_fields_by_schema_patterns(
    *,
    schema: DocSchema,
    text: str,
    pages: list[str] | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    fields: dict[str, Any] = {}
    details: dict[str, dict[str, Any]] = {}

    haystack = text or ""
    page_list = pages or []

    for field in schema.fields:
        patterns = _get_regex_patterns(field)
        if not patterns:
            continue

        extracted_value: Any = None
        extracted_evidence = ""
        extracted_page = 1

        for pattern in patterns:
            token_pattern = _as_token_pattern(pattern)
            if token_pattern is not None:
                token_re = re.compile(token_pattern)
                tokens = re.findall(r"[A-Za-z0-9]+", haystack)
                for token in tokens:
                    if token_re.fullmatch(token):
                        extracted_value = _coerce_value(field, token)
                        extracted_evidence = token
                        break
                if extracted_value is not None or field.type == "boolean":
                    break
            else:
                compiled = re.compile(pattern, flags=re.IGNORECASE)
                match = compiled.search(haystack)
                if match:
                    if match.lastindex and match.lastindex >= 1:
                        raw = match.group(1)
                    else:
                        raw = match.group(0)
                    extracted_value = _coerce_value(field, raw)
                    extracted_evidence = match.group(0)
                    break

        if extracted_value is None and field.type == "boolean":
            extracted_value = None

        if extracted_value is None:
            continue

        if page_list:
            for idx, page_text in enumerate(page_list, start=1):
                if extracted_evidence and extracted_evidence in page_text:
                    extracted_page = idx
                    break

        fields[field.name] = extracted_value
        details[field.name] = {
            "nombre": field.name,
            "valor": extracted_value,
            "confianza": 0.9,
            "evidencia_textual": extracted_evidence,
            "pagina": extracted_page,
        }

    return fields, details

