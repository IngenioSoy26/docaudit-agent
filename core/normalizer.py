from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal

from core.schema_models import DocSchema, SchemaField


ChangeType = Literal["coerce", "trim", "parse"]


@dataclass(frozen=True)
class NormalizationChange:
    field: str
    kind: ChangeType
    before: Any
    after: Any


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _normalize_string(value: Any) -> tuple[Any, list[NormalizationChange]]:
    changes: list[NormalizationChange] = []
    if value is None:
        return value, changes
    if not isinstance(value, str):
        return value, changes
    trimmed = value.strip()
    if trimmed != value:
        changes.append(NormalizationChange(field="", kind="trim", before=value, after=trimmed))
    return trimmed, changes


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

    if re.fullmatch(r"[-+]?\d+([.,]\d+)?", s) is None and re.fullmatch(r"[-+]?\d{1,3}([.,]\d{3})+([.,]\d+)?", s) is None:
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


def _normalize_number(value: Any) -> tuple[Any, bool]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value), False
    if isinstance(value, str):
        parsed = _parse_number_str(value)
        if parsed is not None:
            return parsed, True
    return value, False


def _normalize_integer(value: Any) -> tuple[Any, bool]:
    if isinstance(value, int) and not isinstance(value, bool):
        return value, False
    num, changed = _normalize_number(value)
    if isinstance(num, float) and num.is_integer():
        return int(num), True or changed
    return value, False


def _parse_date_str(raw: str) -> str | None:
    s = raw.strip()
    try:
        return date.fromisoformat(s).isoformat()
    except ValueError:
        pass

    m = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(y, mo, d).isoformat()
        except ValueError:
            return None

    return None


def _normalize_date(value: Any) -> tuple[Any, bool]:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat(), True
    if isinstance(value, str):
        parsed = _parse_date_str(value)
        if parsed is not None:
            return parsed, parsed != value
    return value, False


def _normalize_datetime(value: Any) -> tuple[Any, bool]:
    if isinstance(value, datetime):
        return value.isoformat(), True
    if isinstance(value, str):
        s = value.strip()
        try:
            parsed = datetime.fromisoformat(s)
            return parsed.isoformat(), parsed.isoformat() != value
        except ValueError:
            return value, False
    return value, False


def _normalize_boolean(value: Any) -> tuple[Any, bool]:
    if isinstance(value, bool):
        return value, False
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "t", "1", "si", "sí", "s"}:
            return True, True
        if s in {"false", "f", "0", "no", "n"}:
            return False, True
    return value, False


def _normalize_field(field: SchemaField, value: Any) -> tuple[Any, list[NormalizationChange]]:
    changes: list[NormalizationChange] = []
    if _is_empty(value):
        return None, changes

    enum_values: list[str] | None = None
    for r in field.rules:
        if r.kind == "enum":
            vals = r.params.get("values")
            if isinstance(vals, list) and all(isinstance(x, (str, int, float, bool)) for x in vals):
                enum_values = [str(x) for x in vals]
            break

    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed != value:
            changes.append(NormalizationChange(field=field.name, kind="trim", before=value, after=trimmed))
            value = trimmed

        lower = trimmed.strip().lower()
        if lower in {"null", "none", "no encontrado", "no encontrado.", "n/a", "na"}:
            changes.append(NormalizationChange(field=field.name, kind="coerce", before=trimmed, after=None))
            return None, changes

        if enum_values:
            s = trimmed.upper()
            s = re.sub(r"[\s\.\-_/]+", " ", s).strip()
            canonical = None
            for ev in enum_values:
                if s == ev.upper():
                    canonical = ev
                    break
            if canonical is None:
                if "DNI" in s and any(ev.upper() == "DNI" for ev in enum_values):
                    canonical = next(ev for ev in enum_values if ev.upper() == "DNI")
                elif "NIE" in s and any(ev.upper() == "NIE" for ev in enum_values):
                    canonical = next(ev for ev in enum_values if ev.upper() == "NIE")
                elif "PASAPORTE" in s and any(ev.upper() == "PASAPORTE" for ev in enum_values):
                    canonical = next(ev for ev in enum_values if ev.upper() == "PASAPORTE")
            if canonical is not None and canonical != trimmed:
                changes.append(NormalizationChange(field=field.name, kind="coerce", before=trimmed, after=canonical))
                value = canonical

    if field.type == "string":
        return value, changes
    if field.type == "number":
        after, changed = _normalize_number(value)
        if changed:
            changes.append(NormalizationChange(field=field.name, kind="parse", before=value, after=after))
        return after, changes
    if field.type == "integer":
        after, changed = _normalize_integer(value)
        if changed:
            changes.append(NormalizationChange(field=field.name, kind="parse", before=value, after=after))
        return after, changes
    if field.type == "date":
        after, changed = _normalize_date(value)
        if changed:
            changes.append(NormalizationChange(field=field.name, kind="parse", before=value, after=after))
        return after, changes
    if field.type == "datetime":
        after, changed = _normalize_datetime(value)
        if changed:
            changes.append(NormalizationChange(field=field.name, kind="parse", before=value, after=after))
        return after, changes
    if field.type == "boolean":
        after, changed = _normalize_boolean(value)
        if changed:
            changes.append(NormalizationChange(field=field.name, kind="coerce", before=value, after=after))
        return after, changes

    return value, changes


def normalize_extracted(extracted: dict[str, Any], schema: DocSchema) -> dict[str, Any]:
    normalized: dict[str, Any] = dict(extracted)
    changes: list[NormalizationChange] = []

    for field in schema.fields:
        before = normalized.get(field.name)
        after, field_changes = _normalize_field(field, before)
        normalized[field.name] = after
        changes.extend(field_changes)

    tipo_field = next((f for f in schema.fields if f.name == "tipo_documento"), None)
    num_field = next((f for f in schema.fields if f.name == "numero_documento"), None)
    if tipo_field and num_field:
        allowed: list[str] | None = None
        for r in tipo_field.rules:
            if r.kind == "enum":
                vals = r.params.get("values")
                if isinstance(vals, list):
                    allowed = [str(x) for x in vals]
                break

        current = normalized.get("tipo_documento")
        numero = normalized.get("numero_documento")
        if allowed and (current is None or current not in allowed) and isinstance(numero, str):
            n = numero.strip().upper()
            inferred = None
            if re.fullmatch(r"\d{8}[A-Z]", n) and any(a.upper() == "DNI" for a in allowed):
                inferred = next(a for a in allowed if a.upper() == "DNI")
            elif re.fullmatch(r"[XYZ]\d{7}[A-Z]", n) and any(a.upper() == "NIE" for a in allowed):
                inferred = next(a for a in allowed if a.upper() == "NIE")
            if inferred is not None:
                changes.append(NormalizationChange(field="tipo_documento", kind="coerce", before=current, after=inferred))
                normalized["tipo_documento"] = inferred

    return {
        "normalized": normalized,
        "changes": [
            {"field": c.field, "kind": c.kind, "before": c.before, "after": c.after} for c in changes
        ],
    }
