from __future__ import annotations

"""
Normalización de campos extraídos.

Convierte salidas del LLM a formatos consistentes antes de validar:
- Trim de strings, conversión de \"null/none/n-a\" a None
- Números con formato local (miles/decimales) a float/int
- Fechas a ISO (YYYY-MM-DD) y datetimes a ISO
- Booleans desde variantes (si/sí/no, 0/1, true/false)
- Ajustes de enums (cuando el schema define valores_permitidos)
"""

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal

from core.schema_models import DocSchema, SchemaField


ChangeType = Literal["coerce", "trim", "parse", "reconcile"]


@dataclass(frozen=True)
class NormalizationChange:
    """Registro de un cambio aplicado durante la normalización."""
    field: str
    kind: ChangeType
    before: Any
    after: Any


def _is_empty(value: Any) -> bool:
    """Indica si un valor debe considerarse vacío (None o string en blanco)."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _normalize_string(value: Any) -> tuple[Any, list[NormalizationChange]]:
    """Recorta espacios en strings y registra el cambio si aplica."""
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
    """Convierte strings numéricos (con posibles miles/decimales locales) a float.

    Ejemplos soportados:
    - "5.489,11 EUR" -> 5489.11
    - "1,250.50" -> 1250.50
    - "9000" -> 9000.0
    """
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

    # Limpia secuencias OCR anómalas como "3,.590,00" antes de decidir separadores.
    s = re.sub(r"(?<=\d)[,\.](?=[,\.])", "", s)

    if re.fullmatch(r"[-+]?\d+([.,]\d+)?", s) is None and re.fullmatch(
        r"[-+]?\d{1,3}([.,]\d{3})+([.,]\d+)?", s
    ) is None:
        # Soporta formatos con múltiples separadores erráticos como "4,160,00" o "5,603,25".
        separators = [idx for idx, ch in enumerate(s) if ch in ",."]
        if not separators:
            return None
        last_sep_idx = separators[-1]
        fractional = s[last_sep_idx + 1 :]
        integer = s[:last_sep_idx]

        if fractional.isdigit() and 1 <= len(fractional) <= 2:
            sign = ""
            if integer[:1] in {"+", "-"}:
                sign = integer[:1]
                integer = integer[1:]
            integer_digits = re.sub(r"[,.]", "", integer)
            if integer_digits.isdigit():
                try:
                    return float(f"{sign}{integer_digits}.{fractional}")
                except ValueError:
                    return None
        return None

    separators = [idx for idx, ch in enumerate(s) if ch in ",."]
    if len(separators) > 1:
        last_sep_idx = separators[-1]
        fractional = s[last_sep_idx + 1 :]
        integer = s[:last_sep_idx]
        sign = ""
        if integer[:1] in {"+", "-"}:
            sign = integer[:1]
            integer = integer[1:]

        if fractional.isdigit() and 1 <= len(fractional) <= 2:
            integer_digits = re.sub(r"[,.]", "", integer)
            if integer_digits.isdigit():
                try:
                    return float(f"{sign}{integer_digits}.{fractional}")
                except ValueError:
                    return None

        compact = f"{sign}{re.sub(r'[,.]', '', integer)}{fractional}"
        try:
            return float(compact)
        except ValueError:
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
    """Normaliza un value a float cuando sea posible."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value), False
    if isinstance(value, str):
        parsed = _parse_number_str(value)
        if parsed is not None:
            return parsed, True
    return value, False


def _normalize_integer(value: Any) -> tuple[Any, bool]:
    """Normaliza un value a int cuando sea un número entero representable."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value, False
    num, changed = _normalize_number(value)
    if isinstance(num, float) and num.is_integer():
        return int(num), True or changed
    return value, False


def _parse_date_str(raw: str) -> str | None:
    """Parsea fechas a ISO (YYYY-MM-DD), soportando formatos comunes (DD/MM/YYYY)."""
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
    """Normaliza fechas a string ISO."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat(), True
    if isinstance(value, str):
        parsed = _parse_date_str(value)
        if parsed is not None:
            return parsed, parsed != value
    return value, False


def _normalize_datetime(value: Any) -> tuple[Any, bool]:
    """Normaliza datetimes a string ISO (YYYY-MM-DDTHH:MM:SS...)."""
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
    """Normaliza strings tipo 'si/no', 'true/false', '0/1' a boolean."""
    if isinstance(value, bool):
        return value, False
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "t", "1", "si", "sí", "s"}:
            return True, True
        if s in {"false", "f", "0", "no", "n"}:
            return False, True
    return value, False


def _get_enum_values(field: SchemaField) -> list[str] | None:
    for rule in field.rules:
        if rule.kind == "enum":
            values = rule.params.get("values")
            if isinstance(values, list):
                return [str(v) for v in values]
            break
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _round_money(value: float) -> float:
    return round(float(value), 2)


def _money_candidates_from_ocr(raw_value: Any) -> list[float]:
    """Genera candidatos monetarios a partir de OCR con separadores perdidos."""
    candidates: list[float] = []

    if _is_number(raw_value):
        return [_round_money(float(raw_value))]

    if not isinstance(raw_value, str):
        return candidates

    raw = raw_value.strip()
    parsed = _parse_number_str(raw)
    if parsed is not None:
        candidates.append(_round_money(parsed))

    digits = re.sub(r"\D", "", raw)
    if digits:
        integer_value = int(digits)
        candidates.append(_round_money(float(integer_value)))
        if len(digits) >= 2:
            candidates.append(_round_money(integer_value / 10.0))
        if len(digits) >= 3:
            candidates.append(_round_money(integer_value / 100.0))
        if len(digits) >= 4:
            candidates.append(_round_money(integer_value / 1000.0))

    deduped: list[float] = []
    seen: set[float] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            deduped.append(candidate)
    return deduped


def _reconcile_money_field(
    *,
    field_name: str,
    expected: float | None,
    extracted: dict[str, Any],
    normalized: dict[str, Any],
    tolerance: float = 0.15,
) -> list[NormalizationChange]:
    """Corrige importes OCR cuando una reinterpretación cuadra con la matemática fiscal."""
    if expected is None:
        return []

    current = normalized.get(field_name)
    current_error = abs(float(current) - expected) if _is_number(current) else float("inf")
    if current_error <= tolerance:
        return []

    raw_value = extracted.get(field_name, current)
    candidates = _money_candidates_from_ocr(raw_value)
    if not candidates:
        return []

    best = min(candidates, key=lambda candidate: abs(candidate - expected))
    best_error = abs(best - expected)
    if best_error > tolerance or best_error >= current_error:
        return []

    normalized[field_name] = best
    return [NormalizationChange(field=field_name, kind="reconcile", before=current, after=best)]


def _infer_legal_vat_rate(normalized: dict[str, Any], allowed_rates: list[int]) -> int | None:
    """Infiere el tipo de IVA más probable usando coherencia matemática."""
    base = normalized.get("base_imponible")
    cuota = normalized.get("cuota_iva")
    total = normalized.get("importe_total")
    retencion = normalized.get("retencion_irpf")

    candidates: list[tuple[float, int]] = []
    if _is_number(base) and float(base) > 0 and _is_number(cuota):
        observed = (float(cuota) / float(base)) * 100.0
        for rate in allowed_rates:
            candidates.append((abs(observed - rate), rate))

    if _is_number(base) and float(base) > 0 and _is_number(total):
        withholding = float(retencion) if _is_number(retencion) else 0.0
        observed = ((float(total) + withholding - float(base)) / float(base)) * 100.0
        for rate in allowed_rates:
            candidates.append((abs(observed - rate), rate))

    if not candidates:
        return None

    error, rate = min(candidates, key=lambda item: item[0])
    return rate if error <= 1.25 else None


def _reconcile_auditoria_fiscal(
    *,
    extracted: dict[str, Any],
    normalized: dict[str, Any],
    schema: DocSchema,
) -> list[NormalizationChange]:
    """Corrige errores OCR frecuentes en porcentajes fiscales."""
    if schema.name != "auditoria_fiscal":
        return []

    tipo_field = next((f for f in schema.fields if f.name == "tipo_iva"), None)
    if tipo_field is None:
        return []

    enum_values = _get_enum_values(tipo_field) or []
    allowed_rates = [int(v) for v in enum_values if re.fullmatch(r"\d+", v)]
    if not allowed_rates:
        return []

    current = normalized.get("tipo_iva")
    raw_value = extracted.get("tipo_iva")
    raw_text = str(raw_value).strip() if raw_value is not None else ""
    compact = re.sub(r"\s+", "", raw_text)
    inferred: int | None = None

    # Errores OCR típicos: "4%" -> "49", "4g", "4o", etc.
    source_candidates = [compact, str(current).strip() if current is not None else ""]
    for source in source_candidates:
        if not source:
            continue
        for rate in sorted(allowed_rates, key=lambda item: len(str(item)), reverse=True):
            prefix = str(rate)
            if not source.startswith(prefix):
                continue
            tail = source[len(prefix):]
            if tail and len(tail) <= 2 and re.fullmatch(r"[%‰º°oOgGqQ9]+", tail):
                inferred = rate
                break
        if inferred is not None:
            break

    changes: list[NormalizationChange] = []

    if inferred is None and current not in allowed_rates:
        inferred = _infer_legal_vat_rate(normalized, allowed_rates)

    if inferred is not None and inferred != current:
        normalized["tipo_iva"] = inferred
        changes.append(
            NormalizationChange(
                field="tipo_iva",
                kind="reconcile",
                before=current,
                after=inferred,
            )
        )

    base = normalized.get("base_imponible")
    tipo_iva = normalized.get("tipo_iva")
    cuota_iva = normalized.get("cuota_iva")
    retencion = normalized.get("retencion_irpf")

    expected_quota: float | None = None
    if _is_number(base) and _is_number(tipo_iva):
        expected_quota = _round_money(float(base) * (float(tipo_iva) / 100.0))
        changes.extend(
            _reconcile_money_field(
                field_name="cuota_iva",
                expected=expected_quota,
                extracted=extracted,
                normalized=normalized,
            )
        )
        cuota_iva = normalized.get("cuota_iva")

    if _is_number(base) and _is_number(cuota_iva):
        withholding = float(retencion) if _is_number(retencion) else 0.0
        expected_total = _round_money(float(base) + float(cuota_iva) - withholding)
        changes.extend(
            _reconcile_money_field(
                field_name="importe_total",
                expected=expected_total,
                extracted=extracted,
                normalized=normalized,
            )
        )

    if "retencion_irpf" in extracted and _is_number(base) and _is_number(cuota_iva) and _is_number(normalized.get("importe_total")):
        expected_retention = _round_money(float(base) + float(cuota_iva) - float(normalized["importe_total"]))
        if expected_retention >= 0:
            changes.extend(
                _reconcile_money_field(
                    field_name="retencion_irpf",
                    expected=expected_retention,
                    extracted=extracted,
                    normalized=normalized,
                )
            )

    return changes


def _normalize_field(field: SchemaField, value: Any) -> tuple[Any, list[NormalizationChange]]:
    """Normaliza un campo según su tipo declarado y reglas (incluyendo enum si aplica)."""
    changes: list[NormalizationChange] = []
    if _is_empty(value):
        return None, changes

    enum_values = _get_enum_values(field)

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
    """Normaliza un dict de campos según el tipo y reglas declaradas del esquema."""
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
        # Compatibilidad: si el schema tiene tipo_documento/numero_documento, inferir el tipo por regex.
        allowed = _get_enum_values(tipo_field)

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

    changes.extend(_reconcile_auditoria_fiscal(extracted=extracted, normalized=normalized, schema=schema))

    autocorrections = [
        {"field": c.field, "kind": c.kind, "before": c.before, "after": c.after}
        for c in changes
        if c.kind == "reconcile"
    ]

    return {
        "normalized": normalized,
        "changes": [
            {"field": c.field, "kind": c.kind, "before": c.before, "after": c.after} for c in changes
        ],
        "autocorrections": autocorrections,
    }
