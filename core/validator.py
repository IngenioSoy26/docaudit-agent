from __future__ import annotations

"""
Validación de campos extraídos contra un esquema.

Aplica:
- Required (campo requerido)
- Tipos primitivos (string/number/integer/boolean/date/datetime)
- Reglas declarativas por campo (min/max/regex/enum)

Este módulo valida el resultado ya normalizado (ver core.normalizer).
"""

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal

from core.schema_models import DocSchema, SchemaField


IssueLevel = Literal["error", "warning"]


@dataclass(frozen=True)
class ValidationIssue:
    """Representa una incidencia de validación asociada a un campo."""
    level: IssueLevel
    field: str
    code: str
    message: str


def _is_empty(value: Any) -> bool:
    """Indica si un valor se considera vacío (None o string en blanco)."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _try_parse_date(value: Any) -> bool:
    """Verifica si un valor es una fecha válida (date o string ISO YYYY-MM-DD)."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return True
    if isinstance(value, str):
        try:
            date.fromisoformat(value)
            return True
        except ValueError:
            return False
    return False


def _try_parse_datetime(value: Any) -> bool:
    """Verifica si un valor es un datetime válido (datetime o string ISO)."""
    if isinstance(value, datetime):
        return True
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
            return True
        except ValueError:
            return False
    return False


def _is_number(value: Any) -> bool:
    """Indica si un valor puede interpretarse como número (int/float o string numérico)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return True
    if isinstance(value, str):
        try:
            float(value)
            return True
        except ValueError:
            return False
    return False


def _as_float(value: Any) -> float | None:
    """Convierte un valor a float si es posible; devuelve None si no."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _as_int(value: Any) -> int | None:
    """Convierte un valor a int si representa un entero exacto; si no, devuelve None."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        try:
            as_float = float(value)
        except ValueError:
            return None
        if as_float.is_integer():
            return int(as_float)
        return None
    return None


def _check_type(field: SchemaField, value: Any) -> list[ValidationIssue]:
    """Valida el tipo de un campo según `field.type`."""
    issues: list[ValidationIssue] = []
    t = field.type
    if _is_empty(value):
        return issues

    if t == "string":
        if not isinstance(value, str):
            issues.append(
                ValidationIssue(
                    level="error",
                    field=field.name,
                    code="type",
                    message="Se esperaba string.",
                )
            )
    elif t == "number":
        if not _is_number(value):
            issues.append(
                ValidationIssue(
                    level="error",
                    field=field.name,
                    code="type",
                    message="Se esperaba number.",
                )
            )
    elif t == "integer":
        if _as_int(value) is None:
            issues.append(
                ValidationIssue(
                    level="error",
                    field=field.name,
                    code="type",
                    message="Se esperaba integer.",
                )
            )
    elif t == "boolean":
        if not isinstance(value, bool):
            issues.append(
                ValidationIssue(
                    level="error",
                    field=field.name,
                    code="type",
                    message="Se esperaba boolean.",
                )
            )
    elif t == "date":
        if not _try_parse_date(value):
            issues.append(
                ValidationIssue(
                    level="error",
                    field=field.name,
                    code="type",
                    message="Se esperaba date (ISO: YYYY-MM-DD).",
                )
            )
    elif t == "datetime":
        if not _try_parse_datetime(value):
            issues.append(
                ValidationIssue(
                    level="error",
                    field=field.name,
                    code="type",
                    message="Se esperaba datetime (ISO).",
                )
            )
    return issues


def _check_rules(field: SchemaField, value: Any) -> list[ValidationIssue]:
    """Aplica reglas declarativas por campo (min/max/regex/enum) si el valor no está vacío."""
    issues: list[ValidationIssue] = []
    if _is_empty(value):
        return issues

    for rule in field.rules:
        kind = rule.kind
        params = rule.params or {}

        if kind == "min":
            minimum = params.get("value")
            as_float = _as_float(value)
            if minimum is not None and as_float is not None and as_float < float(minimum):
                issues.append(
                    ValidationIssue(
                        level="error",
                        field=field.name,
                        code="min",
                        message=f"Debe ser >= {minimum}.",
                    )
                )
        elif kind == "max":
            maximum = params.get("value")
            as_float = _as_float(value)
            if maximum is not None and as_float is not None and as_float > float(maximum):
                issues.append(
                    ValidationIssue(
                        level="error",
                        field=field.name,
                        code="max",
                        message=f"Debe ser <= {maximum}.",
                    )
                )
        elif kind == "regex":
            pattern = params.get("pattern")
            if isinstance(pattern, str) and isinstance(value, str):
                if re.fullmatch(pattern, value) is None:
                    issues.append(
                        ValidationIssue(
                            level="error",
                            field=field.name,
                            code="regex",
                            message="No cumple el patrón esperado.",
                        )
                    )
        elif kind == "enum":
            allowed = params.get("values")
            if isinstance(allowed, list) and value not in allowed:
                issues.append(
                    ValidationIssue(
                        level="error",
                        field=field.name,
                        code="enum",
                        message=f"Valor no permitido. Permitidos: {allowed}.",
                    )
                )

    return issues


def validate_extracted(extracted: dict[str, Any], schema: DocSchema) -> dict[str, Any]:
    """Valida los campos extraídos contra el esquema.

    Se asume que `extracted` ya está normalizado (ver core.normalizer), para que la
    validación opere sobre formatos consistentes.

    Args:
        extracted: Dict de campos normalizados.
        schema: Esquema con campos y reglas.

    Returns:
        Un dict con:
        - valid: bool
        - issues: lista serializable de incidencias
    """
    issues: list[ValidationIssue] = []

    for field in schema.fields:
        value = extracted.get(field.name)

        if field.required and _is_empty(value):
            issues.append(
                ValidationIssue(
                    level="error",
                    field=field.name,
                    code="required",
                    message="Campo requerido ausente.",
                )
            )
            continue

        issues.extend(_check_type(field, value))
        issues.extend(_check_rules(field, value))

    return {
        "valid": not any(i.level == "error" for i in issues),
        "issues": [
            {"level": i.level, "field": i.field, "code": i.code, "message": i.message}
            for i in issues
        ],
    }
