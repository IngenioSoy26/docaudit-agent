from __future__ import annotations

"""
Agente auditor: genera el informe final y evalúa reglas de decisión.

Características:
- Evalúa expresiones booleanas declaradas en YAML (reglas_decision) con un evaluador seguro basado en AST.
- Soporta literales tipo YAML/JSON (true/false/null) y funciones utilitarias (abs/min/max/round/coalesce).
- Produce informe JSON + Markdown y calcula un score de confianza.
"""

import ast
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from core.schema_models import DecisionRule, DocSchema


@dataclass(frozen=True)
class RuleResult:
    id: str | None
    descripcion: str
    expresion: str
    severidad: str
    cumple: bool | None
    error: str | None


def coalesce(*args: Any) -> Any:
    """Devuelve el primer argumento no nulo (None-safe).

    Args:
        *args: Valores candidatos.

    Returns:
        El primer valor distinto de None, o None si todos son None.
    """
    for a in args:
        if a is not None:
            return a
    return None


_ALLOWED_FUNCS: dict[str, Any] = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "coalesce": coalesce,
}


def _is_empty(value: Any) -> bool:
    """Indica si un valor debe considerarse vacío a efectos de completitud."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _safe_eval_expr(expr: str, context: dict[str, Any]) -> bool | None:
    """Evalúa una expresión booleana de forma segura usando un subconjunto de AST.

    La expresión proviene de YAML (reglas_decision). Para evitar ejecución de código
    arbitrario, solo se permiten nodos/operadores explícitos y funciones incluidas
    en `_ALLOWED_FUNCS`.

    Operaciones explícitamente NO permitidas (por diseño):
    - imports, acceso al sistema de archivos o red, y cualquier tipo de E/S.
    - llamadas a funciones arbitrarias (solo `abs`, `min`, `max`, `round`, `coalesce`).
    - acceso a atributos peligrosos (p.ej. `__class__`, `__dict__`, `__globals__`, etc.).
      Solo se permite un subconjunto seguro: `year`, `month`, `day` y `days`.
    - acceso por índice / subscripting (p.ej. `x[0]`), comprehensions, lambdas,
      definición de funciones, asignaciones, imports, acceso a módulos y cualquier
      nodo AST no listado en la implementación.

    Args:
        expr: Expresión booleana a evaluar (p.ej. "importe_total > 0 and iva in [0, 21]").
        context: Variables disponibles (campos extraídos y variables del sistema).

    Returns:
        True/False si se pudo evaluar; None si la expresión no es válida o no evaluable.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        # Literales (números, strings, None, True/False) en Python 3.
        if isinstance(node, ast.Constant):
            return node.value

        # Variables de contexto y literales estilo YAML/JSON (true/false/null).
        if isinstance(node, ast.Name):
            if node.id in context:
                return context[node.id]
            low = node.id.lower()
            if low == "true":
                return True
            if low == "false":
                return False
            if low in {"null", "none"}:
                return None
            if node.id in _ALLOWED_FUNCS:
                return _ALLOWED_FUNCS[node.id]
            raise ValueError(f"Nombre no permitido: {node.id}")

        # Colecciones literales para permitir expresiones tipo `x in ["A", "B"]`.
        if isinstance(node, ast.List):
            return [_eval(elt) for elt in node.elts]

        if isinstance(node, ast.Tuple):
            return tuple(_eval(elt) for elt in node.elts)

        # Acceso controlado a algunos atributos útiles (p.ej., fechas y timedeltas).
        if isinstance(node, ast.Attribute):
            value = _eval(node.value)
            attr = node.attr
            if attr == "days":
                days = getattr(value, "days", None)
                if isinstance(days, int):
                    return days
            if attr in {"year", "month", "day"}:
                part = getattr(value, attr, None)
                if isinstance(part, int):
                    return part
            raise ValueError("Acceso a atributo no permitido")

        # Operadores unarios: +x, -x, not x.
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
            operand = _eval(node.operand)
            if isinstance(node.op, ast.Not):
                return not bool(operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            return -operand

        # Operadores lógicos: and/or.
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            values = [_eval(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return all(bool(v) for v in values)
            return any(bool(v) for v in values)

        # Operadores aritméticos básicos.
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod)
        ):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            return left % right

        # Comparaciones: ==, !=, <, <=, >, >=, in, not in, is, is not (encadenadas).
        if isinstance(node, ast.Compare):
            left = _eval(node.left)
            for op, comp in zip(node.ops, node.comparators, strict=False):
                right = _eval(comp)
                ok: bool
                if isinstance(op, ast.Eq):
                    ok = left == right
                elif isinstance(op, ast.NotEq):
                    ok = left != right
                elif isinstance(op, ast.Lt):
                    ok = left < right
                elif isinstance(op, ast.LtE):
                    ok = left <= right
                elif isinstance(op, ast.Gt):
                    ok = left > right
                elif isinstance(op, ast.GtE):
                    ok = left >= right
                elif isinstance(op, ast.In):
                    ok = left in right
                elif isinstance(op, ast.NotIn):
                    ok = left not in right
                elif isinstance(op, ast.Is):
                    ok = left is right
                elif isinstance(op, ast.IsNot):
                    ok = left is not right
                else:
                    raise ValueError("Operador no permitido")
                if not ok:
                    return False
                left = right
            return True

        # Llamadas a funciones utilitarias explícitamente permitidas.
        if isinstance(node, ast.Call):
            func = _eval(node.func)
            if func not in _ALLOWED_FUNCS.values():
                raise ValueError("Función no permitida")
            args = [_eval(a) for a in node.args]
            kwargs = {kw.arg: _eval(kw.value) for kw in node.keywords if kw.arg}
            return func(*args, **kwargs)

        raise ValueError("Expresión no permitida")

    try:
        result = _eval(tree)
        return bool(result)
    except Exception:
        return None


def _evaluate_decision_rules(schema: DocSchema, extracted: dict[str, Any]) -> list[RuleResult]:
    """Evalúa todas las reglas de decisión declaradas en el esquema."""
    context = _prepare_context(extracted)
    context["fecha_actual"] = date.today()
    context["datetime_actual"] = datetime.now()

    results: list[RuleResult] = []
    for rule in schema.decision_rules:
        results.append(_evaluate_one_rule(rule, context))
    return results


def _prepare_context(extracted: dict[str, Any]) -> dict[str, Any]:
    """Prepara el contexto de evaluación (incluye parseo de fechas ISO a date/datetime)."""
    ctx: dict[str, Any] = dict(extracted)
    for k, v in list(ctx.items()):
        if not isinstance(v, str):
            continue
        s = v.strip()
        if not s:
            continue
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            try:
                ctx[k] = date.fromisoformat(s)
            except Exception:
                pass
            continue
        if len(s) >= 19 and s[4] == "-" and s[7] == "-" and "T" in s:
            try:
                ctx[k] = datetime.fromisoformat(s)
            except Exception:
                pass
    return ctx


def _evaluate_one_rule(rule: DecisionRule, context: dict[str, Any]) -> RuleResult:
    """Evalúa una regla y devuelve un resultado estructurado (incluye errores de evaluación)."""
    expr = rule.expresion
    ok = _safe_eval_expr(expr, context)
    if ok is None:
        return RuleResult(
            id=rule.id,
            descripcion=rule.descripcion,
            expresion=expr,
            severidad=rule.severidad,
            cumple=None,
            error="No se pudo evaluar la expresión.",
        )
    return RuleResult(
        id=rule.id,
        descripcion=rule.descripcion,
        expresion=expr,
        severidad=rule.severidad,
        cumple=ok,
        error=None,
    )


def _compute_score(
    extracted: dict[str, Any],
    validation: dict[str, Any],
    total_fields: int,
    avg_confidence: float | None,
) -> float:
    """Calcula un score (0-1) combinando completitud, validez y confianza media (si existe)."""
    filled = sum(1 for v in extracted.values() if not _is_empty(v))
    completeness = (filled / total_fields) if total_fields else 0.0
    valid = bool(validation.get("valid"))
    conf = avg_confidence if isinstance(avg_confidence, (int, float)) else None
    if conf is None:
        score = 0.6 * completeness + 0.4 * (1.0 if valid else 0.0)
    else:
        score = 0.4 * completeness + 0.3 * (1.0 if valid else 0.0) + 0.3 * float(conf)
    return max(0.0, min(1.0, score))


def audit_document(
    schema: DocSchema,
    extracted: dict[str, Any],
    validation: dict[str, Any],
    field_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Genera el informe final de auditoría (JSON y Markdown) para un documento o expediente.

    Args:
        schema: Esquema aplicado (campos + reglas de decisión).
        extracted: Campos ya normalizados.
        validation: Resultado de validación (valid/issues).
        field_details: Metadatos por campo (confianza, evidencia, página) si están disponibles.

    Returns:
        Un dict con dos vistas: `json` (estructura completa) y `markdown` (resumen legible).
    """
    rule_results = _evaluate_decision_rules(schema, extracted)
    total_fields = len(schema.fields)
    details = field_details or {}
    confidences: list[float] = []
    for v in details.values():
        if isinstance(v, dict):
            c = v.get("confianza")
            if isinstance(c, (int, float)):
                confidences.append(float(c))
    avg_confidence = (sum(confidences) / len(confidences)) if confidences else None
    score = _compute_score(extracted, validation, total_fields, avg_confidence)

    rules_json = [
        {
            "id": r.id,
            "descripcion": r.descripcion,
            "expresion": r.expresion,
            "severidad": r.severidad,
            "cumple": r.cumple,
            "error": r.error,
        }
        for r in rule_results
    ]

    report_json = {
        "schema": {"name": schema.name, "version": schema.version},
        "score_confianza": score,
        "confianza_media_campos": avg_confidence,
        "valid": bool(validation.get("valid")),
        "issues": validation.get("issues", []),
        "decision_rules": rules_json,
        "campos": details,
    }

    failed_critical = [
        r for r in rule_results if r.severidad == "critica" and r.cumple is False
    ]
    unknown_critical = [
        r for r in rule_results if r.severidad == "critica" and r.cumple is None
    ]
    status = (
        "APROBADO"
        if report_json["valid"] and not failed_critical and not unknown_critical
        else "REVISAR"
    )

    md_lines: list[str] = []
    md_lines.append(f"# Informe de auditoría — {schema.name}")
    md_lines.append("")
    md_lines.append(f"- Estado: {status}")
    md_lines.append(f"- Válido (validación de campos): {report_json['valid']}")
    md_lines.append(f"- Score confianza (0-1): {score:.2f}")
    md_lines.append("")

    issues = report_json.get("issues") or []
    md_lines.append("## Incidencias")
    if issues:
        for i in issues:
            md_lines.append(f"- [{i.get('level')}] {i.get('field')}: {i.get('message')}")
    else:
        md_lines.append("- Sin incidencias.")
    md_lines.append("")

    md_lines.append("## Reglas de decisión")
    if rule_results:
        for r in rule_results:
            if r.cumple is True:
                res = "OK"
            elif r.cumple is False:
                res = "NO CUMPLE"
            else:
                res = "NO EVALUABLE"
            rid = f"{r.id} — " if r.id else ""
            md_lines.append(f"- {rid}{r.descripcion}: {res}")
    else:
        md_lines.append("- No hay reglas definidas.")

    report_markdown = "\n".join(md_lines).strip() + "\n"
    return {"json": report_json, "markdown": report_markdown}
