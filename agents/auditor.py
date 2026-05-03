from __future__ import annotations

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


_ALLOWED_FUNCS: dict[str, Any] = {"abs": abs, "min": min, "max": max, "round": round}


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _safe_eval_expr(expr: str, context: dict[str, Any]) -> bool | None:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            if node.id in context:
                return context[node.id]
            if node.id in _ALLOWED_FUNCS:
                return _ALLOWED_FUNCS[node.id]
            raise ValueError(f"Nombre no permitido: {node.id}")

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
            operand = _eval(node.operand)
            if isinstance(node.op, ast.Not):
                return not bool(operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            return -operand

        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            values = [_eval(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return all(bool(v) for v in values)
            return any(bool(v) for v in values)

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
                else:
                    raise ValueError("Operador no permitido")
                if not ok:
                    return False
                left = right
            return True

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
    context = dict(extracted)
    context["fecha_actual"] = date.today().isoformat()
    context["datetime_actual"] = datetime.now().isoformat()

    results: list[RuleResult] = []
    for rule in schema.decision_rules:
        results.append(_evaluate_one_rule(rule, context))
    return results


def _evaluate_one_rule(rule: DecisionRule, context: dict[str, Any]) -> RuleResult:
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


def _compute_score(extracted: dict[str, Any], validation: dict[str, Any], total_fields: int) -> float:
    filled = sum(1 for v in extracted.values() if not _is_empty(v))
    completeness = (filled / total_fields) if total_fields else 0.0
    valid = bool(validation.get("valid"))
    score = 0.6 * completeness + 0.4 * (1.0 if valid else 0.0)
    return max(0.0, min(1.0, score))


def audit_document(
    schema: DocSchema, extracted: dict[str, Any], validation: dict[str, Any]
) -> dict[str, Any]:
    rule_results = _evaluate_decision_rules(schema, extracted)
    total_fields = len(schema.fields)
    score = _compute_score(extracted, validation, total_fields)

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
        "valid": bool(validation.get("valid")),
        "issues": validation.get("issues", []),
        "decision_rules": rules_json,
    }

    failed_critical = [
        r for r in rule_results if r.severidad == "critica" and r.cumple is False
    ]
    status = "APROBADO" if report_json["valid"] and not failed_critical else "REVISAR"

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
