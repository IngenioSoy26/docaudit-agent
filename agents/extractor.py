from __future__ import annotations

"""
Agente extractor: extrae campos estructurados a partir de texto.

Estrategia:
- Usa un LLM local (Ollama) para devolver una lista JSON de campos (nombre/valor/confianza/página).
- Post-procesa respuestas “sucias” (comentarios, comas finales, números malformados).
- Completa evidencias por campo con RAG (Chroma) a partir de chunks del documento.

Nota: la ruta de visión/OCR se ejecuta antes del grafo (document_loader) para convertir PDFs escaneados a texto.
"""

import ast
import difflib
import json
import math
import os
import re
from typing import Any
from pathlib import Path

from core.llm import get_text_llm
from core.pattern_extractor import extract_fields_by_schema_patterns
from core.schema_models import DocSchema
from core.rag import (
    build_chunks_from_pages,
    build_chunks_from_text,
    retrieve_best_evidence_batch,
)


def _trace_extract(message: str) -> None:
    trace_path = os.environ.get("DOCAUDIT_EXTRACT_TRACE_PATH", "").strip()
    if not trace_path:
        return
    try:
        with Path(trace_path).open("a", encoding="utf-8") as fh:
            fh.write(message + "\n")
    except Exception:
        pass


def _schema_instructions(schema: DocSchema) -> str:
    """Construye instrucciones para forzar una salida JSON estricta del LLM.

    Args:
        schema: Esquema con la lista exacta de campos esperados.

    Returns:
        Un string con instrucciones y el contrato de salida (lista de CampoExtraido).
    """
    lines: list[str] = []
    lines.append("Devuelve SOLO JSON válido (RFC 8259).")
    lines.append("Usa comillas dobles para claves y strings. No uses comas finales ni comentarios.")
    lines.append("No incluyas comentarios (// o /* */) ni texto fuera del JSON.")
    lines.append("Para números: NO uses separadores de miles (ni '.' ni ','). Usa '.' solo como separador decimal.")
    lines.append("Si el número aparece en formato local (ej: 5.489,11), envíalo como string.")
    lines.append("Devuelve un ÚNICO objeto JSON (dict) con claves=nombre_del_campo y valores=valor_extraído.")
    lines.append('Formato: { "campo_1": valor, "campo_2": valor, ... }')
    lines.append("Los nombres de las claves deben coincidir exactamente con estos campos:")
    for f in schema.fields:
        enum_vals: list[str] | None = None
        for r in f.rules:
            if r.kind == "enum":
                vals = r.params.get("values")
                if isinstance(vals, list) and all(isinstance(x, (str, int, float, bool)) for x in vals):
                    enum_vals = [str(x) for x in vals]
                break
        if enum_vals:
            lines.append(f'- "{f.name}" ({f.type}) requerido={f.required} valores_permitidos={enum_vals}')
        else:
            lines.append(f'- "{f.name}" ({f.type}) requerido={f.required}')
    lines.append("Si no encuentras un valor, usa null.")
    return "\n".join(lines)


def _parse_number_str(raw: str) -> float | None:
    """Intenta convertir un número en formato local (ES/EN) a float.

    Soporta separadores de miles ('.' o ',') y decimales (',' o '.').
    Limpia símbolos monetarios y sufijos comunes.
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
    """Heurística simple para extraer un total aproximado desde un extracto bancario.

    Busca líneas de movimientos (p.ej., transferencias/pagos/cargos) y suma importes
    detectados cerca del trigger.
    """
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


def _ocrish_parse_number(raw: str) -> float | None:
    token = (raw or "").strip()
    if not token:
        return None
    token = re.sub(r"\s+", "", token)
    token = re.sub(r"^[^0-9+\-]+(?=\d)", "", token)
    token = re.sub(r"(?<=\d)[A-Za-z](?=[\d,.\-+])", lambda m: m.group(0).translate(str.maketrans({
        "O": "0",
        "o": "0",
        "Q": "0",
        "q": "0",
        "I": "1",
        "i": "1",
        "L": "1",
        "l": "1",
        "Z": "2",
        "z": "2",
        "S": "5",
        "s": "5",
        "B": "8",
        "b": "8",
        "R": "8",
        "r": "8",
        "A": "4",
        "a": "4",
        "E": "6",
        "e": "6",
    })), token)
    token = re.sub(r"(?<=\d)[A-Za-z]+$", "", token)
    return _parse_number_str(token)


def _ocrish_parse_compact_money(raw: str) -> float | None:
    token = str(raw or "")
    if ":" in token:
        token = token.split(":", 1)[1]
    token = token.upper()
    token = re.sub(r"\s+", "", token)
    token = re.sub(r"EUR$", "", token)
    token = re.sub(r"[^0-9A-Z]", "", token)
    if token.endswith(("E", "S")) and any(ch.isdigit() for ch in token[:-1]):
        token = token[:-1] + "8"
    token = token.translate(str.maketrans({
        "O": "0",
        "Q": "0",
        "G": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "R": "8",
        "S": "8",
        "B": "8",
        "E": "6",
    }))
    digits = re.sub(r"\D", "", token)
    if len(digits) < 3:
        return None
    return round(int(digits) / 100.0, 2)


def _ocrish_parse_percent(raw: str) -> float | None:
    token = (raw or "").upper()
    if not any(ch.isdigit() for ch in token):
        return None
    chunks = [chunk for chunk in re.findall(r"[0-9A-Z]+", token) if any(ch.isdigit() for ch in chunk) or any(ch in "LAES" for ch in chunk)]
    if not chunks:
        return None
    token = chunks[-1]
    leading = token[0] if token else ""
    token = token.translate(str.maketrans({
        "O": "0",
        "Q": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "A": "4",
        "S": "5",
        "B": "8",
        "R": "8",
        "E": "6",
    }))
    digits = re.sub(r"\D", "", token)
    if len(digits) >= 3:
        if leading in {"L", "I"}:
            digits = "6" + digits[1:]
        return round(float(f"{digits[0]}.{digits[1:3]}"), 2)
    if len(digits) == 2:
        return round(float(f"{digits[0]}.{digits[1]}"), 2)
    return _parse_number_str(token)


def _extract_dni_from_labeled_line(raw: str) -> str | None:
    line = str(raw or "").strip().upper()
    if ":" not in line:
        return None
    label, value = line.split(":", 1)
    compact_label = re.sub(r"[^A-Z]", "", label)
    if not compact_label.startswith(("DNI", "DNE", "DANE", "DNNE", "DOCUMENTO")):
        return None

    chunks = [chunk for chunk in re.findall(r"[A-Z0-9]+", value) if any(ch.isdigit() for ch in chunk)]
    if not chunks:
        return None
    token = max(chunks, key=len)
    token = token.translate(str.maketrans({
        "O": "0",
        "Q": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
    }))

    direct = re.search(r"(\d{8})([A-Z])", token)
    if direct:
        return f"{direct.group(1)}{direct.group(2)}"

    trailing_ocr = re.search(r"(\d{8})([0-9A-Z])$", token)
    if trailing_ocr:
        tail = trailing_ocr.group(2).translate(str.maketrans({
            "5": "S",
            "8": "B",
            "0": "O",
            "1": "I",
            "2": "Z",
        }))
        if re.fullmatch(r"[A-Z]", tail):
            return f"{trailing_ocr.group(1)}{tail}"

    return None


def _ocrish_parse_decimal_like(raw: str) -> float | None:
    line = str(raw or "").strip()
    value = line.split(":", 1)[1] if ":" in line else line
    value = re.sub(r"\s+", "", value).upper()
    m = re.search(r"(\d{1,2})[.,]([0-9A-Z]{1,3})", value)
    if not m:
        return None
    left = m.group(1)
    right = m.group(2)
    right = right.translate(
        str.maketrans(
            {
                "O": "0",
                "Q": "0",
                "I": "1",
                "L": "1",
                "Z": "2",
                "A": "4",
                "S": "5",
                "B": "8",
                "R": "8",
                "E": "6",
            }
        )
    )
    digits = re.sub(r"\D", "", right)
    if len(digits) == 0:
        return None
    if len(digits) == 1:
        digits = digits + "0"
    return _parse_number_str(f"{left}.{digits[:2]}")


def _infer_principal_from_payment(payment: float, annual_rate_pct: float, months: int) -> float | None:
    if payment <= 0 or annual_rate_pct < 0 or months <= 0:
        return None
    monthly_rate = (annual_rate_pct / 100.0) / 12.0
    if monthly_rate == 0:
        return payment * months
    try:
        factor = 1.0 - (1.0 + monthly_rate) ** (-months)
    except OverflowError:
        return None
    if factor <= 0:
        return None
    return payment * factor / monthly_rate


def _looks_like_noisy_credito_ocr(text: str) -> bool:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < 6:
        return False
    compact_lines = sum(1 for ln in lines if len(ln.split()) <= 2 and len(ln) >= 6)
    joined = " ".join(lines).upper()
    return compact_lines >= 5 or any(marker in joined for marker in ("HP-000", "CU:", "RATO", "FIAE", "FEIN", "TAE"))


def _heuristic_credito_hipotecario_noisy_ocr(text: str, schema: DocSchema) -> dict[str, Any]:
    if schema.name != "credito_hipotecario":
        return {}
    if not _looks_like_noisy_credito_ocr(text):
        return {}

    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    upper_lines = [ln.upper() for ln in lines]
    fields = {f.name: None for f in schema.fields}

    for ln, upper in zip(lines, upper_lines):
        if fields["id_documento"] is None:
            m = re.search(r"\bH[IP]{1,2}-\d{4}\b", upper)
            if m:
                value = m.group(0).replace("HP-", "HIP-").replace("HPP-", "HIP-")
                fields["id_documento"] = value
            else:
                m = re.search(r"H[IP][A-Z0-9]{0,4}(\d)", upper)
                if m:
                    fields["id_documento"] = f"HIP-000{m.group(1)}"

        if fields["dni_cliente"] is None:
            dni = _extract_dni_from_labeled_line(ln)
            if dni is not None:
                fields["dni_cliente"] = dni

        if fields["fecha_emision"] is None:
            m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", ln)
            if m:
                fields["fecha_emision"] = m.group(1)
            else:
                m = re.search(r"(20\d{2})(\d{2})-(\d{2})", ln)
                if m:
                    fields["fecha_emision"] = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

        if fields["plazo_meses"] is None and ("MES" in upper or upper.startswith(("PA", "PE", "PO"))):
            candidates = [int(v) for v in re.findall(r"(\d{2,3})", ln)]
            if candidates:
                fields["plazo_meses"] = max(candidates)

        if fields["cuota_mensual_eur"] is None and upper.startswith(("CU", "CUO", "CUTA", "CUET")):
            value = _ocrish_parse_number(ln)
            if value is not None and value > 10000:
                value = round(value / 100.0, 2)
            if value is None or value < 50:
                value = _ocrish_parse_compact_money(ln)
            if value is not None and value > 50:
                fields["cuota_mensual_eur"] = round(value, 2)

        if fields["gastos_mensuales_eur"] is None and (upper.startswith("G") or "GAST" in upper):
            value = _ocrish_parse_number(ln)
            if value is not None and value < 100:
                compact = _ocrish_parse_compact_money(ln)
                if compact is not None and compact >= 100:
                    value = compact
            if value is not None and value >= 0:
                fields["gastos_mensuales_eur"] = round(value, 2)

        if fields["ratio_endeudamiento"] is None and (
            upper.startswith(("RAT", "RATO", "RATC"))
            or "RATIO" in upper
            or (
                ":" in ln
                and len(upper) <= 24
                and "EUR" not in upper
                and "%" not in upper
                and re.search(r"0[.,]\d{1,2}", ln)
            )
        ):
            m = re.search(r"(0[.,]\d{1,2})", ln)
            if m:
                ratio = _ocrish_parse_number(m.group(1))
                if ratio is not None and 0 <= ratio <= 1:
                    fields["ratio_endeudamiento"] = round(ratio, 2)

        if fields["tasa_interes"] is None and "RAT" not in upper:
            m = re.search(r"(^|[^0-9])(\d{1,2}[.,]\d{1,2})([^0-9]|$)", ln)
            if m:
                candidate = _ocrish_parse_number(m.group(2))
                if candidate is not None and 1 <= candidate <= 20:
                    fields["tasa_interes"] = round(candidate, 2)
            elif "%" in upper and "TAE" not in upper:
                candidate = _ocrish_parse_percent(ln)
                if candidate is not None and 0 < candidate <= 20:
                    fields["tasa_interes"] = round(candidate, 2)
            elif ":" in ln and len(upper) <= 18 and "TAE" not in upper:
                candidate = _ocrish_parse_decimal_like(ln)
                if candidate is not None and 1 <= candidate <= 20:
                    fields["tasa_interes"] = round(candidate, 2)

        if fields["tae"] is None and "TAE" in upper:
            value = _ocrish_parse_number(ln)
            if value is None or value > 25:
                value = _ocrish_parse_percent(ln)
            if value is not None and 0 < value <= 25:
                fields["tae"] = round(value, 2)

        if fields["comision_apertura_eur"] is None and ("EUR" in upper and ":" in ln) and (upper.startswith("CO") or "COM" in upper or "APERT" in upper):
            value = _ocrish_parse_number(ln)
            if value is None or value < 100:
                value = _ocrish_parse_compact_money(ln)
            if value is not None and value > 0:
                fields["comision_apertura_eur"] = round(value, 2)

        if fields["monto_prestamo_eur"] is None and ":" in ln and ("MORT" in upper or "HOVT" in upper):
            value = _ocrish_parse_number(ln)
            if value is None or value < 10000:
                value = _ocrish_parse_compact_money(ln)
            if value is not None and value > 10000:
                fields["monto_prestamo_eur"] = round(value, 2)

        if fields["fein_entregada"] is None and (re.search(r"F[EA]I[NMWH]", upper) or upper.startswith("FE")):
            fields["fein_entregada"] = True

        if fields["fiae_entregada"] is None and "FIAE" in upper:
            fields["fiae_entregada"] = True

        if fields["sistema_amortizacion"] is None and (
            "SIST" in upper
            or "ITEMA" in upper
            or (":" in ln and upper.startswith(("SI", "ST")) and len(upper) <= 24)
        ):
            suffix = ln.split(":", 1)[-1].strip() if ":" in ln else ln
            letters = re.sub(r"[^a-záéíóúñü]", "", suffix.lower())
            if letters:
                candidate = difflib.get_close_matches(letters, ["frances"], n=1, cutoff=0.3)
                if candidate:
                    fields["sistema_amortizacion"] = candidate[0]

    return fields


def _safe_json_parse(text: str) -> Any:
    """Parsea JSON de salida del LLM de forma robusta.

    En producción, algunos modelos devuelven:
    - fences ```json ... ```
    - comas finales
    - comentarios estilo JS
    - números con miles/decimales en formato local

    Este parser intenta recuperar la estructura sin inventar datos.
    """
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
        # Fallback: intenta recortar a un objeto/lista JSON y corregir fallos comunes.
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
    """Extrae campos estructurados desde texto usando LLM + RAG de evidencias.

    Args:
        text: Texto de entrada (ya extraído del PDF).
        schema: Esquema (campos esperados y reglas).
        pages: Texto por página, si está disponible (mejora chunks y evidencias).
        doc_id: Identificador estable del documento para persistencia del índice RAG.

    Returns:
        Un dict con:
        - {"fields": {...}, "details": {...}} cuando la salida se puede estructurar,
        - o un dict directo de campos si el LLM devolvió formato antiguo.
    """
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

    prefilled_fields = _heuristic_credito_hipotecario_noisy_ocr(text, schema)
    pattern_fields, pattern_details = extract_fields_by_schema_patterns(
        schema=schema,
        text=text,
        pages=pages,
    )
    for name, value in pattern_fields.items():
        if prefilled_fields.get(name) is None:
            prefilled_fields[name] = value
    prefilled_count = sum(value is not None and value != "" for value in prefilled_fields.values())
    if prefilled_count >= 5:
        _trace_extract(f"extract:prefill_only count={prefilled_count}")
        details = {
            name: {"nombre": name, "valor": value, "confianza": 0.55 if value is not None else None, "evidencia_textual": "", "pagina": 1}
            for name, value in prefilled_fields.items()
        }
        for name, detail in pattern_details.items():
            if name in details and isinstance(detail, dict):
                details[name] = detail
        return {"fields": prefilled_fields, "details": details}

    _trace_extract(f"extract:start schema={schema.name} text_len={len(text or '')}")
    try:
        _trace_extract("extract:before_get_llm")
        llm = get_text_llm()
        _trace_extract("extract:after_get_llm")
        prompt = (
            f"{_schema_instructions(schema)}\n\n"
            "Texto de entrada:\n"
            f"{text}\n"
        )
        _trace_extract(f"extract:before_invoke prompt_len={len(prompt)}")
        response = llm.invoke(prompt, stream=False)
        raw = response.content if isinstance(response.content, str) else str(response.content)
        _trace_extract(f"extract:after_invoke raw_len={len(raw)}")
    except Exception:
        _trace_extract("extract:invoke_failed")
        fields = prefilled_fields or {f.name: None for f in schema.fields}
        details = {
            name: {"nombre": name, "valor": fields[name], "confianza": 0.55 if fields[name] is not None else None, "evidencia_textual": "", "pagina": 1}
            for name in fields
        }
        for name, detail in pattern_details.items():
            if name in details and isinstance(detail, dict):
                details[name] = detail
        return {"fields": fields, "details": details}
    allowed = {f.name for f in schema.fields}
    allowed_norm: dict[str, str] = {}
    for name in allowed:
        k = re.sub(r"[^a-z0-9]+", "", name.casefold())
        allowed_norm[k] = name
    _trace_extract("extract:before_parse")
    parsed = _safe_json_parse(raw)
    _trace_extract(f"extract:after_parse type={type(parsed).__name__}")

    fields: dict[str, Any] = {}
    details: dict[str, Any] = {}
    if isinstance(parsed, dict):
        for k, v in parsed.items():
            if not isinstance(k, str):
                continue
            if k in allowed:
                fields[k] = v
                continue
            kn = re.sub(r"[^a-z0-9]+", "", k.casefold())
            mapped = allowed_norm.get(kn)
            if mapped:
                fields[mapped] = v
    elif isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            name = item.get("nombre")
            if not isinstance(name, str):
                continue
            if name not in allowed:
                nn = re.sub(r"[^a-z0-9]+", "", name.casefold())
                mapped = allowed_norm.get(nn)
                if mapped:
                    name = mapped
                else:
                    continue
                continue
            value = item.get("valor")
            fields[name] = value
            details[name] = {
                "nombre": name,
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
        if fields[key] is None and prefilled_fields.get(key) is not None:
            fields[key] = prefilled_fields[key]
            details[key]["valor"] = prefilled_fields[key]
            if details[key].get("confianza") is None:
                details[key]["confianza"] = 0.55
        if fields[key] is None and pattern_fields.get(key) is not None:
            fields[key] = pattern_fields[key]
            details[key]["valor"] = pattern_fields[key]
            if key in pattern_details and isinstance(pattern_details[key], dict):
                details[key] = pattern_details[key]

    try:
        _trace_extract("extract:before_rag")
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
    except Exception:
        _trace_extract("extract:rag_failed")
        pass

    _trace_extract("extract:done")
    return {"fields": fields, "details": details}
