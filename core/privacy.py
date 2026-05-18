from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Redaction:
    kind: str
    original: str
    replacement: str


_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){8,14}\b")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
_DNI_RE = re.compile(r"\b\d{8}[A-Z]\b")
_NIE_RE = re.compile(r"\b[XYZ]\d{7}[A-Z]\b")
_NIF_CIF_RE = re.compile(r"\b[A-Z]\d{7}[0-9A-Z]\b")


def redact_pii(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Redacta PII frecuente en texto libre (modo local).

    Este redactor es deliberadamente conservador: aplica patrones comunes (email, teléfono,
    IBAN, DNI/NIE/NIF) y sustituye por tokens neutros. No pretende ser un detector perfecto
    de PII, pero permite cumplir el requisito de "anonimización/seudonimización" del pipeline
    cuando se habilita por configuración.
    """

    redactions: list[Redaction] = []
    out = text or ""

    def _sub(kind: str, pattern: re.Pattern[str], token: str) -> None:
        nonlocal out, redactions

        def _repl(m: re.Match[str]) -> str:
            original = m.group(0)
            replacement = token
            redactions.append(Redaction(kind=kind, original=original, replacement=replacement))
            return replacement

        out = pattern.sub(_repl, out)

    _sub("email", _EMAIL_RE, "[EMAIL]")
    _sub("iban", _IBAN_RE, "[IBAN]")
    _sub("dni", _DNI_RE, "[DNI]")
    _sub("nie", _NIE_RE, "[NIE]")
    _sub("nif_cif", _NIF_CIF_RE, "[NIF_CIF]")
    _sub("phone", _PHONE_RE, "[PHONE]")

    return out, [r.__dict__ for r in redactions]

