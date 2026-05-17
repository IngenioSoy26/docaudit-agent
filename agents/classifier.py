from __future__ import annotations

"""
Agente clasificador: selecciona el caso de uso (schema) más probable para un texto.

Primero usa heurísticas por palabras clave para velocidad y robustez.
Si no hay señales claras, usa un LLM local para escoger entre los esquemas disponibles.
"""

import re

from core.llm import get_classifier_llm

def classify_text(text: str) -> str:
    t = text.lower()

    schemas = ["credito_hipotecario", "auditoria_fiscal", "kyc_onboarding"]
    kyc_keywords = [
        "kyc",
        "onboarding",
        "pasaporte",
        "empadronamiento",
        "código postal",
        "codigo postal",
        "nacionalidad",
        "fecha de caducidad",
        "fecha caducidad",
        "fecha de expedicion",
        "justificante",
    ]
    if any(k in t for k in kyc_keywords):
        return "kyc_onboarding"

    fiscal_keywords = [
        "factura",
        "modelo 303",
        "iva",
        "cuota",
        "base imponible",
        "cif",
    ]
    if any(k in t for k in fiscal_keywords):
        return "auditoria_fiscal"

    hipotecario_keywords = [
        "hipoteca",
        "hipotecario",
        "irpf",
        "extracto bancario",
        "entidad financiera",
        "préstamo",
        "prestamo",
        "nota simple",
        "registro de la propiedad",
        "comparecen",
        "tasa de interes",
        "tasa_interes",
        "monto_prestamo",
    ]
    if any(k in t for k in hipotecario_keywords):
        return "credito_hipotecario"

    try:
        llm = get_classifier_llm()
        prompt = (
            "Clasifica el texto en uno de estos casos de uso. Devuelve SOLO el id exacto.\n"
            f"Opciones: {', '.join(schemas)}\n\n"
            f"Texto:\n{text}\n"
        )
        resp = llm.invoke(prompt, stream=False)
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        raw = raw.strip().lower()
        raw = re.sub(r"[^a-z0-9_]+", "", raw)
        if raw in schemas:
            return raw
    except Exception:
        pass

    return "credito_hipotecario"
