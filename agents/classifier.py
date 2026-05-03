from __future__ import annotations


def classify_text(text: str) -> str:
    t = text.lower()

    kyc_keywords = [
        "kyc",
        "onboarding",
        "dni",
        "nie",
        "pasaporte",
        "empadronamiento",
        "código postal",
        "codigo postal",
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
    ]
    if any(k in t for k in hipotecario_keywords):
        return "credito_hipotecario"

    return "credito_hipotecario"
