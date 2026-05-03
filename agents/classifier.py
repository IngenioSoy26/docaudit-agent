from __future__ import annotations


def classify_text(text: str) -> str:
    t = text.lower()

    hipotecario_keywords = [
        "hipoteca",
        "escritura",
        "nota simple",
        "registro de la propiedad",
        "entidad financiera",
        "préstamo",
        "prestamo",
    ]

    if any(k in t for k in hipotecario_keywords):
        return "hipotecario"

    return "hipotecario"
