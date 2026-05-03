from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.classifier import classify_text
from agents.extractor import extract_from_text
from core.normalizer import normalize_extracted
from core.schema_loader import load_schema
from core.validator import validate_extracted


def run_pipeline(text: str, schemas_dir: str | Path = "schemas") -> dict[str, Any]:
    schema_name = classify_text(text)
    schema_path = Path(schemas_dir) / f"{schema_name}.yaml"
    schema = load_schema(schema_path)
    extracted_raw = extract_from_text(text, schema)
    normalization = normalize_extracted(extracted_raw, schema)
    extracted = normalization["normalized"]
    validation = validate_extracted(extracted, schema)
    return {
        "schema": {
            "name": schema.name,
            "version": schema.version,
        },
        "extracted_raw": extracted_raw,
        "extracted": extracted,
        "normalization": normalization,
        "validation": validation,
    }
