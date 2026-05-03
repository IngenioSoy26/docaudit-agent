from __future__ import annotations

from pathlib import Path

import yaml

from core.schema_models import DocSchema


def load_schema(schema_path: str | Path) -> DocSchema:
    path = Path(schema_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return DocSchema.model_validate(raw)
