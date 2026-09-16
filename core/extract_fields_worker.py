from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.extractor import extract_from_text
from core.schema_loader import load_schema


def _trace_extract_worker(message: str) -> None:
    trace_path = os.environ.get("DOCAUDIT_EXTRACT_TRACE_PATH", "").strip()
    if not trace_path:
        return
    try:
        with Path(trace_path).open("a", encoding="utf-8") as fh:
            fh.write(message + "\n")
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Worker aislado para extracción estructurada por texto.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output_json)
    _trace_extract_worker("worker:start")
    payload = json.loads(input_path.read_text(encoding="utf-8"))

    schema_name = str(payload["schema_name"])
    text = str(payload.get("text") or "")
    pages = payload.get("pages")
    doc_id = payload.get("doc_id")
    _trace_extract_worker(f"worker:payload schema={schema_name} text_len={len(text)}")

    schema = load_schema(ROOT / "schemas" / f"{schema_name}.yaml")
    _trace_extract_worker("worker:before_extract")
    result = extract_from_text(text, schema, pages=pages, doc_id=doc_id)
    _trace_extract_worker("worker:after_extract")
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _trace_extract_worker("worker:done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
