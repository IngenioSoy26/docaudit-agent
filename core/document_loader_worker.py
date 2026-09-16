from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.document_loader import _extract_text_from_scanned_pdf_bytes_impl


def main() -> int:
    parser = argparse.ArgumentParser(description="Worker aislado para OCR de PDF escaneado.")
    parser.add_argument("--input-pdf", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    pdf_path = Path(args.input_pdf)
    output_path = Path(args.output_json)
    result = _extract_text_from_scanned_pdf_bytes_impl(pdf_path.read_bytes())
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
