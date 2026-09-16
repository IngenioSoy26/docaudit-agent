from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.document_loader import _ocr_best_text


def _trace(message: str) -> None:
    trace_path = os.environ.get("DOCAUDIT_OCR_TRACE_PATH", "").strip()
    if not trace_path:
        return
    try:
        with Path(trace_path).open("a", encoding="utf-8") as fh:
            fh.write(message + "\n")
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Worker aislado por página para OCR local.")
    parser.add_argument("--backend", required=True, choices=["easyocr", "paddleocr", "rapidocr"])
    parser.add_argument("--input-image", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    image_path = Path(args.input_image)
    output_path = Path(args.output_json)
    _trace(f"ocr_worker:start backend={args.backend}")
    text = _ocr_best_text(image_path.read_bytes(), args.backend)
    _trace(f"ocr_worker:done backend={args.backend} text_len={len(text)}")
    output_path.write_text(json.dumps({"text": text}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
