from __future__ import annotations

import argparse
import io
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def _trace(message: str) -> None:
    trace_path = os.environ.get("DOCAUDIT_OCR_TRACE_PATH", "").strip()
    if not trace_path:
        return
    try:
        with Path(trace_path).open("a", encoding="utf-8") as fh:
            fh.write(message + "\n")
    except Exception:
        pass


def _text_signal_score(text: str) -> float:
    normalized = " ".join((text or "").split())
    if not normalized:
        return 0.0
    letters = sum(ch.isalpha() for ch in normalized)
    digits = sum(ch.isdigit() for ch in normalized)
    symbols = sum(1 for ch in normalized if not ch.isalnum() and not ch.isspace())
    word_like = len(re.findall(r"[A-Za-z0-9ÁÉÍÓÚÜÑáéíóúüñ]{2,}", normalized))
    return (letters * 1.6) + (digits * 0.7) + (word_like * 7.5) - (symbols * 2.0)


def _score_ocr_candidate(text: str, confidences: list[float] | None = None) -> float:
    normalized = " ".join((text or "").split())
    avg_conf = (sum(confidences or []) / len(confidences)) if confidences else 0.0
    return _text_signal_score(normalized) + (avg_conf * 40.0)


def _prepare_ocr_variants(image: Image.Image) -> list[Image.Image]:
    gray = ImageOps.grayscale(image)
    base = ImageOps.autocontrast(gray)
    if min(base.size) < 1200:
        base = base.resize((base.width * 2, base.height * 2))
    sharpened = base.filter(ImageFilter.SHARPEN)
    binary = sharpened.point(lambda px: 255 if px >= 180 else 0)
    return [base, sharpened, binary]


def _easyocr_best_text(image_bytes: bytes) -> str:
    import easyocr

    _trace("easyocr_worker:before_reader")
    reader = easyocr.Reader(["es", "en"], gpu=False)
    _trace("easyocr_worker:after_reader")

    base_img = Image.open(io.BytesIO(image_bytes))
    base_img = ImageOps.exif_transpose(base_img).convert("RGB")

    candidates: list[tuple[float, str]] = []
    for angle in (0, 90, 180, 270):
        rotated = base_img.rotate(angle, expand=True) if angle else base_img.copy()
        for prepared in _prepare_ocr_variants(rotated):
            _trace(f"easyocr_worker:readtext angle={angle}")
            results = reader.readtext(np.array(prepared), detail=1)
            texts = [str(item[1]).strip() for item in results if len(item) >= 2 and str(item[1]).strip()]
            confidences = [float(item[2]) for item in results if len(item) >= 3 and isinstance(item[2], (int, float))]
            page_text = "\n".join(texts).strip()
            candidates.append((_score_ocr_candidate(page_text, confidences), page_text))

    best_score, best_text = max(candidates, key=lambda item: item[0], default=(0.0, ""))
    return best_text if best_score > 0 else ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Worker aislado por página para EasyOCR.")
    parser.add_argument("--input-image", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    image_path = Path(args.input_image)
    output_path = Path(args.output_json)
    _trace("easyocr_worker:start")
    text = _easyocr_best_text(image_path.read_bytes())
    _trace(f"easyocr_worker:done text_len={len(text)}")
    output_path.write_text(json.dumps({"text": text}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
