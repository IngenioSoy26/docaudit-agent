from __future__ import annotations

import io
from typing import Any

from pypdf import PdfReader


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> dict[str, Any]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    texts: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        page_text = page_text.strip()
        if page_text:
            texts.append(page_text)
    text = "\n\n".join(texts).strip()
    return {"text": text, "pages": len(reader.pages)}
