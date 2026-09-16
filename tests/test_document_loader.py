"""Tests del cargador documental y sus fallbacks OCR."""

import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.document_loader as loader
from PIL import Image


class _FakePage:
    def __init__(self, text=""):
        self._text = text

    def extract_text(self):
        return self._text


class _FakePdfReader:
    def __init__(self, _stream):
        self.pages = [_FakePage("")]


def test_extract_text_from_scanned_pdf_bytes_uses_vision_only_on_bad_pages(monkeypatch):
    class _FakeImage:
        def __init__(self, page, data):
            self.page = page
            self.data = data

        def __getitem__(self, key):
            return getattr(self, key)

    class _PdfReaderTwoPages:
        def __init__(self, _stream):
            self.pages = [_FakePage(""), _FakePage("")]

    monkeypatch.setattr(loader, "PdfReader", _PdfReaderTwoPages)
    monkeypatch.setattr(
        loader,
        "extract_images_from_pdf_bytes",
        lambda _pdf_bytes: [
            {"page": 1, "data": b"good"},
            {"page": 2, "data": b"bad"},
        ],
    )
    monkeypatch.setattr(loader, "_get_easyocr_reader", lambda: object())
    monkeypatch.setattr(
        loader,
        "_easyocr_best_text",
        lambda image_bytes, reader=None: "Factura valida con fecha 2025-03-08" if image_bytes == b"good" else "1 0 5",
    )
    calls = []

    def _fake_vision(_prompt, images):
        calls.append(images[0])
        return "Texto recuperado por vision"

    monkeypatch.setattr(loader, "chat_with_images", _fake_vision)
    monkeypatch.setattr(loader.settings, "enable_scanned_pdf_ocr_subprocess", False)
    monkeypatch.setattr(loader.settings, "enable_easyocr_page_subprocess", False)

    result = loader.extract_text_from_scanned_pdf_bytes(b"%PDF-fake")

    assert result["method"] == "easyocr+ollama_vision"
    assert result["page_texts"][0] == "Factura valida con fecha 2025-03-08"
    assert result["page_texts"][1] == "Texto recuperado por vision"
    assert calls == [b"bad"]


def test_extract_text_from_pdf_bytes_falls_back_to_scanned_ocr(monkeypatch):
    monkeypatch.setattr(loader, "PdfReader", _FakePdfReader)
    monkeypatch.setattr(loader.settings, "enable_scanned_pdf_ocr_subprocess", False)
    monkeypatch.setattr(loader.settings, "enable_easyocr_page_subprocess", False)
    monkeypatch.setattr(
        loader,
        "extract_text_from_scanned_pdf_bytes",
        lambda _pdf_bytes: {
            "text": "Documento escaneado con nombre, fecha 2025-03-08 e importe 125000",
            "pages": 1,
            "page_texts": ["Documento escaneado con nombre, fecha 2025-03-08 e importe 125000"],
            "method": "easyocr",
        },
    )

    result = loader.extract_text_from_pdf_bytes(b"%PDF-fake")

    assert result["method"] == "easyocr"
    assert "2025-03-08" in result["text"]


def test_extract_text_from_image_bytes_uses_vision_when_easyocr_is_too_sparse(monkeypatch):
    monkeypatch.setattr(loader, "_downscale_image_bytes", lambda image_bytes: image_bytes)
    monkeypatch.setattr(loader, "_easyocr_best_text", lambda _image_bytes: "1 0 5")
    monkeypatch.setattr(
        loader,
        "chat_with_images",
        lambda _prompt, _images: "Nombre Juan Perez DNI 12345678Z Fecha 2025-03-08",
    )

    result = loader.extract_text_from_image_bytes(b"fake-image")

    assert result["method"] == "ollama_vision"
    assert "Juan Perez" in result["text"]


def test_extraction_policy_for_doc_uses_paddleocr_on_blurry_and_auto_on_native(monkeypatch):
    monkeypatch.setattr(loader.settings, "ocr_backend", "easyocr")
    monkeypatch.setattr(loader.settings, "scanned_pdf_text_mode", "auto")
    monkeypatch.setattr(loader.settings, "ocr_backend_scanned_blurry_pdf", "paddleocr")

    with loader.extraction_policy_for_doc("credito_hipotecario_scanned_blurry_pdf_conforme_011.pdf"):
        assert loader.settings.ocr_backend == "paddleocr"
        assert loader.settings.scanned_pdf_text_mode == "ocr_only"

    assert loader.settings.ocr_backend == "easyocr"
    assert loader.settings.scanned_pdf_text_mode == "auto"


def test_extract_text_from_image_bytes_can_use_paddleocr(monkeypatch):
    monkeypatch.setattr(loader, "_downscale_image_bytes", lambda image_bytes: image_bytes)
    monkeypatch.setattr(loader.settings, "ocr_backend", "paddleocr")
    monkeypatch.setattr(loader.settings, "enable_paddleocr", True)
    monkeypatch.setattr(loader, "_paddleocr_best_text", lambda _image_bytes, reader=None: "Texto PaddleOCR con fecha 2025-03-08")

    result = loader.extract_text_from_image_bytes(b"fake-image")

    assert result["method"] == "paddleocr"
    assert "2025-03-08" in result["text"]


def test_extract_text_from_image_bytes_can_use_rapidocr(monkeypatch):
    monkeypatch.setattr(loader, "_downscale_image_bytes", lambda image_bytes: image_bytes)
    monkeypatch.setattr(loader.settings, "ocr_backend", "rapidocr")
    monkeypatch.setattr(loader.settings, "enable_rapidocr", True)
    monkeypatch.setattr(loader, "_rapidocr_best_text", lambda _image_bytes, reader=None: "Texto RapidOCR con DNI 86473212N")

    result = loader.extract_text_from_image_bytes(b"fake-image")

    assert result["method"] == "rapidocr"
    assert "86473212N" in result["text"]


def test_rapidocr_best_text_uses_rgb_rotations_without_easyocr_variants(monkeypatch):
    img = Image.new("RGB", (16, 12), color="white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    calls = []

    class _FakeRapidOCR:
        def __call__(self, arr):
            calls.append(tuple(arr.shape))
            return [([[0, 0], [1, 0], [1, 1], [0, 1]], "Texto RapidOCR estable", 0.95)], None

    def _unexpected_prepare(_image):
        raise AssertionError("RapidOCR no debe reutilizar variantes agresivas de EasyOCR")

    monkeypatch.setattr(loader, "_prepare_ocr_variants", _unexpected_prepare)

    text = loader._rapidocr_best_text(buf.getvalue(), reader=_FakeRapidOCR())

    assert text == "Texto RapidOCR estable"
    assert calls == [(12, 16, 3)]


def test_extract_text_from_scanned_pdf_bytes_respects_rapidocr_page_subprocess_flag(monkeypatch):
    monkeypatch.setattr(loader, "PdfReader", _FakePdfReader)
    monkeypatch.setattr(loader, "extract_images_from_pdf_bytes", lambda _pdf_bytes: [{"page": 1, "data": b"x"}])
    monkeypatch.setattr(loader.settings, "ocr_backend", "rapidocr")
    monkeypatch.setattr(loader.settings, "enable_easyocr_page_subprocess", False)
    monkeypatch.setattr(loader.settings, "enable_rapidocr_page_subprocess", True)
    monkeypatch.setattr(loader.settings, "enable_scanned_pdf_ocr_subprocess", True)
    monkeypatch.setattr(loader.settings, "enable_scanned_pdf_vision_fallback", False)
    monkeypatch.setattr(loader, "_run_ocr_page_subprocess", lambda _image_bytes, backend: "Texto rapido con fecha 2025-03-08" if backend == "rapidocr" else "")
    monkeypatch.setattr(
        loader,
        "_run_scanned_pdf_ocr_subprocess",
        lambda _pdf_bytes: {
            "text": "",
            "pages": 1,
            "page_texts": [""],
            "images": 1,
            "method": "ocr_subprocess_failed",
        },
    )

    result = loader.extract_text_from_scanned_pdf_bytes(b"%PDF-fake")

    assert result["method"] == "rapidocr"
    assert "2025-03-08" in result["text"]


def test_extract_text_from_scanned_pdf_bytes_uses_subprocess_result(monkeypatch):
    monkeypatch.setattr(loader, "extract_images_from_pdf_bytes", lambda _pdf_bytes: [{"page": 1, "data": b"x"}])
    monkeypatch.setattr(loader, "PdfReader", _FakePdfReader)
    monkeypatch.setattr(loader.settings, "enable_scanned_pdf_ocr_subprocess", True)
    monkeypatch.setattr(loader.settings, "enable_easyocr_page_subprocess", False)

    def _fake_run(cmd, cwd=None, env=None, stdout=None, stderr=None, timeout=None):
        output_arg = Path(cmd[cmd.index("--output-json") + 1])
        output_arg.write_text(
            json.dumps(
                {
                    "text": "Texto OCR recuperado",
                    "pages": 1,
                    "page_texts": ["Texto OCR recuperado"],
                    "images": 1,
                    "method": "easyocr",
                }
            ),
            encoding="utf-8",
        )

        class _Completed:
            returncode = 0

        return _Completed()

    monkeypatch.setattr(loader.subprocess, "run", _fake_run)

    result = loader.extract_text_from_scanned_pdf_bytes(b"%PDF-fake")

    assert result["method"] == "easyocr"
    assert result["text"] == "Texto OCR recuperado"


def test_extract_text_from_scanned_pdf_bytes_returns_timeout_marker_when_worker_hangs(monkeypatch):
    monkeypatch.setattr(loader, "extract_images_from_pdf_bytes", lambda _pdf_bytes: [{"page": 1, "data": b"x"}])
    monkeypatch.setattr(loader, "PdfReader", _FakePdfReader)
    monkeypatch.setattr(loader.settings, "enable_scanned_pdf_ocr_subprocess", True)
    monkeypatch.setattr(loader.settings, "enable_easyocr_page_subprocess", False)

    def _fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="python worker", timeout=30)

    monkeypatch.setattr(loader.subprocess, "run", _fake_run)

    result = loader.extract_text_from_scanned_pdf_bytes(b"%PDF-fake")

    assert result["method"] == "ocr_subprocess_timeout"
    assert result["text"] == ""
