from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

import requests


def _trace(message: str) -> None:
    trace_path = os.environ.get("DOCAUDIT_OCR_TRACE_PATH", "").strip()
    if not trace_path:
        return
    try:
        with Path(trace_path).open("a", encoding="utf-8") as fh:
            fh.write(message + "\n")
    except Exception:
        pass


def _chat_with_images(prompt: str, image_bytes: bytes) -> str:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.environ.get("OLLAMA_VISION_MODEL", "qwen2.5vl:7b")
    timeout_s = int(os.environ.get("OLLAMA_TIMEOUT_S", "1200") or "1200")
    num_predict = int(os.environ.get("OLLAMA_VISION_NUM_PREDICT", "256") or "256")
    url = base_url + "/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "keep_alive": "10m",
        "options": {"num_predict": num_predict, "temperature": 0},
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [base64.b64encode(image_bytes).decode("ascii")],
            }
        ],
    }
    _trace("vision_worker:before_http")
    resp = requests.post(url, json=payload, timeout=timeout_s)
    _trace(f"vision_worker:after_http status={resp.status_code}")
    resp.raise_for_status()
    data = resp.json()
    message = data.get("message") or {}
    content = message.get("content")
    return content if isinstance(content, str) else str(content)


def main() -> int:
    parser = argparse.ArgumentParser(description="Worker aislado por página para VLM.")
    parser.add_argument("--input-image", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    image_path = Path(args.input_image)
    output_path = Path(args.output_json)
    _trace("vision_worker:start")
    content = _chat_with_images(args.prompt, image_path.read_bytes()).strip()
    _trace(f"vision_worker:done text_len={len(content)}")
    output_path.write_text(json.dumps({"text": content}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
