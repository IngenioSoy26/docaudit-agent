from __future__ import annotations

import base64
from typing import Any

import requests

from core.settings import settings


def chat_with_images(prompt: str, images: list[bytes], model: str | None = None) -> str:
    url = settings.ollama_base_url.rstrip("/") + "/api/chat"
    b64_images = [base64.b64encode(b).decode("ascii") for b in images]
    payload: dict[str, Any] = {
        "model": model or settings.ollama_vision_model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": b64_images,
            }
        ],
    }
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    message = data.get("message") or {}
    content = message.get("content")
    return content if isinstance(content, str) else str(content)
