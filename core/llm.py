"""
Factory de clientes LLM (LangChain + Ollama).

Centraliza la creación de modelos para:
- extracción (formato JSON)
- clasificación
"""

from typing import Any

from core.settings import settings


def _get_llm(
    model: str,
    *,
    num_predict: int | None = None,
    format: str | None = None,
) -> "_LocalOllamaChat":
    """Crea un cliente HTTP ligero para Ollama con interfaz `.invoke()`.

    Args:
        model: Nombre del modelo en Ollama (p.ej. "llama3.2:3b").
        num_predict: Límite aproximado de tokens a generar.
        format: Si se define, fuerza formato (p.ej. "json").

    Returns:
        Cliente compatible con el uso actual del proyecto.
    """
    return _LocalOllamaChat(model=model, num_predict=num_predict, format=format)


class _SimpleResponse:
    def __init__(self, content: str):
        self.content = content


class _LocalOllamaChat:
    def __init__(self, *, model: str, num_predict: int | None = None, format: str | None = None):
        self._model = model
        self._num_predict = num_predict
        self._format = format

    def invoke(self, prompt: str, stream: bool = False) -> _SimpleResponse:
        import requests

        url = settings.ollama_base_url.rstrip("/") + "/api/chat"
        payload: dict[str, Any] = {
            "model": self._model,
            "stream": bool(stream),
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0},
        }
        if self._num_predict is not None:
            payload["options"]["num_predict"] = self._num_predict
        if self._format:
            payload["format"] = self._format

        resp = requests.post(url, json=payload, timeout=settings.ollama_timeout_s)
        resp.raise_for_status()
        data = resp.json()
        message = data.get("message") or {}
        content = message.get("content")
        return _SimpleResponse(content if isinstance(content, str) else str(content))


class _OpenAIChat:
    def __init__(self, *, model: str):
        self._model = model

    def invoke(self, prompt: str, stream: bool = False) -> _SimpleResponse:
        import requests

        api_key = settings.openai_api_key
        if not api_key:
            raise RuntimeError("openai_api_key no está configurada (OPENAI_API_KEY).")

        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": self._model,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = requests.post(url, json=payload, headers=headers, timeout=settings.ollama_timeout_s)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        msg = (choices[0] or {}).get("message") or {} if choices else {}
        content = msg.get("content")
        return _SimpleResponse(content if isinstance(content, str) else str(content))


def get_text_llm() -> _LocalOllamaChat | _OpenAIChat:
    """Devuelve el LLM para extracción de campos (formato JSON)."""
    if (settings.llm_backend or "local").lower() in {"gpt4mini", "gpt-4o-mini"}:
        return _OpenAIChat(model="gpt-4o-mini")  # type: ignore[return-value]
    return _get_llm(
        settings.ollama_text_model,
        num_predict=settings.ollama_text_num_predict,
        format="json",
    )


def get_classifier_llm() -> _LocalOllamaChat | _OpenAIChat:
    """Devuelve el LLM para clasificación de caso de uso."""
    if (settings.llm_backend or "local").lower() in {"gpt4mini", "gpt-4o-mini"}:
        return _OpenAIChat(model="gpt-4o-mini")  # type: ignore[return-value]
    return _get_llm(
        settings.ollama_classifier_model,
        num_predict=settings.ollama_classifier_num_predict,
    )
