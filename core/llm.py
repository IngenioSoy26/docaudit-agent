"""
Factory de clientes LLM (LangChain + Ollama).

Centraliza la creación de modelos para:
- extracción (formato JSON)
- clasificación
"""

from langchain_ollama import ChatOllama

from core.settings import settings


def _get_llm(
    model: str,
    *,
    num_predict: int | None = None,
    format: str | None = None,
) -> ChatOllama:
    return ChatOllama(
        model=model,
        base_url=settings.ollama_base_url,
        temperature=0,
        num_predict=num_predict,
        format=format,
        client_kwargs={"timeout": settings.ollama_timeout_s},
    )


def get_text_llm() -> ChatOllama:
    return _get_llm(
        settings.ollama_text_model,
        num_predict=settings.ollama_text_num_predict,
        format="json",
    )


def get_classifier_llm() -> ChatOllama:
    return _get_llm(settings.ollama_classifier_model, num_predict=settings.ollama_classifier_num_predict)
