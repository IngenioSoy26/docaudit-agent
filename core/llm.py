from langchain_ollama import ChatOllama

from core.settings import settings


def _get_llm(model: str) -> ChatOllama:
    return ChatOllama(
        model=model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )


def get_text_llm() -> ChatOllama:
    return _get_llm(settings.ollama_text_model)


def get_classifier_llm() -> ChatOllama:
    return _get_llm(settings.ollama_classifier_model)
