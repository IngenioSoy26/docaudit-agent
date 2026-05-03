from langchain_ollama import ChatOllama

from core.settings import settings


def get_text_llm() -> ChatOllama:
    return ChatOllama(
        model=settings.ollama_text_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )
