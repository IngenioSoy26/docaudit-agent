"""
Configuración central (Pydantic Settings).

Las variables se leen desde entorno o desde un archivo .env en la raíz del repositorio.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Parámetros de configuración del sistema.

    Las variables pueden definirse como variables de entorno o en un archivo `.env`.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_backend: str = Field(default="local")
    openai_api_key: str | None = Field(default=None)

    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_text_model: str = Field(default="llama3.2:3b")
    ollama_vision_model: str = Field(default="qwen2.5vl:7b")
    ollama_classifier_model: str = Field(default="llama3.2:3b")
    ollama_embedding_model: str = Field(default="nomic-embed-text")
    ollama_timeout_s: int = Field(default=1200)
    ollama_text_num_predict: int = Field(default=256)
    ollama_vision_num_predict: int = Field(default=1024)
    ollama_vision_max_dim: int = Field(default=1024)
    ollama_vision_jpeg_quality: int = Field(default=70)
    ollama_classifier_num_predict: int = Field(default=64)
    rag_persist_dir: str = Field(default=".chroma")

    enable_pii_redaction: bool = Field(default=False)


settings = Settings()
