from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_text_model: str = Field(default="llama3.2:3b")
    ollama_vision_model: str = Field(default="qwen2.5vl:7b")
    ollama_classifier_model: str = Field(default="llama3.2:3b")
    ollama_embedding_model: str = Field(default="nomic-embed-text")
    ollama_timeout_s: int = Field(default=600)
    ollama_text_num_predict: int = Field(default=256)
    ollama_vision_num_predict: int = Field(default=384)
    ollama_classifier_num_predict: int = Field(default=64)
    rag_persist_dir: str = Field(default=".chroma")


settings = Settings()
