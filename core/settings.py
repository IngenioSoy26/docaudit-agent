from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_text_model: str = Field(default="mistral:7b-instruct")
    ollama_vision_model: str = Field(default="qwen2.5vl:7b")


settings = Settings()
