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
    ocr_backend: str = Field(default="easyocr")
    ocr_backend_native_pdf: str = Field(default="easyocr")
    ocr_backend_scanned_blurry_pdf: str = Field(default="rapidocr")
    ocr_backend_image_photo: str = Field(default="rapidocr")
    ocr_backend_image_handwritten: str = Field(default="rapidocr")
    enable_easyocr: bool = Field(default=True)
    enable_easyocr_page_subprocess: bool = Field(default=True)
    easyocr_page_timeout_s: int = Field(default=120)
    easyocr_page_max_dim: int = Field(default=256)
    easyocr_page_jpeg_quality: int = Field(default=60)
    enable_paddleocr: bool = Field(default=True)
    enable_paddleocr_page_subprocess: bool = Field(default=True)
    paddleocr_page_timeout_s: int = Field(default=120)
    paddleocr_page_max_dim: int = Field(default=384)
    paddleocr_page_jpeg_quality: int = Field(default=70)
    enable_rapidocr: bool = Field(default=True)
    enable_rapidocr_page_subprocess: bool = Field(default=True)
    rapidocr_page_timeout_s: int = Field(default=120)
    rapidocr_page_max_dim: int = Field(default=512)
    rapidocr_page_jpeg_quality: int = Field(default=75)
    enable_vision_page_subprocess: bool = Field(default=True)
    vision_page_timeout_s: int = Field(default=180)
    vision_page_max_dim: int = Field(default=256)
    vision_page_jpeg_quality: int = Field(default=60)
    scanned_pdf_text_mode: str = Field(default="auto")
    enable_scanned_pdf_vision_fallback: bool = Field(default=True)
    enable_scanned_pdf_ocr_fallback: bool = Field(default=True)
    enable_scanned_pdf_ocr_subprocess: bool = Field(default=True)
    scanned_pdf_ocr_timeout_s: int = Field(default=300)
    enable_extract_llm_subprocess: bool = Field(default=True)
    extract_llm_timeout_s: int = Field(default=240)
    ollama_classifier_num_predict: int = Field(default=64)
    rag_persist_dir: str = Field(default=".chroma")

    enable_pii_redaction: bool = Field(default=False)


settings = Settings()
