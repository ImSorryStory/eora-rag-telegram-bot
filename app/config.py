from __future__ import annotations
import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    OPENAI_API_KEY: str = Field(..., description="OpenAI API key")
    TELEGRAM_BOT_TOKEN: str = Field(..., description="Telegram bot token")

    ALLOWED_DOMAINS: str = Field("eora.ru")
    DATA_DIR: str = Field("./data")
    INDEX_PATH: str = Field("./data/index.faiss")
    CHUNKS_PATH: str = Field("./data/chunks.jsonl")

    # совет: сделайте модель обязательной через .env, чтобы избежать 404 по дефолтной модели
    GEN_MODEL: str = Field(..., description="Chat model name, e.g. gpt-4o-mini")
    EMBED_MODEL: str = Field("text-embedding-3-large")
    TOP_K: int = Field(8)
    ATTACH_TOP_N: int = Field(3)
    TEMPERATURE: float = Field(0.2)
    MAX_OUTPUT_TOKENS: int = Field(800)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
os.makedirs(settings.DATA_DIR, exist_ok=True)