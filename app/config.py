
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки приложения. Читаются из переменных окружения и файла .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Выбор провайдера: DeepSeek или Ollama (что удобнее)
    llm_provider: Literal["deepseek", "ollama"] = "deepseek"

    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: str = "not-set"
    llm_model: str = "deepseek-flash"

    llm_timeout_seconds: float = 90.0
    llm_max_retries: int = 1

    structured_output_mode: Literal[
        "json_schema", "json_object", "prompt"
    ] = "json_object"


@lru_cache
def get_settings() -> Settings:
    return Settings()
