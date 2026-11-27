from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки приложения."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str = "not-set"
    llm_model: str = "openai/gpt-4o-mini"

    llm_project: str | None = None
    llm_timeout_seconds: float = 60.0

    llm_max_retries: int = 1
    structured_output_mode: Literal["json_schema", "json_object", "prompt"] = "json_schema"


@lru_cache
def get_settings() -> Settings:
    return Settings()
