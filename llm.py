import json
import logging
from typing import TypeVar

from openai import APIError, AsyncOpenAI
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.schemas import Usage

log = logging.getLogger("ai_debate_lab.llm")

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Ошибка обращения к LLM-провайдеру (сеть, авторизация, лимиты...)."""


class LLMValidationError(Exception):
    """Модель вернула ответ, не соответствующий схеме, даже после повторов."""


def _extract_json(text: str) -> str:
    """Достаёт JSON из ответа, если модель обернула его в ```json ... ``` или добавила текст."""
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


class LLMClient:
    """Тонкая обёртка над OpenAI-совместимым API со structured output."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.llm_model
        self._client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout_seconds,
            project=settings.llm_project,
        )

    def _response_format(self, schema_model: type[BaseModel]) -> dict | None:
        if self.settings.structured_output_mode == "json_schema":
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_model.__name__,
                    "strict": True,
                    "schema": schema_model.model_json_schema(),
                },
            }
        if self.settings.structured_output_mode == "json_object":
            return {"type": "json_object"}
        return None  # режим "prompt": формат не навязываем, схема только в промпте

    async def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema_model: type[T],
        temperature: float,
    ) -> tuple[T, Usage]:
        """Вызывает LLM и возвращает проверенный Pydantic-объект + расход токенов."""
        if self.settings.structured_output_mode != "json_schema":
            # В этих режимах схему провайдер не навязывает — описываем её в промпте.
            system += (
                "\n\nВерни ТОЛЬКО валидный JSON-объект, строго соответствующий схеме:\n"
                + json.dumps(schema_model.model_json_schema(), ensure_ascii=False)
            )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        total = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        last_error: Exception | None = None

        for attempt in range(self.settings.llm_max_retries + 1):
            try:
                kwargs: dict = dict(
                    model=self.model, messages=messages, temperature=temperature
                )
                response_format = self._response_format(schema_model)
                if response_format is not None:
                    kwargs["response_format"] = response_format
                resp = await self._client.chat.completions.create(**kwargs)
            except APIError as e:
                raise LLMError(f"{type(e).__name__}: {e}") from e

            if resp.usage:
                total.prompt_tokens += resp.usage.prompt_tokens or 0
                total.completion_tokens += resp.usage.completion_tokens or 0
                total.total_tokens += resp.usage.total_tokens or 0

            content = (resp.choices[0].message.content or "") if resp.choices else ""
            try:
                return schema_model.model_validate_json(_extract_json(content)), total
            except ValidationError as e:
                last_error = e
                log.warning("Ответ не прошёл валидацию (попытка %d): %s", attempt + 1, e)

        raise LLMValidationError(str(last_error))
