from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------- Запрос ----------

class DebateRequest(BaseModel):
    task: str = Field(
        ..., min_length=5, max_length=4000,
        description="Идея, задача или вопрос для обсуждения",
        examples=["Как небольшому интернет-магазину организовать доставку: "
                  "создать собственную курьерскую службу или использовать стороннего оператора?"],
    )
    temperature: float = Field(
        0.3, ge=0.0, le=2.0,
        description="Температура генерации LLM (0 — детерминированно, выше — разнообразнее)",
    )
    rounds: int = Field(
        1, ge=1, le=3,
        description="Число раундов обсуждения. Каждый раунд = 4 обращения к модели, плюс 1 на судью.",
    )


# ---------- Структуры ответов агентов ----------
# Эти модели превращаются в JSON Schema для structured output.
# extra="forbid" -> additionalProperties: false (требование strict-режима).
# Значений по умолчанию нет намеренно: в strict-режиме все поля обязательны.

class _AgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _no_blank_strings(self):
        for name, value in self:
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"поле {name} не должно быть пустым")
        return self


def _need_items(v: list[str]) -> list[str]:
    if not any(item.strip() for item in v):
        raise ValueError("список не должен быть пустым")
    return v


class ResearcherOutput(_AgentOutput):
    position: str = Field(description="Предлагаемое решение (1-3 предложения)")
    arguments: list[str] = Field(description="Аргументы в пользу решения")
    constraints: list[str] = Field(description="Ограничения и условия задачи, которые влияют на решение")

    _v = field_validator("arguments")(_need_items)


class CriticOutput(_AgentOutput):
    objections: list[str] = Field(description="Конкретные возражения против предложенного решения")
    risks: list[str] = Field(description="Практические риски")
    weak_assumptions: list[str] = Field(description="Необоснованные допущения и логические слабые места")

    _v = field_validator("objections")(_need_items)


class OpponentOutput(_AgentOutput):
    alternative: str = Field(description="Альтернативный вариант решения")
    comparison: str = Field(description="Сравнение альтернативы с исходным решением")
    overlooked_opportunities: list[str] = Field(description="Возможности, которые могли быть упущены")


class MediatorOutput(_AgentOutput):
    key_factors: list[str] = Field(description="Ключевые факторы, по которым нужно сравнивать варианты")
    agreements: list[str] = Field(description="В чём участники согласны")
    disagreements: list[str] = Field(description="В чём участники расходятся")
    open_questions: list[str] = Field(description="Нерешённые вопросы")


class JudgeOutput(_AgentOutput):
    final_decision: str = Field(description="Итоговое решение или условная рекомендация")
    reasoning: str = Field(description="Почему выбрано именно это: как оценены аргументы сторон")
    deciding_factors: list[str] = Field(description="Решающие факторы")
    limitations: list[str] = Field(description="Ограничения и оставшиеся неопределённости")
    data_needed: list[str] = Field(description="Какие данные нужны для окончательного выбора")

    _v = field_validator("deciding_factors")(_need_items)


# ---------- Ответ API ----------

class Usage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class DiscussionEntry(BaseModel):
    role: str
    role_title: str
    round: int
    content: dict[str, Any]


class DebateResponse(BaseModel):
    task: str
    model: str
    temperature: float
    rounds: int
    discussion: list[DiscussionEntry]
    final_report: JudgeOutput
    report_by: str
    llm_calls: int
    usage: Usage
    latency_seconds: float
