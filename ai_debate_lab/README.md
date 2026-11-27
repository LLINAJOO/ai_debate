# AI Debate Lab

Веб-приложение на **Python + FastAPI**: несколько ИИ-агентов с разными ролями по очереди
обсуждают вашу идею, а **Судья** читает всё обсуждение и пишет итоговый отчёт.

**Ход обсуждения:** Исследователь → Критик → Оппонент → Модератор (повторяется `rounds` раз) → Судья.
Все агенты — одна LLM (OpenAI-совместимый API) с разными системными инструкциями.
Ответ каждого агента — валидный JSON по своей схеме (structured output + проверка Pydantic).

## Быстрый старт

1. Установите Python 3.10+ (https://www.python.org/downloads/; на Windows отметьте «Add Python to PATH»).
2. Откройте терминал в папке проекта и создайте окружение:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Получите бесплатный ключ Gemini: https://aistudio.google.com → **Get API key**.
4. Создайте файл настроек и вставьте ключ:
   ```bash
   cp .env.example .env             # Windows: copy .env.example .env
   ```
   Откройте `.env` и замените `ВСТАВЬТЕ_СВОЙ_КЛЮЧ` на ваш ключ.
5. Запустите:
   ```bash
   uvicorn app.main:app --reload
   ```
6. Откройте http://127.0.0.1:8000 — это веб-интерфейс.
   Документация API (Swagger): http://127.0.0.1:8000/docs

> ⚠️ Файл `.env` содержит секретный ключ и добавлен в `.gitignore` — не публикуйте его.

## API

`POST /debate`

```json
{ "task": "Свой курьер или сторонний оператор?", "temperature": 0.3, "rounds": 1 }
```

Один запуск = `rounds × 4 + 1` обращений к модели.

Ответ (сокращённо):

```json
{
  "task": "...", "model": "...", "temperature": 0.3, "rounds": 1,
  "discussion": [
    {"role": "researcher", "role_title": "Исследователь", "round": 1,
     "content": {"position": "...", "arguments": ["..."], "constraints": ["..."]}},
    {"role": "critic", "...": "..."}, {"role": "opponent", "...": "..."}, {"role": "mediator", "...": "..."}
  ],
  "final_report": {"final_decision": "...", "reasoning": "...", "deciding_factors": ["..."],
                   "limitations": ["..."], "data_needed": ["..."]},
  "report_by": "Судья",
  "llm_calls": 5, "usage": {"total_tokens": 0}, "latency_seconds": 0.0
}
```

Ошибки: `422` — некорректный запрос; `502` — сбой провайдера LLM или ответ не по схеме.

## Как это работает

1. `app/schemas.py` — Pydantic-модели запроса, ответов агентов и итогового ответа.
2. `app/roles.py` — роли агентов (инструкции и схемы). Новую роль можно добавить здесь.
3. `app/orchestrator.py` — запускает агентов по очереди и передаёт каждому ход обсуждения.
4. `app/llm.py` — клиент OpenAI-совместимого API: отправляет схему, проверяет ответ, повторяет запрос при ошибке.
5. `app/main.py` — FastAPI: страница `/`, `GET /roles`, `GET /health`, `POST /debate`.
6. `app/static/index.html` — веб-интерфейс.

## Другие модели

Меняются только значения в `.env` (примеры для OpenRouter, Yandex AI Studio и Ollama — в `.env.example`).

## Тесты

```bash
pytest -q
```
Тесты работают без сети и без ключа.

## Планы

Сравнение конфигураций (один агент / два агента / полная дискуссия), набор тестовых задач,
сохранение истории в PostgreSQL.
