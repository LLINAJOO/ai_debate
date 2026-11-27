import json
from collections.abc import AsyncIterator

from app.llm import LLMClient, LLMError, LLMValidationError
from app.roles import DISCUSSION_ORDER, JUDGE, JUDGE_ID, ROLES, system_prompt
from app.schemas import DiscussionEntry, JudgeOutput, Usage


def _transcript(discussion: list[DiscussionEntry]) -> str:
    if not discussion:
        return "(пока никто не высказывался)"
    parts = []
    for e in discussion:
        body = json.dumps(e.content, ensure_ascii=False, indent=2)
        parts.append(f"### {e.role_title} (раунд {e.round})\n{body}")
    return "\n\n".join(parts)


def _user_message(task: str, discussion: list[DiscussionEntry], role: dict,
                  rnd: int | None = None, rounds: int | None = None) -> str:
    msg = f"Задача пользователя:\n{task}\n\nХод обсуждения:\n{_transcript(discussion)}\n\n"
    if rnd and rnd > 1:
        msg += (f"Это раунд {rnd} из {rounds}. Учти замечания предыдущего раунда "
                "и уточни или измени свою позицию.\n")
    return msg + f"Твоя задача: {role['instruction']}"


def _add(total: Usage, u: Usage) -> None:
    total.prompt_tokens += u.prompt_tokens or 0
    total.completion_tokens += u.completion_tokens or 0
    total.total_tokens += u.total_tokens or 0


async def _ask(llm: LLMClient, role: dict, user: str, temperature: float):
    try:
        return await llm.generate_structured(
            system=system_prompt(role), user=user,
            schema_model=role["schema"], temperature=temperature,
        )
    except LLMError as e:
        raise LLMError(f"{role['title']}: {e}") from e
    except LLMValidationError as e:
        raise LLMValidationError(f"{role['title']}: {e}") from e


async def debate_events(
    llm: LLMClient, task: str, temperature: float, rounds: int
) -> AsyncIterator[dict]:
    """Исследователь → Критик → Оппонент → Модератор (× rounds) → Судья.

    Выдаёт события: start (агент начал), entry (агент закончил), final (всё готово).
    """
    discussion: list[DiscussionEntry] = []
    total = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    calls = 0

    for rnd in range(1, rounds + 1):
        for role_id in DISCUSSION_ORDER:
            role = ROLES[role_id]
            yield {"type": "start", "role": role_id, "title": role["title"],
                   "round": rnd, "rounds": rounds}
            out, usage = await _ask(
                llm, role, _user_message(task, discussion, role, rnd, rounds), temperature
            )
            _add(total, usage)
            calls += 1
            entry = DiscussionEntry(
                role=role_id, role_title=role["title"], round=rnd, content=out.model_dump()
            )
            discussion.append(entry)
            yield {"type": "entry", "entry": entry}

    yield {"type": "start", "role": JUDGE_ID, "title": JUDGE["title"],
           "round": None, "rounds": rounds}
    final, usage = await _ask(llm, JUDGE, _user_message(task, discussion, JUDGE), temperature)
    _add(total, usage)
    calls += 1
    assert isinstance(final, JudgeOutput)
    yield {"type": "final", "discussion": discussion, "final": final,
           "usage": total, "calls": calls}


async def run_debate(llm: LLMClient, task: str, temperature: float, rounds: int):
    """Запускает всё обсуждение целиком и возвращает итог (без потока событий)."""
    result = None
    async for ev in debate_events(llm, task, temperature, rounds):
        if ev["type"] == "final":
            result = (ev["discussion"], ev["final"], ev["usage"], ev["calls"])
    return result
