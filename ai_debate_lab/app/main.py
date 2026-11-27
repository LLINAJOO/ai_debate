import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from app.config import get_settings
from app.llm import LLMClient, LLMError, LLMValidationError
from app.orchestrator import debate_events, run_debate
from app.roles import DISCUSSION_ORDER, JUDGE, ROLES
from app.schemas import DebateRequest, DebateResponse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ai_debate_lab")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llm = LLMClient(get_settings())
    yield


app = FastAPI(
    title="AI Debate Lab",
    description="Многоагентное обсуждение идеи: Исследователь, Критик, Оппонент, "
                "Модератор и итоговый отчёт Судьи.",
    version="0.3.0",
    lifespan=lifespan,
)


def get_llm(request: Request) -> LLMClient:
    return request.app.state.llm


def _build_response(req: DebateRequest, llm: LLMClient, discussion, final, usage, calls,
                    started: float) -> DebateResponse:
    return DebateResponse(
        task=req.task,
        model=llm.model,
        temperature=req.temperature,
        rounds=req.rounds,
        discussion=discussion,
        final_report=final,
        report_by=JUDGE["title"],
        llm_calls=calls,
        usage=usage,
        latency_seconds=round(time.perf_counter() - started, 3),
    )


def _error_text(e: Exception) -> str:
    if isinstance(e, LLMValidationError):
        return f"Модель вернула JSON, не соответствующий схеме ({e})"
    return f"Ошибка LLM-провайдера ({e})"


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/roles")
async def roles():
    items = [{"id": rid, "title": ROLES[rid]["title"], "description": ROLES[rid]["description"]}
             for rid in DISCUSSION_ORDER]
    items.append({"id": "judge", "title": JUDGE["title"], "description": JUDGE["description"]})
    return items


@app.get("/health")
async def health(llm: LLMClient = Depends(get_llm)):
    return {"status": "ok", "model": llm.model}


@app.post("/debate", response_model=DebateResponse)
async def debate(req: DebateRequest, llm: LLMClient = Depends(get_llm)):
    """Обычный запрос: ждёт конца обсуждения и возвращает весь результат одним JSON."""
    started = time.perf_counter()
    try:
        discussion, final, usage, calls = await run_debate(
            llm, req.task, req.temperature, req.rounds
        )
    except (LLMError, LLMValidationError) as e:
        raise HTTPException(status_code=502, detail=_error_text(e))
    return _build_response(req, llm, discussion, final, usage, calls, started)


def _line(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


@app.post("/debate/stream")
async def debate_stream(req: DebateRequest, llm: LLMClient = Depends(get_llm)):
    """То же обсуждение, но с потоком событий (по одному JSON на строку).

    События: start — агент начал работу; entry — агент закончил;
    result — итоговый ответ (как у /debate); error — ошибка.
    """
    started = time.perf_counter()

    async def gen():
        try:
            async for ev in debate_events(llm, req.task, req.temperature, req.rounds):
                if ev["type"] == "start":
                    yield _line(ev)
                elif ev["type"] == "entry":
                    yield _line({"type": "entry", "entry": ev["entry"].model_dump()})
                else:
                    resp = _build_response(req, llm, ev["discussion"], ev["final"],
                                           ev["usage"], ev["calls"], started)
                    yield _line({"type": "result", "data": resp.model_dump()})
        except (LLMError, LLMValidationError) as e:
            yield _line({"type": "error", "detail": _error_text(e)})
        except Exception:
            log.exception("Непредвиденная ошибка в обсуждении")
            yield _line({"type": "error", "detail": "Внутренняя ошибка сервера"})

    return StreamingResponse(
        gen(), media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
