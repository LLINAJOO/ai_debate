import json

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm import LLMClient
from app.main import app, get_llm

OUT = {
    "researcher": {"position": "Использовать стороннего оператора.",
                   "arguments": ["Низкие стартовые расходы"], "constraints": ["Малый бюджет"]},
    "critic": {"objections": ["Меньше контроля качества"], "risks": ["Рост тарифов"],
               "weak_assumptions": ["Объём заказов неизвестен"]},
    "opponent": {"alternative": "Гибридная модель", "comparison": "Дешевле при росте",
                 "overlooked_opportunities": ["Собственный курьер в городе"]},
    "mediator": {"key_factors": ["Объём заказов"], "agreements": ["Нужны данные"],
                 "disagreements": ["Контроль vs цена"], "open_questions": ["Какой объём?"]},
    "judge": {"final_decision": "Начать со сторонней службы.", "reasoning": "Критик и оппонент "
              "указали на риски, но данных мало.", "deciding_factors": ["Объём заказов"],
              "limitations": ["Нет данных о тарифах"], "data_needed": ["Число заказов в день"]},
}
ORDER = ["researcher", "critic", "opponent", "mediator"]
TITLES = ["Исследователь", "Критик", "Оппонент", "Модератор"]


class _Msg:
    def __init__(self, c): self.content = c
class _Choice:
    def __init__(self, c): self.message = _Msg(c)
class _Usage:
    prompt_tokens, completion_tokens, total_tokens = 10, 20, 30
class _Resp:
    def __init__(self, c): self.choices, self.usage = [_Choice(c)], _Usage()


class FakeCompletions:
    def __init__(self, contents):
        self.contents, self.calls = list(contents), []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Resp(self.contents.pop(0))


def good_run(rounds=1):
    return [json.dumps(OUT[r], ensure_ascii=False) for _ in range(rounds) for r in ORDER] + \
           [json.dumps(OUT["judge"], ensure_ascii=False)]


def make_client(contents, mode="json_schema", retries=1):
    llm = LLMClient(Settings(llm_api_key="x", structured_output_mode=mode, llm_max_retries=retries))
    fake = FakeCompletions(contents)
    llm._client.chat.completions = fake
    return llm, fake


@pytest.fixture
def client_factory():
    def _make(contents, **kw):
        llm, fake = make_client(contents, **kw)
        app.dependency_overrides[get_llm] = lambda: llm
        return TestClient(app), fake
    yield _make
    app.dependency_overrides.clear()


def test_full_debate_order_and_final_report(client_factory):
    client, fake = client_factory(good_run())
    r = client.post("/debate", json={"task": "Свой курьер или подрядчик?", "temperature": 0.7})
    assert r.status_code == 200
    body = r.json()
    assert [d["role"] for d in body["discussion"]] == ORDER
    assert body["final_report"]["final_decision"] == OUT["judge"]["final_decision"]
    assert body["report_by"] == "Судья"
    assert body["llm_calls"] == 5 and len(fake.calls) == 5
    assert body["usage"]["total_tokens"] == 150
    # температура одинакова для всех обращений
    assert {c["temperature"] for c in fake.calls} == {0.7}
    # судья видит всё обсуждение
    judge_prompt = fake.calls[-1]["messages"][1]["content"]
    for title in TITLES:
        assert title in judge_prompt
    # каждый агент получает свою роль в system-промпте
    for call, title in zip(fake.calls, TITLES + ["Судья"]):
        assert title in call["messages"][0]["content"]


def test_critic_sees_researcher(client_factory):
    client, fake = client_factory(good_run())
    client.post("/debate", json={"task": "Свой курьер или подрядчик?"})
    assert "Использовать стороннего оператора" in fake.calls[1]["messages"][1]["content"]


def test_two_rounds(client_factory):
    client, fake = client_factory(good_run(rounds=2))
    r = client.post("/debate", json={"task": "Свой курьер или подрядчик?", "rounds": 2})
    assert r.status_code == 200
    assert r.json()["llm_calls"] == 9
    assert "раунд 2 из 2" in fake.calls[4]["messages"][1]["content"]


def test_retry_on_invalid_json(client_factory):
    contents = good_run()
    contents.insert(0, "not json")
    client, fake = client_factory(contents)
    r = client.post("/debate", json={"task": "Свой курьер или подрядчик?"})
    assert r.status_code == 200
    assert len(fake.calls) == 6


def test_failure_names_the_role(client_factory):
    contents = [json.dumps(OUT["researcher"], ensure_ascii=False), '{"x": 1}', '{"x": 1}']
    client, _ = client_factory(contents)
    r = client.post("/debate", json={"task": "Свой курьер или подрядчик?"})
    assert r.status_code == 502
    assert "Критик" in r.json()["detail"]


def test_request_validation(client_factory):
    client, _ = client_factory([])
    assert client.post("/debate", json={"task": "ок"}).status_code == 422
    assert client.post("/debate", json={"task": "Нормальная задача", "temperature": 5}).status_code == 422
    assert client.post("/debate", json={"task": "Нормальная задача", "rounds": 9}).status_code == 422


def test_strict_schema_sent(client_factory):
    client, fake = client_factory(good_run())
    client.post("/debate", json={"task": "Свой курьер или подрядчик?"})
    rf = fake.calls[0]["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"]["additionalProperties"] is False


def test_prompt_mode_with_fenced_json(client_factory):
    fenced = ["Вот ответ:\n```json\n" + c + "\n```" for c in good_run()]
    client, fake = client_factory(fenced, mode="prompt")
    assert client.post("/debate", json={"task": "Свой курьер или подрядчик?"}).status_code == 200
    assert "response_format" not in fake.calls[0]
    assert "position" in fake.calls[0]["messages"][0]["content"]


def test_index_and_roles(client_factory):
    client, _ = client_factory([])
    assert "AI Debate Lab" in client.get("/").text
    roles = client.get("/roles").json()
    assert [r["id"] for r in roles] == ORDER + ["judge"]


def _stream(client, **body):
    with client.stream("POST", "/debate/stream", json=body) as r:
        assert r.status_code == 200
        return [json.loads(line) for line in r.iter_lines() if line]


def test_stream_events(client_factory):
    client, _ = client_factory(good_run())
    events = _stream(client, task="Свой курьер или подрядчик?")
    assert [e["type"] for e in events] == ["start", "entry"] * 4 + ["start", "result"]
    assert [e["role"] for e in events if e["type"] == "start"] == ORDER + ["judge"]
    assert events[-1]["data"]["final_report"]["final_decision"] == OUT["judge"]["final_decision"]


def test_stream_error_event(client_factory):
    contents = [json.dumps(OUT["researcher"], ensure_ascii=False), '{"x": 1}', '{"x": 1}']
    client, _ = client_factory(contents)
    events = _stream(client, task="Свой курьер или подрядчик?")
    assert events[-1]["type"] == "error"
    assert "Критик" in events[-1]["detail"]


def test_brevity_rule_in_prompt(client_factory):
    client, fake = client_factory(good_run())
    client.post("/debate", json={"task": "Свой курьер или подрядчик?"})
    assert "кратко" in fake.calls[0]["messages"][0]["content"]
