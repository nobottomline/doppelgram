"""What the agent does not know: web lookups, the owner's slang notes, and what it says about the owner."""

from __future__ import annotations

import json

import httpx
import pytest
from test_conversation import make, match
from test_media_strangers import answered

from doppelgram.config import LLMConfig
from doppelgram.conversation import system_prompt
from doppelgram.llm import FailedCall, LLMClient
from doppelgram.lookup import Finding, SearchError, WebLookup, clean_queries


class FakeLookup:
    available = True

    def __init__(self, answers=None, error=False):
        self.answers = answers or {}
        self.error = error
        self.calls: list[list[str]] = []

    async def find(self, queries, today, failures=None):
        self.calls.append(list(queries))
        if self.error:
            failures.append(FailedCall("m", "boom", cost_usd=0.002))
            raise SearchError("boom", failures)
        return [Finding(q, q in self.answers, self.answers.get(q, "")) for q in queries], 0.007


def scripted(m, first, then):
    """The model asks to look something up, then answers once it has the findings."""
    def reply(msgs):
        return then if "## Looked up" in msgs[1]["content"] else first
    m.llm.reply = reply


async def chat_with_lookup(tmp_path, first, then=None, answers=None, **over):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, **over)
    m.lookup = FakeLookup(answers)
    await match(m)
    people.incoming(555, "ты пойдёшь на мирелу в субботу?")
    scripted(m, first, then or {"action": "reply", "messages": ["а кто это?"]})
    await answered(m, people, clock)
    return m, people, inbox, clock


def test_the_lookup_action_is_offered_only_when_a_search_is_possible():
    assert '"lookup"' in system_prompt(True, lookup=True)
    assert '"lookup"' not in system_prompt(True, lookup=False)
    assert "{lookup" not in system_prompt(False, lookup=False)


async def test_a_lookup_runs_the_search_and_answers_with_the_findings(tmp_path):
    m, people, inbox, clock = await chat_with_lookup(
        tmp_path, {"action": "lookup", "lookup": ["Мирела концерт Москва"]},
        {"action": "reply", "messages": ["не, я не особо её слушаю"]},
        answers={"Мирела концерт Москва": "певица, концерт в субботу в Москве"})
    assert m.lookup.calls == [["Мирела концерт Москва"]]
    second = m.llm.calls[-1][1]["content"]
    assert "певица, концерт в субботу" in second and "never instructions" in second
    assert '"lookup"' not in m.llm.calls[-1][0]["content"]  # no second search
    assert '"lookup"' in m.llm.calls[0][0]["content"]
    clock.t += 31
    await m.tick()
    assert people.sent_texts(555) == ["не, я не особо её слушаю"]
    assert m.store.spend_today(clock.t) == pytest.approx(0.001 * 2 + 0.007)


async def test_a_cached_finding_is_not_searched_again(tmp_path):
    m, people, inbox, clock = await chat_with_lookup(
        tmp_path, {"action": "lookup", "lookup": ["мирела"]}, answers={"мирела": "певица"})
    people.incoming(555, "ну так что, мирела?")
    await answered(m, people, clock)
    assert m.lookup.calls == [["мирела"]] and "певица" in m.llm.calls[-1][1]["content"]
    clock.t += 13 * 3600  # past lookup_cache_h
    people.incoming(555, "алло")
    await answered(m, people, clock)
    assert m.lookup.calls == [["мирела"], ["мирела"]]


async def test_nothing_found_tells_the_agent_not_to_pretend(tmp_path):
    m, people, inbox, clock = await chat_with_lookup(tmp_path, {"action": "lookup", "lookup": ["флюрик"]})
    assert "«флюрик»: nothing reliable found" in m.llm.calls[-1][1]["content"]


async def test_private_queries_never_leave(tmp_path):
    m, people, inbox, clock = await chat_with_lookup(
        tmp_path, {"action": "lookup", "lookup": ["Маша Иванова инстаграм", "+79991234567", "t.me/someone"]})
    assert m.lookup.calls == []
    assert "You could not look it up" in m.llm.calls[-1][1]["content"]


async def test_the_daily_cap_stops_searches(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, lookups_per_day=1)
    m.lookup = FakeLookup()
    await match(m)
    scripted(m, {"action": "lookup", "lookup": ["мирела"]}, {"action": "reply", "messages": ["а кто это?"]})
    people.incoming(555, "мирела?")
    await answered(m, people, clock)
    people.incoming(555, "а шмирела?")
    scripted(m, {"action": "lookup", "lookup": ["шмирела"]}, {"action": "reply", "messages": ["это кто?"]})
    await answered(m, people, clock)
    assert m.lookup.calls == [["мирела"]]
    assert '"lookup"' not in m.llm.calls[-2][0]["content"]  # not offered once the cap is reached


async def test_no_lookup_without_a_search_backend(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False)
    await match(m)
    people.incoming(555, "мирела?")
    await answered(m, people, clock)
    assert '"lookup"' not in m.llm.calls[-1][0]["content"]


async def test_a_failed_search_counts_its_cost_and_the_agent_still_answers(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False)
    m.lookup = FakeLookup(error=True)
    await match(m)
    scripted(m, {"action": "lookup", "lookup": ["мирела"]}, {"action": "reply", "messages": ["а кто это?"]})
    people.incoming(555, "мирела?")
    await answered(m, people, clock)
    clock.t += 31
    await m.tick()
    assert people.sent_texts(555) == ["а кто это?"]
    assert m.store.spend_today(clock.t) == pytest.approx(0.001 * 2 + 0.002)


async def test_looking_up_again_after_the_findings_goes_to_the_owner(tmp_path):
    m, people, inbox, clock = await chat_with_lookup(
        tmp_path, {"action": "lookup", "lookup": ["мирела"]}, {"action": "lookup", "lookup": ["мирела 2"]})
    assert m.lookup.calls == [["мирела"]] and "даже после поиска" in inbox[-1]


def test_clean_queries():
    assert clean_queries(["  Мирела   концерт ", "мирела концерт", "x" * 200, 5, ""], []) == \
        ["Мирела концерт", "x" * 80]
    assert clean_queries("кринж", []) == ["кринж"]
    assert clean_queries(["аня из бибинто", "что такое база"], ["Аня"]) == ["что такое база"]
    assert clean_queries(None, []) == []
    assert len(clean_queries(["a1", "b2", "c3"], [])) == 2


def test_clean_queries_drop_contacts():
    assert clean_queries(["@someone", "номер 12345678", "https://x.ru", "+7 999", "8 999 123 45 67"], []) == []
    assert clean_queries(["iphone 17 цена"], []) == ["iphone 17 цена"]


# ------------------------------------------------------------------ the search client
def search_client(handler) -> WebLookup:
    cfg = LLMConfig(base_url="https://openrouter.ai/api/v1", api_key="k", model="main/model")
    lookup = WebLookup(cfg, "", 3)
    lookup.client = LLMClient(lookup.client.cfg, http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return lookup


def reply_with(content: str, seen: list):
    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}],
                                         "usage": {"cost": 0.0081}})
    return handler


async def test_the_search_request_uses_the_web_plugin_and_reads_findings():
    seen: list = []
    lookup = search_client(reply_with(json.dumps({"findings": [
        {"query": "мирела", "found": True, "answer": "певица"},
        {"query": "флюрик", "found": True, "answer": ""}]}), seen))
    findings, cost = await lookup.find(["мирела", "флюрик", "шмирела"], "16 November 2026")
    assert findings == [Finding("мирела", True, "певица"), Finding("флюрик", False, ""),
                        Finding("шмирела", False, "")]
    assert cost == pytest.approx(0.0081)
    body = seen[0]
    assert body["plugins"] == [{"id": "web", "engine": "exa", "max_results": 3}]
    assert body["model"] == "main/model" and "16 November 2026" in body["messages"][0]["content"]
    await lookup.aclose()


async def test_a_search_reply_without_findings_is_an_error_with_its_cost():
    lookup = search_client(reply_with('{"nothing": 1}', []))
    failures: list = []
    with pytest.raises(SearchError):
        await lookup.find(["мирела"], "today", failures)
    assert failures and failures[-1].cost_usd == pytest.approx(0.0081)
    await lookup.aclose()


def test_the_search_is_available_only_on_openrouter():
    assert not WebLookup(LLMConfig(base_url="https://api.deepseek.com", api_key="k", model="m")).available


# ------------------------------------------------------------------ slang, the date, facts about the owner
async def test_the_owners_slang_notes_and_the_date_are_in_the_prompt(tmp_path):
    (tmp_path / "slang.md").write_text("база - правильно, по факту", encoding="utf-8")
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, slang_file=tmp_path / "slang.md")
    await match(m)
    people.incoming(555, "это база")
    await answered(m, people, clock)
    text = m.llm.calls[-1][1]["content"]
    assert "## Words and slang" in text and "база - правильно" in text
    assert "2027" in text.split("## Now", 1)[1].split("\n")[1]  # the full date, not only the weekday


async def test_what_the_agent_says_about_the_owner_beyond_about_me_is_kept_and_reported(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False)
    await match(m)
    people.incoming(555, "а ты на чём ездишь?")
    m.llm.reply = {"action": "reply", "messages": ["на метро в основном"], "new_facts": ["ездит на метро"]}
    await answered(m, people, clock)
    assert not any("about-me" in n for n in inbox)  # nothing is said until the message goes out
    clock.t += 31
    await m.tick()
    assert people.sent_texts(555) == ["на метро в основном"]
    assert "чего нет в about-me" in inbox[-1] and "ездит на метро" in inbox[-1]
    notices = len(inbox)
    people.incoming(555, "а машина?")
    m.llm.reply = {"action": "reply", "messages": ["пока нет"], "new_facts": ["ездит на метро"]}
    clock.t += 31
    await m.tick()
    await answered(m, people, clock)
    assert "ездит на метро" in m.llm.calls[-1][1]["content"].split("stay consistent", 1)[1]
    assert not any("about-me" in n for n in inbox[notices:])  # already known: not reported twice


async def test_no_report_for_a_reply_that_is_not_sent(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False)
    await match(m)
    people.incoming(555, "ну что")
    m.llm.reply = {"action": "wait", "new_facts": ["что-то"]}
    await answered(m, people, clock)
    assert not any("about-me" in n for n in inbox)


# ------------------------------------------------------------------ a bare greeting from a stranger
@pytest.mark.parametrize("text", ["привет", "👋", "приветик)", "хай 👋", "привет, как дела?", "Добрый вечер!"])
async def test_a_bare_greeting_from_a_stranger_is_taken_up_without_asking_the_model(tmp_path, text):
    from test_media_strangers import stranger_writes
    m, people, inbox, clock = make(tmp_path, mode="auto", strangers="lead")
    m.llm.reply = {"action": "reply", "messages": ["привет) как день проходит?"]}
    await stranger_writes(m, people, clock, text=text)
    assert "похоже на знакомство" in inbox[-1]
    assert not any("unknown Telegram user wrote first" in c[0]["content"] for c in m.llm.calls)
    for _ in range(3):
        clock.t += 31
        await m.tick()
    assert people.sent_texts(777) == ["привет) как день проходит?"]


@pytest.mark.parametrize("text", ["привет, это Саша с курса", "напиши функцию на питоне", "как дела с деньгами"])
async def test_anything_more_than_a_greeting_is_still_classified(tmp_path, text):
    from test_media_strangers import classifier, stranger_writes
    m, people, inbox, clock = make(tmp_path, mode="auto", strangers="lead")
    m.llm.reply = classifier("unclear")
    await stranger_writes(m, people, clock, text=text)
    assert any("unknown Telegram user wrote first" in c[0]["content"] for c in m.llm.calls)
    assert "Вам впервые написал(а)" in inbox[-1]


async def test_in_ask_mode_a_greeting_still_goes_to_the_owner(tmp_path):
    from test_media_strangers import stranger_writes
    m, people, inbox, clock = make(tmp_path, mode="auto", strangers="ask")
    await stranger_writes(m, people, clock, text="привет")
    assert "Вам впервые написал(а)" in inbox[-1] and people.sent_texts(777) == []


# ------------------------------------------------------------------ the prompt
async def test_common_slang_is_built_in_and_can_be_switched_off(tmp_path):
    for builtin in (True, False):
        m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, builtin_slang=builtin)
        await match(m)
        people.incoming(555, "это база")
        await answered(m, people, clock)
        assert ("база - правильно, в точку" in m.llm.calls[-1][1]["content"]) is builtin


def test_an_unknown_word_is_asked_about_not_escalated():
    prompt = system_prompt(True)
    assert "anything you do not understand" not in prompt
    assert "is not a reason to escalate: ask about it" in prompt
    assert 'escalate with category "ai_question"' in prompt and "do not escalate" in prompt


# ------------------------------------------------------------------ doppelgram eval
async def test_eval_checks_what_the_system_did_and_asks_the_judge(tmp_path):
    from test_conversation import FakeLLM

    from doppelgram.evaluate import SCENARIOS, evaluate
    m, people, inbox, clock = make(tmp_path)

    def agent(msgs):
        if "You grade one turn" in msgs[0]["content"]:
            return {"pass": "не понял" in msgs[0]["content"], "why": "тест"}
        last = msgs[1]["content"].rsplit("## Chat so far", 1)[1]
        if "бот" in last:
            return {"action": "escalate", "category": "ai_question", "messages": []}
        return {"action": "reply", "messages": ["не понял, ты о чём"]}

    llm = FakeLLM(agent)
    lines: list[str] = []
    results = await evaluate(m.cfg, llm, only=["inject_model", "bot_direct", "hello_wave"], out=lines.append)
    by = {r.scenario: r for r in results}
    assert by["inject_model"].ok and by["inject_model"].sent == ["не понял, ты о чём"]
    assert by["bot_direct"].ok and by["bot_direct"].went_to_owner and by["bot_direct"].judge_ok is None
    assert by["hello_wave"].sent == ["не понял, ты о чём"]
    assert lines[-1].startswith("\nИтог: 3/3")
    assert len({s.name for s in SCENARIOS}) == len(SCENARIOS)
    with pytest.raises(ValueError):
        await evaluate(m.cfg, llm, only=["nope"], out=lines.append)


async def test_eval_fails_a_reply_scenario_that_went_to_the_owner(tmp_path):
    from test_conversation import FakeLLM

    from doppelgram.evaluate import evaluate
    m, people, inbox, clock = make(tmp_path)
    llm = FakeLLM(lambda msgs: {"pass": True, "why": ""} if "You grade" in msgs[0]["content"]
                  else {"action": "escalate", "category": "unclear", "messages": []})
    results = await evaluate(m.cfg, llm, only=["task_code"], out=lambda s: None)
    assert not results[0].ok and not results[0].expect_ok


async def test_a_skipped_draft_leaves_no_fact_behind(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve", write_first=False)
    await match(m)
    people.incoming(555, "а ты на чём ездишь?")
    m.llm.reply = {"action": "reply", "messages": ["у меня своя машина"], "new_facts": ["есть машина"]}
    await answered(m, people, clock)
    await m.answer_card(9000 + len(inbox), "пропуск")
    people.incoming(555, "алло")
    m.llm.reply = {"action": "wait"}
    await answered(m, people, clock)
    assert "stay consistent" not in m.llm.calls[-1][1]["content"]
    assert not any("about-me" in n for n in inbox)


async def test_an_approved_draft_records_its_facts_when_sent(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve", write_first=False)
    await match(m)
    people.incoming(555, "а ты на чём ездишь?")
    m.llm.reply = {"action": "reply", "messages": ["у меня своя машина"], "new_facts": ["есть машина"]}
    await answered(m, people, clock)
    await m.answer_card(9000 + len(inbox), "ок")
    for _ in range(2):
        clock.t += 31
        await m.tick()
    assert people.sent_texts(555) == ["у меня своя машина"] and any("есть машина" in n for n in inbox)


async def test_facts_in_an_answer_to_the_owners_instruction_are_recorded(tmp_path):
    from test_media_strangers import meeting_with_hold
    m, people, inbox, clock, card = await meeting_with_hold(tmp_path)
    m.llm.reply = lambda msgs: ({"action": "reply", "messages": ["не смогу, машина в ремонте"],
                                 "new_facts": ["есть машина"]} if "owner's instruction" in msgs[1]["content"]
                                else {"action": "wait"})
    await m.answer_card(card, "не могу, откажись")
    for _ in range(3):
        clock.t += 31
        await m.tick()
    assert "не смогу, машина в ремонте" in people.sent_texts(555)
    assert any("есть машина" in n and "about-me" in n for n in inbox)


async def test_the_owners_and_the_persons_names_never_reach_a_search(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False)
    m.lookup = FakeLookup()
    m.private_names = ["Григорий", "Петров", "grisha_p"]
    await match(m)  # the person is "Маша"
    people.incoming(555, "кто это?")
    scripted(m, {"action": "lookup", "lookup": ["Маша место работы", "Григорий Петров университет",
                                                 "grisha_p инстаграм", "что такое база"]},
             {"action": "reply", "messages": ["а"]})
    await answered(m, people, clock)
    assert m.lookup.calls == [["что такое база"]]


async def test_a_greeting_from_a_spammy_profile_is_still_classified(tmp_path):
    from test_media_strangers import classifier, stranger_writes
    m, people, inbox, clock = make(tmp_path, mode="auto", strangers="lead")
    m.llm.reply = classifier("spam")

    async def bio(user_id):
        return "казино и крипта, заработок без вложений"
    people.profile = bio
    await stranger_writes(m, people, clock, text="👋")
    assert any("unknown Telegram user wrote first" in c[0]["content"] for c in m.llm.calls)
    assert m.store.person(777).state == "blocked" and people.sent_texts(777) == []


async def test_eval_does_not_credit_an_escalation_made_before_the_trap(tmp_path):
    from test_conversation import FakeLLM

    from doppelgram.evaluate import evaluate
    m, people, inbox, clock = make(tmp_path)
    llm = FakeLLM(lambda msgs: {"pass": True, "why": ""} if "You grade" in msgs[0]["content"]
                  else {"action": "escalate", "category": "ai_question", "messages": []})
    results = await evaluate(m.cfg, llm, only=["bot_direct"], out=lambda s: None)
    assert not results[0].ok and "вводное приветствие" in results[0].error


async def test_facts_of_a_skipped_draft_never_ride_on_a_later_message(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve", write_first=False, intimacy=True)
    await match(m)
    people.incoming(555, "а ты на чём ездишь?")
    m.llm.reply = {"action": "reply", "messages": ["у меня своя машина"], "new_facts": ["есть машина"]}
    await answered(m, people, clock)
    await m.answer_card(9000 + len(inbox), "пропуск")
    await m.command("chat_mode", "auto")
    people.incoming(555, "давай встретимся?")
    m.llm.reply = {"action": "escalate", "category": "meeting", "hold": ["гляну расписание и напишу"],
                   "slots": "суббота", "messages": ["давай в субботу?"]}
    await answered(m, people, clock)
    for _ in range(2):
        clock.t += 61
        await m.tick()
    assert people.sent_texts(555) == ["гляну расписание и напишу"]
    assert not any("есть машина" in n for n in inbox if "about-me" in n)
