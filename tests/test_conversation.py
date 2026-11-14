from __future__ import annotations

import contextlib
import random
from dataclasses import replace
from pathlib import Path

import pytest
from fakes import Clock

from doppelgram.config import ConversationConfig
from doppelgram.conversation import (
    ChatMessage,
    ConversationManager,
    ConversationStore,
    PersonRef,
    classify_incoming,
    humanize,
)
from doppelgram.llm import LLMResult
from doppelgram.transport import AccountRestricted


class FakePeople:
    def __init__(self):
        self.chats: dict[int, list[ChatMessage]] = {}
        self.next_id = 1
        self.typing_calls = 0
        self.fail_send: Exception | None = None

    async def resolve(self, links, mention_ids):
        if not links:
            return None
        return PersonRef(555, "masha_example", "Маша")

    async def history(self, user_id, limit):
        return self.chats.get(user_id, [])[-limit:]

    def incoming(self, user_id, text, media_kind=None):
        self.next_id += 1
        self.chats.setdefault(user_id, []).append(ChatMessage(self.next_id, False, text, 0.0, media_kind))

    def owner_writes(self, user_id, text):
        self.next_id += 1
        self.chats.setdefault(user_id, []).append(ChatMessage(self.next_id, True, text, 0.0))

    async def send(self, user_id, text):
        if self.fail_send:
            raise self.fail_send
        self.next_id += 1
        self.chats.setdefault(user_id, []).append(ChatMessage(self.next_id, True, text, 0.0))
        return self.next_id

    async def typing(self, user_id, seconds):
        self.typing_calls += 1

    def sent_texts(self, user_id):
        return [m.text for m in self.chats.get(user_id, []) if m.outgoing]


class FakeLLM:
    def __init__(self, reply=None):
        self.calls = []
        self.reply = reply or {"action": "reply", "messages": ["Привет — увидел, что ты любишь горы."],
                               "memory": "любит горы"}

    async def complete_json(self, messages, failures=None):
        self.calls.append(messages)
        data = self.reply(messages) if callable(self.reply) else self.reply
        return LLMResult(data=data, model="m", prompt_tokens=10, completion_tokens=10, cost_usd=0.001)


def make(tmp_path: Path, **over):
    (tmp_path / "about-me.md").write_text("23 года, москва. машины нет (можно сказать: скоро будет)", encoding="utf-8")
    cfg = ConversationConfig(about_file=tmp_path / "about-me.md", goal_file=tmp_path / "goal.md",
                             style_file=tmp_path / "style.md", enabled=True, settle_s=10, first_messages_per_day=3,
                             reply_delay_min_s=30, reply_delay_max_s=30)
    cfg = replace(cfg, **over)
    clock = Clock()
    people = FakePeople()
    inbox: list[str] = []

    async def notify(text, image=None):
        inbox.append(text)
        return 9000 + len(inbox)

    m = ConversationManager(cfg, FakeLLM(), people, ConversationStore(":memory:"), notify,
                            clock=clock, sleep=clock.sleep, rng=random.Random(0))
    return m, people, inbox, clock


async def match(m, age=None):
    await m.on_match("дайвинчик", ["https://t.me/masha_example"], [], "Маша, 23, Москва – люблю горы", age)


def last_card(inbox) -> int:
    return 9000 + len(inbox)


async def test_disabled_by_default_does_nothing(tmp_path):
    m, people, inbox, clock = make(tmp_path, enabled=False)
    await match(m)
    await m.tick()
    assert m.store.people() == [] and inbox == [] and m.llm.calls == []


async def test_first_message_is_drafted_and_sent_after_ok(tmp_path):
    m, people, inbox, clock = make(tmp_path)
    await match(m)
    await m.tick()
    assert people.sent_texts(555) == []  # nothing goes out without the owner
    assert "Черновик:\nпривет - увидел, что ты любишь горы" in inbox[-1]
    prompt = m.llm.calls[0][1]["content"]
    assert "машины нет (можно сказать: скоро будет)" in prompt and "люблю горы" in prompt
    await m.answer_card(last_card(inbox), "ок")
    assert people.sent_texts(555) == ["привет - увидел, что ты любишь горы"]
    assert "Отправлено" in inbox[-1]


async def test_owner_text_is_sent_verbatim_and_kept_as_style(tmp_path):
    m, people, inbox, clock = make(tmp_path)
    await match(m)
    await m.tick()
    await m.answer_card(last_card(inbox), "хай, ты тоже из москвы?")
    assert people.sent_texts(555) == ["хай, ты тоже из москвы?"]
    assert m.store.style_examples() == ["хай, ты тоже из москвы?"]


async def test_ai_question_is_escalated_without_the_model_and_pauses_the_chat(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto")
    await match(m)
    await m.tick()
    clock.t += 31
    await m.tick()
    calls = len(m.llm.calls)
    people.incoming(555, "слушай, а ты случайно не бот?")
    m.on_message(555)
    clock.t += 11
    await m.tick()
    assert len(m.llm.calls) == calls  # deterministic: the model is not asked
    assert "не бот ли это" in inbox[-1]
    assert m.store.person(555).state == "paused"
    people.incoming(555, "ну?")
    m.on_message(555)
    clock.t += 11
    await m.tick()
    assert people.sent_texts(555)[-1] != "ну?" and len(people.sent_texts(555)) == 1
    await m.answer_card(last_card(inbox) - 0, "нет конечно, а что?")
    assert people.sent_texts(555)[-1] == "нет конечно, а что?"
    assert m.store.person(555).state == "active"


@pytest.mark.parametrize("text,category", [
    ("давай встретимся в субботу", "meeting"), ("скинь на карту 500", "money"),
    ("вот мой инст @masha_travel", "contact"), ("мне 16 если что", "minor"), ("пришли нюдс", "explicit"),
])
def test_sensitive_topics_are_detected(text, category):
    assert classify_incoming(text, None) == category


def test_media_is_always_escalated():
    assert classify_incoming("", "voice") == "media" and classify_incoming("смотри", "photo") == "media"


def test_humanize():
    assert humanize("Привет — как дела?", lowercase=True) == "привет - как дела?"
    assert humanize("Отлично.", lowercase=True) == "отлично"
    assert humanize("МГУ норм", lowercase=True) == "МГУ норм"
    assert humanize("Ну…", lowercase=False) == "Ну..."


async def test_auto_mode_sends_after_delay_with_typing(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto")
    await match(m)
    await m.tick()
    assert people.sent_texts(555) == []
    clock.t += 31
    await m.tick()
    assert people.sent_texts(555) == ["привет - увидел, что ты любишь горы"] and people.typing_calls == 1


async def test_quiet_hours_postpone_auto_messages(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", quiet_hours="00:00-23:59")
    await match(m)
    await m.tick()
    clock.t += 31
    await m.tick()
    assert people.sent_texts(555) == []
    assert m.store.next_due() > clock.t


async def test_owner_under_18_never_gets_auto_mode(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", owner_age=17)
    assert m.mode == "approve"
    await m.command("chat_mode", "auto")
    assert "недоступен" in inbox[-1]
    await match(m)
    await m.tick()
    clock.t += 31
    await m.tick()
    assert people.sent_texts(555) == [] and "Черновик" in inbox[-1]


async def test_minor_match_is_blocked(tmp_path):
    m, people, inbox, clock = make(tmp_path)
    await match(m, age=17)
    await m.tick()
    assert m.store.person(555).state == "blocked" and m.llm.calls == []
    m.on_message(555)
    await m.tick()
    assert m.llm.calls == []


async def test_owner_writing_themselves_takes_over(tmp_path):
    m, people, inbox, clock = make(tmp_path)
    await match(m)
    people.owner_writes(555, "привет)")
    await m.tick()
    assert m.store.person(555).state == "owner" and m.llm.calls == []
    assert "Вы сами пишете" in inbox[-1]


async def test_stale_draft_is_not_sent(tmp_path):
    m, people, inbox, clock = make(tmp_path)
    await match(m)
    people.incoming(555, "привет")
    await m.tick()
    card = last_card(inbox)
    people.incoming(555, "ты тут?")
    await m.answer_card(card, "ок")
    assert people.sent_texts(555) == []
    assert "успел(а) написать ещё" in inbox[-1]


async def test_model_output_with_contact_is_escalated(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto")
    m.llm.reply = {"action": "reply", "messages": ["пиши в инст @my_insta"]}
    await match(m)
    await m.tick()
    clock.t += 31
    await m.tick()
    assert people.sent_texts(555) == [] and "контакт" in inbox[-1]


async def test_first_message_cap(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", first_messages_per_day=1)

    async def resolve(links, ids):
        return PersonRef(int(links[0]), None, f"p{links[0]}")

    people.resolve = resolve
    await m.on_match("дайвинчик", ["1"], [], "", None)
    await m.on_match("дайвинчик", ["2"], [], "", None)
    await m.tick()
    clock.t += 31
    await m.tick()
    assert len(people.sent_texts(1)) + len(people.sent_texts(2)) == 1


async def test_restriction_on_send_trips_the_breaker(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto")
    tripped = []

    async def breaker(reason):
        tripped.append(reason)

    m.on_restricted = breaker
    people.fail_send = AccountRestricted("PeerFloodError")
    await match(m)
    await m.tick()
    clock.t += 31
    await m.tick()
    assert tripped == ["PeerFloodError"] and m.stopped
    clock.t += 1000
    people.fail_send = None
    await m.tick()
    assert people.sent_texts(555) == []


async def test_end_to_end_match_from_leomatch_to_owner_approved_message(tmp_path):
    from fakes import ScriptedLLM, make_config
    from fakes_dating import FakeLeomatch
    from test_dating import LEO, leo_decide

    from doppelgram.agent import Agent
    from doppelgram.models import OwnerMessage
    from doppelgram.storage import Store
    from doppelgram.supervisor import Supervisor

    m, people, inbox, clock = make(tmp_path)
    cfg = make_config(tmp_path)
    leo = FakeLeomatch()
    leo.notify_owner = m._notify
    agent = Agent(cfg, leo, ScriptedLLM(leo_decide), Store(":memory:"), clock=clock, sleep=clock.sleep,
                  rng=random.Random(0), behavior=replace(cfg.behavior, mode="auto", **LEO), name="дайвинчик")
    sup = Supervisor({"дайвинчик": agent}, m._notify, conversations=m)
    for _ in range(4):
        await agent.step_safely()
    person = m.store.person(555)
    assert person is not None and person.card.startswith("Маша, Москва")  # attached, without the age
    await m.tick()
    draft_card = last_card(inbox)
    assert "Черновик" in inbox[-1]
    # A slash command in reply to a draft runs the command instead of being sent to the person.
    await sup.handle_owner(OwnerMessage(id=1, chat_id=42, text="/стоп", reply_to=draft_card))
    assert not agent.running and people.sent_texts(555) == []
    await sup.handle_owner(OwnerMessage(id=2, chat_id=42, text="ок", reply_to=draft_card))
    assert people.sent_texts(555) == ["привет - увидел, что ты любишь горы"]


# ---- regressions from the conversations review ------------------------------------------------
async def _auto_queued(tmp_path, **over):
    m, people, inbox, clock = make(tmp_path, mode="auto", **over)
    await match(m)
    await m.tick()  # composed and queued, delay 30 s
    return m, people, inbox, clock


async def test_switch_to_approve_turns_queued_message_into_draft(tmp_path):
    m, people, inbox, clock = await _auto_queued(tmp_path)
    await m.command("chat_mode", "approve")
    clock.t += 31
    await m.tick()
    assert people.sent_texts(555) == [] and "Черновик" in inbox[-1]


async def test_breaker_blocks_card_answers(tmp_path):
    m, people, inbox, clock = make(tmp_path)
    await match(m)
    await m.tick()
    card = last_card(inbox)
    m.restrict("PeerFloodError")
    await m.answer_card(card, "ок")
    await m.answer_card(card, "своё сообщение")
    assert people.sent_texts(555) == [] and "ограничен" in inbox[-1]


async def test_message_arriving_during_typing_cancels_the_send(tmp_path):
    m, people, inbox, clock = await _auto_queued(tmp_path)

    async def typing(user_id, seconds):
        people.incoming(555, "слушай, ты бот?")

    people.typing = typing
    clock.t += 31
    await m.tick()
    assert people.sent_texts(555) == []


async def test_restriction_while_notifying_trips_breaker(tmp_path):
    m, people, inbox, clock = make(tmp_path)
    tripped = []

    async def breaker(reason):
        tripped.append(reason)

    async def notify(text, image=None):
        raise AccountRestricted("PeerFloodError")

    m.on_restricted = breaker
    m._notify = notify
    await match(m)
    await m.tick()
    assert tripped == ["PeerFloodError"] and m.stopped


async def test_later_match_with_minor_age_blocks_known_person(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto")
    await match(m, age=None)
    await match(m, age=16)
    await m.tick()
    clock.t += 31
    await m.tick()
    assert m.store.person(555).state == "blocked" and people.sent_texts(555) == []


async def test_owner_text_respects_hourly_cap(tmp_path):
    m, people, inbox, clock = make(tmp_path, max_sends_per_hour=0)
    await match(m)
    await m.tick()
    await m.answer_card(last_card(inbox), "привет, я сам")
    assert people.sent_texts(555) == [] and "лимит" in inbox[-1]


async def test_batch_respects_hourly_cap(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", max_sends_per_hour=1)
    m.llm.reply = {"action": "reply", "messages": ["раз", "два", "три"]}
    await match(m)
    await m.tick()
    clock.t += 31
    await m.tick()
    assert people.sent_texts(555) == ["раз"]


async def test_approved_draft_waits_for_quiet_hours_to_end(tmp_path):
    m, people, inbox, clock = make(tmp_path, quiet_hours="00:00-23:59")
    await match(m)
    await m.tick()
    await m.answer_card(last_card(inbox), "ок")
    assert people.sent_texts(555) == [] and "тихие часы" in inbox[-1]


@pytest.mark.parametrize("text", ["живой человек?", "ты человек?", "ты вообще настоящий?",
                                  "это правда ты пишешь?", "кто мне отвечает?"])
def test_real_person_questions_are_ai_questions(text):
    assert classify_incoming(text, None) == "ai_question"


@pytest.mark.parametrize("text", ["я человек простой", "живу одна", "настоящий рок"])
def test_ordinary_text_is_not_an_ai_question(text):
    assert classify_incoming(text, None) is None


def test_humanize_keeps_inner_quotes():
    assert humanize("Люблю «Кино»", lowercase=True) == "люблю «Кино»"
    assert humanize("«привет»", lowercase=True) == "привет"


async def test_lead_after_takeover_is_not_reverted_by_old_owner_messages(tmp_path):
    m, people, inbox, clock = make(tmp_path)
    await match(m)
    people.owner_writes(555, "привет)")
    await m.tick()
    assert m.store.person(555).state == "owner"
    await m.command("lead", 555)
    people.incoming(555, "привет, как ты?")
    m.on_message(555)
    clock.t += 11
    await m.tick()
    assert m.store.person(555).state == "active" and "Черновик" in inbox[-1]


# ---- regressions from the verification of the send rework -------------------------------------
def _typing_hook(people, action):
    async def typing(user_id, seconds):
        action()
    people.typing = typing


async def test_mode_switch_during_typing_cancels_auto_send(tmp_path):
    m, people, inbox, clock = await _auto_queued(tmp_path)
    _typing_hook(people, lambda: (m.store.set("mode", "approve"), setattr(m, "epoch", m.epoch + 1)))
    clock.t += 31
    await m.tick()
    assert people.sent_texts(555) == []


async def test_breaker_during_history_cancels_send(tmp_path):
    m, people, inbox, clock = await _auto_queued(tmp_path)
    original = people.history
    calls = []

    async def history(user_id, limit):
        calls.append(1)
        if len(calls) >= 1:
            m.restrict("PeerFloodError")
        return await original(user_id, limit)

    people.history = history
    clock.t += 31
    m.store.set("restricted", "")  # tick must start, the restriction lands during the send
    await m.tick()
    assert people.sent_texts(555) == []


async def test_takeover_during_history_cancels_send(tmp_path):
    m, people, inbox, clock = await _auto_queued(tmp_path)
    original = people.history

    async def history(user_id, limit):
        m.store.update(555, state="owner")
        return await original(user_id, limit)

    people.history = history
    clock.t += 31
    await m.tick()
    assert people.sent_texts(555) == []


async def test_old_database_is_migrated(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite3"
    db = sqlite3.connect(path)
    db.executescript("""
    CREATE TABLE people (user_id INTEGER PRIMARY KEY, username TEXT, name TEXT NOT NULL, bot TEXT NOT NULL,
        card TEXT NOT NULL DEFAULT '', age INTEGER, state TEXT NOT NULL DEFAULT 'active',
        memory TEXT NOT NULL DEFAULT '', due_at REAL, handled_upto INTEGER NOT NULL DEFAULT 0,
        first_sent INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL);
    CREATE TABLE outbox (user_id INTEGER PRIMARY KEY, upto INTEGER NOT NULL, messages TEXT NOT NULL,
        send_at REAL NOT NULL);
    INSERT INTO people(user_id, name, bot, created) VALUES (1, 'Маша', 'дайвинчик', 0);
    INSERT INTO outbox VALUES (1, 0, '["привет"]', 0);
    """)
    db.commit()
    db.close()
    store = ConversationStore(path)
    assert store.person(1).owner_upto == 0
    assert store.due_outbox(1.0) == [(1, 0, ["привет"], False)]


async def test_capped_batch_remainder_is_not_marked_approved(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", max_sends_per_hour=1)
    m.llm.reply = {"action": "reply", "messages": ["раз", "два"]}
    await match(m)
    await m.tick()
    clock.t += 31
    await m.tick()
    assert people.sent_texts(555) == ["раз"]
    await m.command("chat_mode", "approve")
    clock.t += 3600
    await m.tick()
    assert people.sent_texts(555) == ["раз"]  # "два" became a draft, not an automatic send
    assert "Черновик" in inbox[-1]


async def test_queued_approved_message_and_new_draft_never_both_send(tmp_path):
    m, people, inbox, clock = make(tmp_path, quiet_hours="00:00-23:59")
    await match(m)
    people.incoming(555, "привет")
    await m.tick()
    await m.answer_card(last_card(inbox), "ок")  # queued until quiet hours end
    await m.command("lead", 555)  # agent re-reads and drafts again
    await m.tick()
    await m.answer_card(last_card(inbox), "ок")
    m.cfg = replace(m.cfg, quiet_hours="")
    clock.t += 200_000
    await m.tick()
    assert len(people.sent_texts(555)) <= 1


async def test_failure_mid_batch_never_resends(tmp_path):
    m, people, inbox, clock = make(tmp_path)
    m.llm.reply = {"action": "reply", "messages": ["раз", "два"]}
    await match(m)
    await m.tick()
    card = last_card(inbox)
    original = people.send
    count = []

    async def flaky(user_id, text):
        count.append(text)
        if len(count) == 2:
            raise RuntimeError("network")
        return await original(user_id, text)

    people.send = flaky
    with contextlib.suppress(RuntimeError):
        await m.answer_card(card, "ок")
    people.send = original
    await m.answer_card(card, "ок")
    assert people.sent_texts(555) == ["раз"]


async def test_takeover_during_owner_send_is_kept(tmp_path):
    m, people, inbox, clock = make(tmp_path)
    await match(m)
    await m.tick()
    card = last_card(inbox)
    sent_once = []

    async def typing(user_id, seconds):
        if not sent_once:
            sent_once.append(1)
    people.typing = typing
    original = people.send

    async def send(user_id, text):
        msg_id = await original(user_id, text)
        await m.command("self", 555)  # owner takes over while the message is in flight
        return msg_id

    people.send = send
    await m.answer_card(card, "я сам напишу")
    assert m.store.person(555).state == "owner"


async def test_escalation_draft_can_be_approved(tmp_path):
    m, people, inbox, clock = make(tmp_path)
    m.llm.reply = {"action": "escalate", "category": "unclear", "messages": ["хм, расскажи подробнее"]}
    await match(m)
    await m.tick()
    assert m.store.person(555).state == "paused"
    await m.answer_card(last_card(inbox), "ок")
    assert people.sent_texts(555) == ["хм, расскажи подробнее"]


@pytest.mark.parametrize("text", ["ты реально любишь горы?", "ты настоящая молодец", "на работе пишу скрипты",
                                  "кто тебе пишет в такое время", "мне это автоматически пришло"])
def test_human_regex_ignores_ordinary_speech(text):
    assert classify_incoming(text, None) is None


@pytest.mark.parametrize("text", ["ты живой?", "ты настоящая?", "ты вообще человек?", "живой человек?",
                                  "это правда ты пишешь?", "кто мне отвечает?", "это автоматический ответ?"])
def test_human_regex_still_catches_questions(text):
    assert classify_incoming(text, None) == "ai_question"


def test_humanize_strips_ascii_quote_pair():
    assert humanize('"привет"', lowercase=True) == "привет"
    assert humanize('он сказал "да" и ушёл', lowercase=True) == 'он сказал "да" и ушёл'


@pytest.mark.parametrize("text,expected", [("на работе пишу скрипты", None), ("вложился в крипту", "money"),
                                           ("займи до пятницы", "money"), ("оплатишь такси?", "money")])
def test_money_regex_word_boundaries(text, expected):
    assert classify_incoming(text, None) == expected


async def test_takeover_command_during_approved_send_stops_it(tmp_path):
    m, people, inbox, clock = make(tmp_path)
    await match(m)
    await m.tick()
    card = last_card(inbox)

    async def typing(user_id, seconds):
        await m.command("self", 555)  # "сам 555" arrives while the approved draft is being typed

    people.typing = typing
    await m.answer_card(card, "ок")
    assert people.sent_texts(555) == [] and m.store.person(555).state == "owner"


# ---- regressions from the last verification ---------------------------------------------------
async def test_ok_waiting_for_the_lock_is_cancelled_by_takeover(tmp_path):
    import asyncio

    m, people, inbox, clock = make(tmp_path)
    await match(m)
    await m.tick()
    card = last_card(inbox)
    await m.lock.acquire()  # another chat is being sent
    task = asyncio.create_task(m.answer_card(card, "ок"))
    await asyncio.sleep(0)
    await m.command("self", 555)
    m.lock.release()
    await task
    assert people.sent_texts(555) == [] and m.store.person(555).state == "owner"


async def test_capped_escalation_batch_remainder_is_delivered_later(tmp_path):
    m, people, inbox, clock = make(tmp_path, max_sends_per_hour=1)
    m.llm.reply = {"action": "escalate", "category": "unclear", "messages": ["раз", "два"]}
    await match(m)
    await m.tick()
    await m.answer_card(last_card(inbox), "ок")
    assert people.sent_texts(555) == ["раз"]
    clock.t += 3700
    await m.tick()
    assert people.sent_texts(555) == ["раз", "два"]


async def test_history_failure_requeues_unsent_auto_reply(tmp_path):
    from doppelgram.transport import TransportError

    m, people, inbox, clock = await _auto_queued(tmp_path)
    original = people.history

    async def broken(user_id, limit):
        raise TransportError("network")

    people.history = broken
    clock.t += 31
    await m.tick()
    assert people.sent_texts(555) == []
    people.history = original
    clock.t += 121
    await m.tick()
    assert people.sent_texts(555) == ["привет - увидел, что ты любишь горы"]


async def test_send_failure_is_never_retried_automatically(tmp_path):
    from doppelgram.transport import TransportError

    m, people, inbox, clock = await _auto_queued(tmp_path)
    people.fail_send = TransportError("timeout")
    clock.t += 31
    await m.tick()
    people.fail_send = None
    clock.t += 10_000
    await m.tick()
    assert people.sent_texts(555) == [] and "Не уверен, дошло ли" in inbox[-1]


@pytest.mark.parametrize("failure", ["history", "cap"])
async def test_restored_card_keeps_the_owner_text(tmp_path, failure):
    from doppelgram.transport import TransportError

    m, people, inbox, clock = make(tmp_path, max_sends_per_hour=0 if failure == "cap" else 20)
    await match(m)
    await m.tick()
    card = last_card(inbox)
    original = people.history
    if failure == "history":
        async def broken(user_id, limit):
            raise TransportError("network")
        people.history = broken
    await m.answer_card(card, "мой собственный текст")
    people.history = original
    m.cfg = replace(m.cfg, max_sends_per_hour=20)
    await m.answer_card(card, "ок")
    assert people.sent_texts(555) == ["мой собственный текст"]


async def test_match_profile_is_stored_without_age_and_age_is_never_stored(tmp_path):
    m, people, inbox, clock = make(tmp_path)
    await m.on_match("дайвинчик", ["https://t.me/masha_example"], [], "Маша, 23, Москва – рост 170", 23)
    person = m.store.person(555)
    assert person.age is None and not any(ch.isdigit() for ch in person.card)
    raw = m.store.db.execute("SELECT card, age FROM people").fetchall()
    assert raw == [("Маша, Москва – рост", None)]
    await m.tick()
    person_section = m.llm.calls[-1][1]["content"].split("## The person")[1].split("## Chat so far")[0]
    assert "23" not in person_section


async def test_conversation_memory_is_sanitized_in_the_prompt(tmp_path):
    from doppelgram.brain import sanitize_age

    m, people, inbox, clock = make(tmp_path)
    m.sanitize = lambda bot, text: sanitize_age(text, "18")
    await match(m)
    m.store.update(555, memory="маше 23 года, любит горы")
    await m.tick()
    notes = m.llm.calls[-1][1]["content"].split("## Your notes about this person")[1].split("##")[0]
    assert "23" not in notes and "18 года" in notes


# ------------------------------------------------------------- open-source review regressions
async def test_with_no_first_messages_the_agent_still_answers(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", first_messages_per_day=0)
    await match(m)
    await m.tick()
    assert people.sent_texts(555) == [] and m.llm.calls == []  # never writes first
    people.incoming(555, "привет, как дела?")
    m.on_message(555)
    m.llm.reply = {"action": "reply", "messages": ["привет, всё хорошо"]}
    for _ in range(3):
        clock.t += 31
        await m.tick()
    assert people.sent_texts(555) == ["привет, всё хорошо"]


async def test_a_failed_chat_step_is_retried(tmp_path):
    from doppelgram.llm import LLMError
    m, people, inbox, clock = make(tmp_path, mode="auto")
    await match(m)
    m.llm.reply = lambda msgs: (_ for _ in ()).throw(LLMError("down"))
    await m.tick()
    assert m.store.next_due() is not None
    m.llm.reply = {"action": "reply", "messages": ["привет"]}
    clock.t += ConversationManager.RETRY_S + 1
    await m.tick()
    clock.t += 31
    await m.tick()
    assert people.sent_texts(555) == ["привет"]


async def test_a_malformed_reply_escalates_and_does_not_crash(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto")
    m.llm.reply = {"action": "reply", "messages": 42}
    await match(m)
    await m.tick()
    assert people.sent_texts(555) == [] and inbox


async def test_the_agent_never_proposes_a_meeting_by_itself(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto")
    await match(m)
    people.incoming(555, "чем занимаешься?")
    m.on_message(555)
    m.llm.reply = {"action": "reply", "messages": ["приеду завтра в восемь"]}
    await m.tick()
    clock.t += 31
    await m.tick()
    assert people.sent_texts(555) == [] and any("встреч" in t.lower() or "решаете вы" in t for t in inbox)
