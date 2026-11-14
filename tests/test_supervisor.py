from __future__ import annotations

import contextlib
import random
from dataclasses import replace

from fakes import Clock, FakeRatingBot, ScriptedLLM, make_config, rate_or_wait

from doppelgram.agent import Agent
from doppelgram.models import OwnerMessage
from doppelgram.storage import Store
from doppelgram.supervisor import Supervisor


class Inbox:
    """Shared owner chat: both agents and the supervisor write here."""

    def __init__(self):
        self.messages: list[str] = []

    async def __call__(self, text, image=None):
        self.messages.append(text)
        return 9000 + len(self.messages)


def like_decision(text, has_image):
    if has_image:
        return {"action": "click", "button_id": "r0c0", "reason": "лайк"}
    return {"action": "ask_owner", "reason": "необычное"}


def setup(tmp_path):
    cfg = make_config(tmp_path)
    clock = Clock()
    inbox = Inbox()
    agents, bots = {}, {}
    for name, mode, decide in (("арт", "supervised", rate_or_wait), ("лайки", "auto", like_decision)):
        bot = FakeRatingBot()
        bot.notify_owner = inbox
        bots[name] = bot
        agents[name] = Agent(cfg, bot, ScriptedLLM(decide), Store(":memory:"), clock=clock, sleep=clock.sleep,
                             rng=random.Random(0), behavior=replace(cfg.behavior, mode=mode), label=name)
    return Supervisor(agents, inbox), agents, bots, inbox, clock


async def say(sup, text, reply_to=None):
    """Commands get a slash; a reply to a card is sent as typed (that is how the owner answers)."""
    if not reply_to and not text.startswith("/"):
        text = "/" + text
    await sup.handle_owner(OwnerMessage(id=1, chat_id=42, text=text, reply_to=reply_to))


async def steps(agents, n):
    for _ in range(n):
        for a in agents.values():
            await a.step_safely()


async def test_two_bots_run_independently_with_their_own_modes(tmp_path):
    sup, agents, bots, inbox, clock = setup(tmp_path)
    await steps(agents, 6)
    assert bots["лайки"].ratings == ["1", "1", "1"]      # auto: liked everything itself
    assert bots["арт"].ratings == []                       # supervised: waits for the owner
    assert any(m.startswith("<b>арт</b> · <b>Новая анкета</b>") for m in inbox.messages)


async def test_commands_target_one_bot_or_all(tmp_path):
    sup, agents, bots, inbox, clock = setup(tmp_path)
    await say(sup, "стоп лайки")
    assert not agents["лайки"].running and agents["арт"].running
    await say(sup, "стоп")
    assert not agents["арт"].running
    await say(sup, "старт")
    assert agents["арт"].running and agents["лайки"].running
    await say(sup, "пауза 2ч арт")
    assert not agents["арт"].active() and agents["лайки"].active()


async def test_mode_applies_to_one_bot_or_all(tmp_path):
    sup, agents, bots, inbox, clock = setup(tmp_path)
    await say(sup, "режим умный арт")
    assert agents["арт"].mode == "smart" and agents["лайки"].mode == "auto"
    await say(sup, "режим ручной")
    assert agents["арт"].mode == "supervised" and agents["лайки"].mode == "supervised"


async def test_answers_are_routed_by_reply(tmp_path):
    sup, agents, bots, inbox, clock = setup(tmp_path)
    agents["лайки"].b = replace(agents["лайки"].b, mode="supervised")
    agents["лайки"].store.set("mode", "supervised")
    await steps(agents, 2)
    art_card = max(p.control_msg_id for p in [agents["арт"].store.latest_pending()])
    like_card = agents["лайки"].store.latest_pending().control_msg_id
    await say(sup, "2")
    assert "нескольких ботов" in inbox.messages[-1]
    await say(sup, "2", reply_to=art_card)
    await say(sup, "ок", reply_to=like_card)
    assert bots["арт"].ratings == ["2"] and bots["лайки"].ratings == ["1"]


async def test_status_and_help_cover_all_bots(tmp_path):
    sup, agents, bots, inbox, clock = setup(tmp_path)
    await say(sup, "статус")
    assert "<b>Арт</b>" in inbox.messages[-1] and "<b>Лайки</b>" in inbox.messages[-1]
    await say(sup, "помощь")
    assert "Боты: арт, лайки" in inbox.messages[-1]


async def test_unusual_message_stops_only_that_bot(tmp_path):
    sup, agents, bots, inbox, clock = setup(tmp_path)
    agents["лайки"].b = replace(agents["лайки"].b, mode="supervised")  # in auto it would keep going
    await steps(agents, 1)
    bots["лайки"].messages[-1] = replace(bots["лайки"].messages[-1], media_kind=None, media_key="",
                                         text="Подпишись на канал, чтобы продолжить")
    for _ in range(4):  # rating buttons without a picture: it first waits for the photo three times
        await steps(agents, 1)
        clock.t += 11
    assert not agents["лайки"].running and agents["арт"].running
    assert inbox.messages[-1].startswith("<b>лайки</b> · Нужна ваша помощь")


# ---- regressions from the second review --------------------------------------------------------
async def test_answer_with_bot_name_goes_to_that_bot(tmp_path):
    sup, agents, bots, inbox, clock = setup(tmp_path)
    await steps(agents, 2)  # only "арт" is supervised → only it has a pending card
    await say(sup, "2 лайки")
    assert bots["арт"].ratings == []
    assert "нет анкет" in inbox.messages[-1]
    art_card = agents["арт"].store.latest_pending().control_msg_id
    # A slash answer naming another bot, in reply to this bot's card, is a mistake and is refused.
    await say(sup, "/2 лайки", reply_to=art_card)
    assert "от другого бота" in inbox.messages[-1] and bots["арт"].ratings == []
    await say(sup, "2 арт")
    assert bots["арт"].ratings == ["2"]


async def test_mistyped_bot_name_is_rejected(tmp_path):
    sup, agents, bots, inbox, clock = setup(tmp_path)
    await say(sup, "стоп")
    await say(sup, "старт лйаки")
    assert not agents["арт"].running and not agents["лайки"].running
    assert "Не понял «лйаки»" in inbox.messages[-1]
    await say(sup, "пауза 2ч лйаки")
    assert "Не понял «2ч лйаки»" in inbox.messages[-1]


async def test_broadcast_stop_takes_effect_before_any_notification(tmp_path):
    sup, agents, bots, inbox, clock = setup(tmp_path)
    states = []

    async def slow_notify(text, image=None):
        states.append([a.running for a in agents.values()])
        return 1

    sup.notify = slow_notify
    for a in agents.values():
        a.tg.notify_owner = slow_notify
    await say(sup, "стоп")
    assert states and all(s == [False, False] for s in states)


async def test_borderline_confidence_in_smart_mode_goes_to_owner(tmp_path):
    sup, agents, bots, inbox, clock = setup(tmp_path)
    likes = agents["лайки"]
    likes.b = replace(likes.b, mode="smart", confidence_threshold=0.75)
    likes.llm.decide = lambda t, img: {"action": "click", "button_id": "r0c0", "confidence": 0.6,
                                       "reason": "сомневаюсь"}
    await likes.step_safely()
    await likes.step_safely()
    assert bots["лайки"].ratings == []
    assert "Не уверен — решите вы" in inbox.messages[-1]
    likes.llm.decide = lambda t, img: {"action": "click", "button_id": "r0c0", "confidence": 0.9}
    await say(sup, "ок лайки")
    await likes.step_safely()
    assert bots["лайки"].ratings == ["1", "1"]


async def test_pause_with_mistyped_bot_containing_digits_is_rejected(tmp_path):
    sup, agents, bots, inbox, clock = setup(tmp_path)
    await say(sup, "пауза 2ч арт3")
    assert agents["арт"].active() and agents["лайки"].active()
    await say(sup, "пауза 30 м")
    assert not agents["арт"].active() and not agents["лайки"].active()


async def test_nan_confidence_still_escalates_in_auto(tmp_path):
    sup, agents, bots, inbox, clock = setup(tmp_path)
    likes = agents["лайки"]
    likes.b = replace(likes.b, mode="smart", confidence_threshold=0.75)
    likes.llm.decide = lambda t, img: {"action": "click", "button_id": "r0c0", "confidence": "NaN"}
    await likes.step_safely()
    await likes.step_safely()
    assert bots["лайки"].ratings == []


async def test_billed_failures_survive_cancellation(tmp_path):
    import asyncio

    from doppelgram.llm import FailedCall

    sup, agents, bots, inbox, clock = setup(tmp_path)
    likes = agents["лайки"]
    await likes.step_safely()  # /start

    class SlowFallback:
        async def complete_json(self, messages, failures=None):
            failures.append(FailedCall("primary", "bad json", 1, 1, 0.25))
            await asyncio.sleep(3600)  # the fallback is still answering when we get cancelled

    likes.llm = SlowFallback()
    task = asyncio.create_task(likes.step())
    await asyncio.sleep(0.01)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert likes.store.spend_today(clock()) == 0.25


async def test_invalid_arguments_are_rejected_even_with_a_bot_name(tmp_path):
    sup, agents, bots, inbox, clock = setup(tmp_path)
    await say(sup, "пауза 2ч арт3 арт")
    assert agents["арт"].active()
    assert "Не понял «2ч арт3»" in inbox.messages[-1]
    await say(sup, "пауза 30 м арт")
    assert not agents["арт"].active() and agents["лайки"].active()


async def test_conversation_sanitizer_for_removed_bot_hides_digits(tmp_path):
    sup, agents, bots, inbox, clock = setup(tmp_path)
    for a in agents.values():
        a.b = replace(a.b, age_in_prompt="18", age_pattern=r"^[^,\n]+,\s*(\d{1,2}),")
    assert sup._sanitize_for_bot("арт", "Маша, 23, Москва") == "Маша, 18, Москва"
    assert "23" not in sup._sanitize_for_bot("удалённый", "Маша, 23, Москва")
    for a in agents.values():
        a.b = replace(a.b, age_in_prompt="keep")
    assert sup._sanitize_for_bot("удалённый", "Маша, 23, Москва") == "Маша, 23, Москва"
