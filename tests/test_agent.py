from __future__ import annotations

import random

import pytest
from fakes import Clock, FakeRatingBot, ScriptedLLM, make_config, rate_or_wait

from doppelgram.agent import Agent
from doppelgram.models import OwnerMessage
from doppelgram.storage import Store


def build(tmp_path, bot=None, decide=rate_or_wait, **behavior):
    cfg = make_config(tmp_path, **behavior)
    bot = bot or FakeRatingBot()
    llm = ScriptedLLM(decide)
    clock = Clock()
    store = Store(":memory:")
    agent = Agent(cfg, bot, llm, store, clock=clock, sleep=clock.sleep, rng=random.Random(0))
    return agent, bot, llm, clock, store


async def run_steps(agent, n):
    for _ in range(n):
        await agent.step_safely()


async def owner(agent, text, reply_to=None):
    """Commands get a slash; a reply to a card is sent as typed (that is how the owner answers)."""
    if not reply_to and not text.startswith("/"):
        text = "/" + text
    await agent.handle_owner(OwnerMessage(id=1, chat_id=42, text=text, reply_to=reply_to))


async def test_auto_mode_rates_cards_edited_in_place_and_then_waits(tmp_path):
    agent, bot, llm, clock, store = build(tmp_path)
    await run_steps(agent, 10)

    assert bot.sent == ["/start"]
    assert bot.ratings == ["4", "4", "4"]  # one click per card, no double clicks
    assert len(llm.calls) == 4  # 3 cards + the "come back later" message
    status = store.handled_status(bot.messages[-1].id, bot.messages[-1].revision, clock())
    assert status == "waiting"
    # The url button is listed for the model but marked as not clickable.
    assert '"Сайт"  [not clickable]' in llm.calls[0][1]["content"][0]["text"]
    # The callback answer toast is fed back into the next observation.
    assert "Оценка принята" in llm.calls[1][1]["content"][0]["text"]


async def test_wait_is_re_evaluated_after_its_deadline(tmp_path):
    agent, bot, llm, clock, store = build(tmp_path)
    await run_steps(agent, 10)
    calls = len(llm.calls)
    clock.t += 3 * 3600 + 1
    await agent.step_safely()
    assert len(llm.calls) == calls + 1


async def test_supervised_mode_owner_chooses_and_example_is_recorded(tmp_path):
    agent, bot, llm, clock, store = build(tmp_path, mode="supervised")
    await run_steps(agent, 3)

    assert bot.ratings == []  # nothing clicked without the owner
    text, with_image = bot.owner_inbox[-1]
    assert with_image and "Я бы выбрал: <b>4</b>" in text and "девушка с мечом" in text
    assert len(llm.calls) == 1  # sending /start needs no model; pending blocks re-deciding

    await owner(agent, "2 слишком мутно", reply_to=5000 + len(bot.owner_inbox))
    assert bot.ratings == ["2"]
    ex = store.recent_examples(10)
    assert ex == [{"description": "фото: портрет; лицо: видно", "model_choice": "4", "owner_choice": "2",
                   "note": "слишком мутно"}]

    await run_steps(agent, 1)  # next card goes to the owner again
    await owner(agent, "ок")
    assert bot.ratings == ["2", "4"]
    assert store.example_stats() == (2, 1)
    # Examples are part of the next prompt.
    await run_steps(agent, 1)
    assert 'owner chose "2" (you proposed "4"); owner\'s note: слишком мутно' in llm.calls[-1][1]["content"][0]["text"]


async def test_switching_to_auto_releases_pending_card(tmp_path):
    agent, bot, llm, clock, store = build(tmp_path, mode="supervised")
    await run_steps(agent, 3)
    assert bot.ratings == []
    await owner(agent, "режим авто")
    await run_steps(agent, 5)
    assert bot.ratings == ["4", "4", "4"]


async def test_stale_pending_card_is_not_clicked(tmp_path):
    agent, bot, llm, clock, store = build(tmp_path, mode="supervised")
    await run_steps(agent, 3)
    # The owner rates the card manually from the phone meanwhile: the bot edits the message.
    await bot.click(bot.messages[-1].id, bot.messages[-1].buttons[0])
    await owner(agent, "3")
    assert bot.ratings == ["1"]
    assert "уже сменилась" in bot.owner_inbox[-1][0]


async def test_stop_start_and_pause(tmp_path):
    agent, bot, llm, clock, store = build(tmp_path)
    await owner(agent, "стоп")
    assert not agent.active()
    await owner(agent, "старт")
    assert agent.active()
    await owner(agent, "пауза 30м")
    assert not agent.active()
    clock.t += 30 * 60 + 1
    assert agent.active()
    await owner(agent, "статус")
    assert "Состояние: <b>работает</b>" in bot.owner_inbox[-1][0]


async def test_ask_owner_stops_until_start(tmp_path):
    agent, bot, llm, clock, store = build(tmp_path, mode="supervised", decide=lambda t, img: {
        "action": "ask_owner", "reason": "капча"})
    await run_steps(agent, 3)
    assert not agent.running
    assert "капча" in bot.owner_inbox[-1][0]
    await owner(agent, "старт")
    await run_steps(agent, 1)
    assert len(llm.calls) == 2  # decision, then re-decision after "старт"


async def test_invalid_model_button_becomes_ask_owner(tmp_path):
    agent, bot, llm, clock, store = build(tmp_path, mode="supervised", decide=lambda t, img: {
        "action": "click", "button_id": "r1c0"})
    await run_steps(agent, 3)
    assert bot.ratings == []
    assert "нельзя нажать автоматически" in bot.owner_inbox[-1][0]


async def test_companion_image_from_previous_message(tmp_path):
    agent, bot, llm, clock, store = build(tmp_path, bot=FakeRatingBot(separate_keyboard=True))
    await run_steps(agent, 2)
    first_card = llm.calls[0]
    assert len(first_card[1]["content"]) == 2
    assert "from the previous bot message" in first_card[1]["content"][0]["text"]
    assert bot.ratings == ["4"]


async def test_budget_pauses_until_next_day(tmp_path):
    agent, bot, llm, clock, store = build(tmp_path, daily_budget_usd=0.0015)
    await run_steps(agent, 10)
    assert len(bot.ratings) == 2  # 0.001 per call, the third call would exceed the budget
    assert "бюджет исчерпан" in bot.owner_inbox[-1][0]
    assert not agent.active()


async def test_llm_failures_pause_then_stop(tmp_path):
    agent, bot, llm, clock, store = build(tmp_path)
    await run_steps(agent, 1)  # sends /start
    llm.fail_next = 100
    for _ in range(5):
        clock.t = max(clock.t, agent.paused_until) + 1
        await agent.step_safely()
    assert not agent.running
    assert "остановился" in bot.owner_inbox[-1][0]


async def test_no_reaction_retries_once_then_stops(tmp_path):
    bot = FakeRatingBot()
    bot.timeout_clicks = 2
    agent, bot, llm, clock, store = build(tmp_path, bot=bot, stuck_timeout_s=60)
    await run_steps(agent, 3)
    assert bot.ratings == []
    clock.t += 61
    await run_steps(agent, 2)  # retry once
    assert "no visible reaction" in llm.calls[-1][1]["content"][0]["text"]
    clock.t += 61
    await run_steps(agent, 2)
    assert not agent.running
    assert "не реагирует" in bot.owner_inbox[-1][0]


async def test_saved_messages_ignores_chatter_and_own_replies(tmp_path):
    agent, bot, llm, clock, store = build(tmp_path)
    # Notes (no slash) are never commands, not even a bare "стоп".
    await agent.handle_owner(OwnerMessage(1, 7, "купить молоко"), from_saved_messages=True)
    await agent.handle_owner(OwnerMessage(2, 7, "стоп"), from_saved_messages=True)
    await agent.handle_owner(OwnerMessage(3, 7, "Состояние: работает"), from_saved_messages=True)
    assert bot.owner_inbox == [] and agent.running
    await agent.handle_owner(OwnerMessage(4, 7, "/стоп"), from_saved_messages=True)
    assert not agent.running


@pytest.mark.parametrize("answer", ["0", "9"])
async def test_out_of_range_choice(tmp_path, answer):
    agent, bot, llm, clock, store = build(tmp_path, mode="supervised")
    await run_steps(agent, 3)
    await owner(agent, answer)
    assert bot.ratings == []
    assert "номер от 1 до" in bot.owner_inbox[-1][0]


async def test_run_forever_is_driven_by_bot_events(tmp_path):
    import asyncio
    import time

    cfg = make_config(tmp_path)
    bot = FakeRatingBot()
    store = Store(":memory:")
    agent = Agent(cfg, bot, ScriptedLLM(rate_or_wait), store, clock=time.time, rng=random.Random(0))
    original_click, original_send = bot.click, bot.send_to_bot

    async def click(msg_id, button):  # the real transport wakes the agent via MessageEdited
        out = await original_click(msg_id, button)
        agent.on_bot_event()
        return out

    async def send(text):
        await original_send(text)
        agent.on_bot_event()

    bot.click, bot.send_to_bot = click, send
    task = asyncio.create_task(agent.run_forever())
    agent.wake.set()
    try:
        for _ in range(200):
            if len(bot.ratings) == 3:
                break
            await asyncio.sleep(0.01)
    finally:
        task.cancel()
    assert bot.ratings == ["4", "4", "4"]
    assert bot.owner_inbox[0][0].startswith("<b>Doppelgram запущен.</b>")


async def test_concurrent_owner_answers_click_once(tmp_path):
    import asyncio

    bot = FakeRatingBot()
    agent, bot, llm, clock, store = build(tmp_path, bot=bot, mode="supervised")
    await run_steps(agent, 3)
    bot.timeout_clicks = 1  # the bot does not edit the card after the first press
    await asyncio.gather(owner(agent, "ок"), owner(agent, "ок"))
    assert store.actions_since(0, ("click",)) == 1
    assert "уже ответили" in bot.owner_inbox[-1][0]


# ---- regressions from the independent review ---------------------------------------------------
class SlowSleepClock(Clock):
    """Lets other tasks run during the pre-action delay, like the real asyncio.sleep."""

    async def sleep(self, s: float) -> None:
        import asyncio
        self.t += s
        for _ in range(5):
            await asyncio.sleep(0)


def build_slow(tmp_path, **behavior):
    cfg = make_config(tmp_path, **behavior)
    bot = FakeRatingBot()
    llm = ScriptedLLM(rate_or_wait)
    clock = SlowSleepClock()
    store = Store(":memory:")
    agent = Agent(cfg, bot, llm, store, clock=clock, sleep=clock.sleep, rng=random.Random(0))
    return agent, bot, llm, clock, store


@pytest.mark.parametrize("command", ["стоп", "пауза 1ч", "режим учёба"])
async def test_owner_command_during_delay_cancels_prepared_click(tmp_path, command):
    import asyncio

    agent, bot, llm, clock, store = build_slow(tmp_path)
    await agent.step_safely()  # /start
    step = asyncio.create_task(agent.step_safely())
    for _ in range(1000):  # bounded: a broken step must fail the test, not hang it
        if llm.calls:
            break
        await asyncio.sleep(0)
    assert llm.calls
    await owner(agent, command)
    await step
    assert bot.ratings == []


async def test_mode_switch_while_card_is_sent_does_not_leave_it_pending(tmp_path):
    agent, bot, llm, clock, store = build(tmp_path, mode="supervised")
    await run_steps(agent, 1)
    original = bot.notify_owner

    async def notify(text, image=None):  # the owner switches to auto while the card is in flight
        if text.startswith("<b>Новая анкета</b>"):
            await owner(agent, "режим авто")
        return await original(text, image)

    bot.notify_owner = notify
    await run_steps(agent, 1)
    bot.notify_owner = original
    assert store.latest_pending() is None
    await run_steps(agent, 4)
    assert bot.ratings == ["4", "4", "4"]


async def test_reply_to_stale_notification_does_not_rate_current_card(tmp_path):
    agent, bot, llm, clock, store = build(tmp_path, mode="supervised")
    await run_steps(agent, 2)
    first_card_ctrl = 5000 + len(bot.owner_inbox)
    await owner(agent, "3", reply_to=first_card_ctrl)
    await run_steps(agent, 1)  # next card is sent to the owner
    inbox = len(bot.owner_inbox)
    await owner(agent, "3", reply_to=first_card_ctrl)  # repeated answer to the OLD card: ignored
    assert bot.ratings == ["3"] and len(bot.owner_inbox) == inbox


async def test_new_picture_with_same_keyboard_is_a_new_card(tmp_path):
    from dataclasses import replace

    bot = FakeRatingBot(separate_keyboard=True)
    bot.timeout_clicks = 1  # the press is accepted but the keyboard message stays as it was
    agent, bot, llm, clock, store = build(tmp_path, bot=bot)
    await run_steps(agent, 3)
    assert len(llm.calls) == 1
    # The bot replaces the picture message but keeps the keyboard message unchanged.
    bot.messages[-2] = replace(bot.messages[-2], media_key="p-new")
    await run_steps(agent, 1)
    assert len(llm.calls) == 2


async def test_click_is_refused_when_a_newer_card_arrived(tmp_path):
    from doppelgram.models import BotMessage

    agent, bot, llm, clock, store = build(tmp_path)
    await run_steps(agent, 1)
    original = llm.decide

    def decide_and_bot_sends_new_card(text, img):
        bot.messages.append(BotMessage(id=999, text="Новая карточка", buttons=bot.messages[-1].buttons,
                                       media_kind="photo", media_key="x"))
        llm.decide = original
        return original(text, img)

    llm.decide = decide_and_bot_sends_new_card
    old_id = bot.messages[-1].id
    clicked = []
    orig_click = bot.click

    async def click(msg_id, button):
        clicked.append(msg_id)
        return await orig_click(msg_id, button)

    bot.click = click
    await run_steps(agent, 1)
    assert old_id not in clicked


async def test_supervised_needs_owner_even_if_picture_fails_to_load(tmp_path):
    agent, bot, llm, clock, store = build(tmp_path, mode="supervised", decide=lambda t, img: {
        "action": "click", "button_id": "r0c3", "reason": "по тексту"})

    async def broken_download(msg_id):
        return None

    bot.download_image = broken_download
    await run_steps(agent, 3)
    assert bot.ratings == []
    assert "картинку загрузить не удалось" in bot.owner_inbox[-1][0]


async def test_restriction_during_owner_answer_stops_everything(tmp_path):
    from doppelgram.transport import AccountRestricted

    agent, bot, llm, clock, store = build(tmp_path, mode="supervised")
    await run_steps(agent, 2)

    async def restricted_click(msg_id, button):
        raise AccountRestricted("FrozenMethodInvalidError")

    bot.click = restricted_click
    await owner(agent, "ок")
    assert agent.restricted and not agent.running
    await owner(agent, "ок")
    assert "Аккаунт ограничен" in bot.owner_inbox[-1][0]


async def test_flood_wait_during_owner_answer_pauses(tmp_path):
    from doppelgram.transport import RateLimited

    agent, bot, llm, clock, store = build(tmp_path, mode="supervised")
    await run_steps(agent, 2)

    async def flood_click(msg_id, button):
        raise RateLimited(3600)

    bot.click = flood_click
    await owner(agent, "ок")
    assert not agent.active()
    await owner(agent, "ок")
    assert "просит подождать" in bot.owner_inbox[-1][0]
