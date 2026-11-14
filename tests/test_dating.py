from __future__ import annotations

import random
import re
from dataclasses import replace
from pathlib import Path

import pytest
from fakes import Clock, ScriptedLLM, make_config, png_bytes
from fakes_dating import FakeBibinto, FakeLeomatch, reply_kb

from doppelgram.agent import Agent
from doppelgram.models import BotMessage, OwnerMessage
from doppelgram.storage import Store
from doppelgram.supervisor import Supervisor
from doppelgram.taste import TasteProfile

LEO = dict(deny_buttons=["💌", "📹", "⭐", "VIP"], match_pattern="Начинай общаться",
           min_age=18, age_pattern=r"^[^,\n]+,\s*(\d{1,2}),", skip_buttons=["👎"])
BIB = dict(deny_buttons=["💌", "💘", "⚠️"], min_age=18, age_pattern=r"(\d{1,2})\s*лет", skip_buttons=["Назад"])


def prompt_text(call) -> str:
    return call[1]["content"][0]["text"]


def button_id(call, text: str) -> str:
    m = re.search(rf'- (\S+) \(\w+ \w+\): "{re.escape(text)}"', prompt_text(call))
    assert m, f"button {text!r} not offered"
    return m.group(1)


def leo_decide(text, has_image):
    if "Анкеты закончились" in text:
        return {"action": "wait", "wait_seconds": 3600}
    if not has_image:
        return {"action": "ask_owner", "reason": "?"}
    like = "Маша" in text.split("## Current bot message")[1]
    return {"action": "click", "button_id": "k0c0" if like else "k0c2", "confidence": 0.9,
            "description": "девушка в горах"}


def build(tmp_path, bot, decide, behavior, mode="auto", taste=None):
    cfg = make_config(tmp_path)
    clock = Clock()
    agent = Agent(cfg, bot, ScriptedLLM(decide), Store(":memory:"), clock=clock, sleep=clock.sleep,
                  rng=random.Random(0), behavior=replace(cfg.behavior, mode=mode, **behavior), taste=taste)
    return agent, clock


async def run(agent, n):
    for _ in range(n):
        await agent.step_safely()


async def test_leomatch_swipes_with_reply_keyboard_and_records_match(tmp_path):
    bot = FakeLeomatch()
    agent, clock = build(tmp_path, bot, leo_decide, LEO)
    await run(agent, 8)
    assert bot.reactions == [("Аня", "👎"), ("Маша", "❤️"), ("Лиза", "👎"), ("Катя", "👎")]
    # Лиза is 17: skipped by the age rule, the model never saw her card.
    assert not any("Лиза, 17" in prompt_text(c).split("## Current bot message")[1] for c in agent.llm.calls)
    # The keyboard is offered with blocked paid/message buttons.
    first = prompt_text(agent.llm.calls[0])
    assert '"❤️"' in first and '"💌 / 📹"  [blocked]' in first
    # The match was recorded and the owner was told how to reach the person.
    assert agent.store.matches()[0]["links"] == ["https://t.me/masha_example"]
    assert any("Взаимная симпатия" in t and "masha_example" in t for t, _ in bot.owner_inbox)


async def test_age_gate_does_not_loop_on_the_same_card(tmp_path):
    bot = FakeLeomatch(cards=[("Лиза", 16)])
    agent, clock = build(tmp_path, bot, leo_decide, LEO)
    await run(agent, 1)
    original = bot.send_to_bot

    async def no_progress(text):
        bot.sent.append(text)

    bot.send_to_bot = no_progress
    await run(agent, 1)
    assert bot.sent[-1] == "👎"
    # The bot shows the same profile again as a new message: never loop, ask the owner.
    card = next(m for m in reversed(bot.messages) if m.has_image)
    bot.messages.append(replace(card, id=bot._id()))
    await run(agent, 1)
    assert not agent.running
    assert "16 &lt; 18" in bot.owner_inbox[-1][0]
    bot.send_to_bot = original


async def test_bibinto_rates_with_keyboard_sent_before_the_picture(tmp_path):
    bot = FakeBibinto()

    def decide(text, has_image):
        if "Коала накушалась" in text:
            return {"action": "wait", "wait_seconds": 21600}
        return {"action": "click", "button_id": "k1c1", "confidence": 0.8}  # "7"

    agent, clock = build(tmp_path, bot, decide, BIB)
    await run(agent, 8)
    assert bot.ratings == [("Аня", "7"), ("Оля", "7"), ("Вика", "7")]
    first = agent.llm.calls[0]
    assert len(first[1]["content"]) == 2  # the picture was attached
    text = prompt_text(first)
    assert '"💌Сообщение"  [blocked]' in text and '"⚠️Жалоба"  [blocked]' in text and '"7"' in text
    assert len(agent.llm.calls) == 4  # three cards + the limit message; edits of old pictures are ignored


async def test_model_cannot_press_blocked_button(tmp_path):
    bot = FakeBibinto()
    agent, clock = build(tmp_path, bot, lambda t, i: {"action": "click", "button_id": "k3c0"}, BIB, mode="supervised")
    await run(agent, 2)
    assert bot.ratings == [] and "💌Сообщение" not in bot.sent
    assert "нельзя нажать автоматически" in bot.owner_inbox[-1][0]


async def test_taste_rules_and_reference_photos_reach_the_model(tmp_path):
    taste_file = tmp_path / "my-taste.md"
    taste_file.write_text("Нравятся спортивные.", encoding="utf-8")
    profile = TasteProfile("default", tmp_path, taste_file)
    profile.add_rule("не нравятся полные")
    profile.add_reference(png_bytes((10, 200, 10)), liked=True, note="улыбка")
    profile.add_reference(png_bytes((200, 10, 10)), liked=False)
    bot = FakeLeomatch()
    agent, clock = build(tmp_path, bot, leo_decide, LEO, taste=profile)
    await run(agent, 2)
    content = agent.llm.calls[0][1]["content"]
    text = content[0]["text"]
    assert "Нравятся спортивные." in text and "- не нравятся полные" in text
    labels = [c["text"] for c in content[1:] if c["type"] == "text"]
    assert labels == ["Reference 1: owner LIKES this (улыбка)", "Reference 2: owner DISLIKES this",
                      "Current card image:"]
    assert sum(c["type"] == "image_url" for c in content) == 3


async def test_owner_manages_taste_from_telegram(tmp_path):
    profile = TasteProfile("default", tmp_path, None)
    bot = FakeLeomatch()
    agent, clock = build(tmp_path, bot, leo_decide, LEO, taste=profile)
    inbox = []

    async def notify(text, image=None):
        inbox.append(text)
        return 1

    sup = Supervisor({"дайвинчик": agent}, notify, {"default": profile})

    async def say(text, image=None, saved=False):
        await sup.handle_owner(OwnerMessage(id=1, chat_id=42, text=text, image=image), from_saved_messages=saved)

    await say("/like улыбка", image=png_bytes())
    await say("/dislike", image=png_bytes())
    inbox_before = len(inbox)
    await say("", image=png_bytes())  # a photo without /like or /dislike is the owner's own
    await say("+", image=png_bytes())
    await say("отпуск 2025", image=png_bytes(), saved=True)
    assert len(inbox) == inbox_before
    assert profile.counts() == (0, 1, 1)
    await say("/rule не нравятся полные")
    await say("/taste")
    assert "№1 не нравятся полные" in inbox[-1] and "1 нравится, 1 не нравится" in inbox[-1]
    await say("/delete фото 1")
    assert profile.counts() == (1, 0, 1)
    assert not (tmp_path / "refs" / "default" / "1.jpg").exists()
    await say("/delete правило 1")
    assert profile.rules() == []


async def test_restriction_in_one_bot_stops_all_bots(tmp_path):
    from doppelgram.transport import AccountRestricted

    leo, bib = FakeLeomatch(), FakeBibinto()
    a1, clock = build(tmp_path, leo, leo_decide, LEO)
    a2, _ = build(tmp_path, bib, lambda t, i: {"action": "wait"}, BIB)
    inbox = []

    async def notify(text, image=None):
        inbox.append(text)

    Supervisor({"дайвинчик": a1, "бибинто": a2}, notify)  # wires the account-wide breaker into both agents

    async def flooded(text):
        raise AccountRestricted("PeerFloodError")

    leo.send_to_bot = flooded
    await a1.step_safely()
    assert a1.restricted and a2.restricted and not a2.active()
    assert "Все боты остановлены" in inbox[-1]


# ---- regressions from the review of the swipe commit -------------------------------------------
async def test_typing_a_blocked_button_text_is_refused(tmp_path):
    bot = FakeBibinto()
    agent, clock = build(tmp_path, bot, lambda t, i: {"action": "send", "text": "💌Сообщение"}, BIB, mode="supervised")
    await run(agent, 2)
    assert "💌Сообщение" not in bot.sent and "нельзя нажать" in bot.owner_inbox[-1][0]


async def test_typing_a_rating_goes_through_supervision(tmp_path):
    bot = FakeBibinto()
    agent, clock = build(tmp_path, bot, lambda t, i: {"action": "send", "text": "7"}, BIB, mode="supervised")
    await run(agent, 2)
    assert bot.ratings == [] and "Я бы выбрал" in bot.owner_inbox[-1][0]


async def test_edited_rated_picture_is_not_rated_again(tmp_path):
    bot = FakeBibinto(people=["Аня"])
    agent, clock = build(tmp_path, bot, lambda t, i: {"action": "click", "button_id": "k1c1"}, BIB)
    await run(agent, 2)
    # The next card is delayed: only the edited old picture and the old keyboard are on screen.
    bot.messages = [m for m in bot.messages if "Коала" not in m.text]
    await run(agent, 3)
    assert bot.ratings == [("Аня", "7")]


async def test_restriction_during_owner_confirmed_click_blocks_it(tmp_path):
    bot = FakeBibinto()
    agent, clock = build(tmp_path, bot, lambda t, i: {"action": "click", "button_id": "k1c1"}, BIB,
                         mode="supervised")
    await run(agent, 2)
    original_sleep = agent.sleep

    async def sleep_then_restrict(s):
        agent.restrict("PeerFloodError")
        await original_sleep(s)

    agent.sleep = sleep_then_restrict
    await agent.handle_owner(OwnerMessage(id=1, chat_id=42, text="ок"))
    assert bot.ratings == []


async def test_oversized_photo_caption_is_not_a_command(tmp_path):
    bot = FakeBibinto()
    agent, clock = build(tmp_path, bot, lambda t, i: {"action": "click", "button_id": "k1c1"}, BIB,
                         mode="supervised")
    sup = Supervisor({"бибинто": agent}, bot.notify_owner, {"default": TasteProfile("default", tmp_path, None)})
    await run(agent, 2)
    inbox = len(bot.owner_inbox)
    await sup.handle_owner(OwnerMessage(id=1, chat_id=42, text="+", has_media=True))
    assert bot.ratings == [] and len(bot.owner_inbox) == inbox  # "+" on an attachment is not an answer
    await sup.handle_owner(OwnerMessage(id=2, chat_id=42, text="/like", has_media=True))
    assert bot.ratings == [] and "до 10 МБ" in bot.owner_inbox[-1][0]


async def test_age_gate_respects_the_hourly_cap(tmp_path):
    bot = FakeLeomatch(cards=[("Лиза", 16), ("Оля", 15), ("Катя", 14)])
    agent, clock = build(tmp_path, bot, leo_decide, dict(LEO, max_actions_per_hour=2))  # /start + one skip
    await run(agent, 6)
    assert len(bot.reactions) == 1


async def test_restriction_while_notifying_trips_the_breaker(tmp_path):
    from doppelgram.transport import AccountRestricted

    bot = FakeLeomatch()
    agent, clock = build(tmp_path, bot, leo_decide, LEO)
    tripped = []

    async def breaker(reason):
        tripped.append(reason)

    agent.on_restricted = breaker
    await run(agent, 2)  # /start, rate Аня

    async def restricted_notify(text, image=None):
        raise AccountRestricted("PeerFloodError")

    bot.notify_owner = restricted_notify
    reactions = len(bot.reactions)
    await run(agent, 3)  # the match notice for Маша fails to reach the owner
    assert tripped == ["PeerFloodError"] and agent.restricted
    assert len(bot.reactions) <= reactions + 1


async def test_new_caption_on_same_picture_is_a_new_card(tmp_path):
    bot = FakeBibinto(people=["Аня"])
    agent, clock = build(tmp_path, bot, lambda t, i: {"action": "click", "button_id": "k1c1"}, BIB)
    await run(agent, 2)
    calls = len(agent.llm.calls)
    bot.messages = [m for m in bot.messages if "Коала" not in m.text]
    pic = max((m for m in bot.messages if m.has_image), key=lambda m: m.id)
    bot.messages[bot.messages.index(pic)] = replace(pic, text="👩‍🦰: Оля, 24 лет\n🌇Город: Казань", buttons=())
    await run(agent, 1)
    assert len(agent.llm.calls) == calls + 1


async def test_changed_separate_text_with_same_picture_is_decided(tmp_path):
    from fakes_dating import _Base

    bot = _Base()
    bot.messages = [BotMessage(id=1, text="", media_kind="photo", media_key="p1", date=0),
                    BotMessage(id=2, text="Оцени", date=1, buttons=reply_kb([["1", "2", "3"]]))]

    async def click(msg_id, button):
        from doppelgram.models import ClickOutcome
        return ClickOutcome(ok=True)  # the bot does not move on

    bot.click = click
    agent, clock = build(tmp_path, bot, lambda t, i: {"action": "click", "button_id": "k0c1"}, {})
    await run(agent, 1)
    assert len(agent.llm.calls) == 1
    # The picture stays; the separate keyboard message is edited into a new question: decide again.
    bot.messages[1] = replace(bot.messages[1], text="А теперь эту?", edit_date=2.0)
    await run(agent, 1)
    assert len(agent.llm.calls) == 2


@pytest.mark.parametrize("mode,shown,hidden", [("18", "Аня, 18, Москва", "21"), ("hide", "Аня, Москва", "21"),
                                               ("keep", "Аня, 21, Москва", None)])
async def test_ages_sent_to_the_model_follow_the_setting(tmp_path, mode, shown, hidden):
    bot = FakeLeomatch()
    agent, clock = build(tmp_path, bot, leo_decide, dict(LEO, min_age=0, age_in_prompt=mode))
    await run(agent, 4)  # /start, Аня, Маша (like → match), next card (the match is recorded)
    second = prompt_text(agent.llm.calls[1])  # Маша's card, with Аня's card in the history
    current = second.split("## Current bot message")[1]
    assert ("Маша, 18" in current) if mode == "18" else True
    assert shown in second.split("## Current bot message")[0]  # history is sanitized too
    if hidden:
        assert hidden not in second and "23" not in current
    # The owner still sees the real profile text in the match notice.
    assert any("Анкета: Маша, 23" in t for t, _ in bot.owner_inbox)


# ---- privacy: nothing about the people in the bots is kept beyond what the agent needs -----------
async def test_profiles_are_not_stored(tmp_path):
    bot = FakeLeomatch()

    def decide(text, has_image):
        r = leo_decide(text, has_image)
        r.update(reason="Маша, 23 года, стройная", description="девушка 23 лет в горах")
        return r

    agent, clock = build(tmp_path, bot, decide, dict(LEO, min_age=0))
    await run(agent, 6)
    dump = "\n".join(str(row) for table in ("kv", "actions", "examples", "matches", "handled")
                     for row in agent.store.db.execute(f"SELECT * FROM {table}"))
    for leaked in ("Москва", "люблю горы", "23 года", "Маша, 23", "Аня, 21", "Начинай общаться"):
        assert leaked not in dump, leaked
    assert agent.store.matches()[0]["links"] == ["https://t.me/masha_example"]


async def test_status_counts_ratings_separately_from_menu_presses(tmp_path):
    bot = FakeBibinto(people=["Аня"])
    agent, clock = build(tmp_path, bot, lambda t, i: {"action": "click", "button_id": "k1c1"}, BIB)
    await run(agent, 2)
    pic = next(m for m in bot.messages if not m.outgoing and "🐨" in m.text)
    agent.store.log_action(clock(), pic.id, "x", "click", button_text="🚀Оценивать")  # a menu press
    counts = agent.store.counts_today(clock())
    assert (counts["rated"], counts["clicks"]) == (1, 2)
    assert "<b>1</b> оценка, <b>1</b> нажатие в меню, <b>1</b> запрос к модели" in agent.status_text()


def test_one_time_scrub_of_old_data(tmp_path):

    path = tmp_path / "old.sqlite3"
    s = Store(path)
    s.db.execute("DELETE FROM kv WHERE key='privacy_scrub'")
    s.set("acted:1:p", "Аня, 21, Москва – люблю горы")
    s.db.execute("INSERT INTO actions(ts, action, reason) VALUES(0, 'click', 'Аня, 21 — красивая')")
    s.db.execute("INSERT INTO matches(ts, bot_msg_id, text, links, mention_ids) "
                 "VALUES(0, 1, 'Начинай 👉 Аня', '[]', '[]')")
    s.db.execute("INSERT INTO examples(ts, description, owner_choice) VALUES(0, 'девушка 21 года', '7')")
    s.db.execute("INSERT INTO actions(ts, action, text) VALUES(0, 'send', 'Юля')")
    s.db.execute("INSERT INTO actions(ts, action, text) VALUES(0, 'send', '/search Анна 23')")
    s.db.close()
    s = Store(path)
    assert s.get("acted:1:p") is None
    assert s.db.execute("SELECT reason FROM actions").fetchone() == ("",)
    assert s.db.execute("SELECT text FROM matches").fetchone() == ("",)
    assert s.db.execute("SELECT description FROM examples").fetchone() == ("",)  # free text never kept
    assert [r[0] for r in s.db.execute("SELECT text FROM actions WHERE action='send' ORDER BY id")] == [
        "(сообщение)", "/search"]


def test_template_comments_are_never_shown_to_the_model(tmp_path):
    from doppelgram.storage import read_user_text

    root = Path(__file__).parent.parent
    for name in ("taste.example.md", "about-me.example.md", "style.example.md"):
        assert read_user_text(root / name) == "", name  # templates contain no active facts or tastes
    f = tmp_path / "t.md"
    f.write_text("# пример: нравятся блондинки\n\nНравятся брюнетки\n# Не нравятся\n#фитнес тоже ок\n",
                 encoding="utf-8")
    # Only the leading comment block is hidden; headings and hashtags further down stay.
    assert read_user_text(f) == "Нравятся брюнетки\n# Не нравятся\n#фитнес тоже ок"


def test_traits_keep_only_the_closed_vocabulary():
    from doppelgram.brain import clean_traits, traits_text

    t = clean_traits({"телосложение": "Стройное", "волосы": "светлые", "имя": "Аня", "лицо": "Аня, 21 год",
                      "стиль": "как у Ани из Москвы"})
    assert t == {"телосложение": "стройное", "волосы": "светлые"}
    assert traits_text(t) == "телосложение: стройное; волосы: светлые"
    assert clean_traits("Аня, 21") == {} and clean_traits(None) == {}


async def test_pending_card_and_bot_answers_are_stored_without_personal_text(tmp_path):
    bot = FakeLeomatch()
    agent, clock = build(tmp_path, bot, lambda t, i: {
        "action": "click", "button_id": "k0c0", "confidence": 0.9, "reason": "Аня, 21, классная",
        "description": "Аня, 21 год, блондинка из Москвы", "traits": {"волосы": "светлые", "имя": "Аня"}},
        dict(LEO, min_age=0), mode="supervised")
    await run(agent, 2)
    import json

    decision = json.loads(agent.store.db.execute("SELECT decision FROM pending").fetchone()[0])
    stored = json.dumps(decision, ensure_ascii=False)
    for leaked in ("21", "Аня", "классная", "Москвы", "блондинка"):
        assert leaked not in stored, leaked
    assert decision["description"] == "волосы: светлые" and decision["reason"] == ""
    # The owner still sees the free description in the card in Telegram.
    assert "блондинка из Москвы" in bot.owner_inbox[-1][0]


def test_llm_parse_errors_do_not_carry_the_reply():
    from doppelgram.llm import LLMError, extract_json

    with pytest.raises(LLMError) as err:
        extract_json("Аня, 21 год, Москва — не JSON")
    assert "Аня" not in str(err.value) and "21" not in str(err.value)


def test_old_conversation_rows_are_scrubbed(tmp_path):
    from doppelgram.conversation import ConversationStore

    path = tmp_path / "conv.sqlite3"
    c = ConversationStore(path)
    c.db.execute("DELETE FROM kv WHERE key='privacy_scrub'")
    c.db.execute("INSERT INTO people(user_id, name, bot, card, age, memory, created) "
                 "VALUES(1, 'Маша', 'д', 'Маша, 23, Москва', 23, 'ей 23 года', 0)")
    c.db.close()
    c = ConversationStore(path)
    assert c.db.execute("SELECT age, card, memory FROM people").fetchone() == (None, "Маша, Москва", "ей года")


def test_typed_messages_are_stored_only_as_commands_numbers_or_labels():
    from doppelgram.agent import _loggable

    msg = BotMessage(id=1, text="", buttons=reply_kb([["❤️", "👎"]]))
    assert [_loggable(t, msg) for t in ("/start", "/search Анна 23", "1", "❤️", "Анна, 23, ветеринар", "Юля")] == [
        "/start", "/search", "1", "❤️", "(сообщение)", "(сообщение)"]


async def test_provider_errors_never_carry_the_response_body():
    import httpx

    from doppelgram.config import LLMConfig
    from doppelgram.llm import LLMClient, LLMError

    def handler(request):
        return httpx.Response(400, json={"error": {"code": "invalid_request",
                                                   "message": "bad input: 'Анна, 23, Москва'"}})

    c = LLMClient(LLMConfig(base_url="http://x/v1", api_key="", model="m"),
                  http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(LLMError) as err:
        await c.complete_json([])
    assert "Анна" not in str(err.value) and "invalid_request" in str(err.value)


def test_traits_accept_e_and_yo_and_old_free_text_never_survives(tmp_path):
    from doppelgram.brain import clean_traits, parse_traits_text

    assert clean_traits({"волосы": "темные"}) == {"волосы": "тёмные"}
    assert parse_traits_text("волосы: тёмные; лицо: видно") == {"волосы": "тёмные", "лицо": "видно"}
    assert parse_traits_text("Описание: Анне из Москвы идёт красное платье") == {}
    assert parse_traits_text("лицо: Анна из Москвы; лицо: видно") == {}

    path = tmp_path / "old.sqlite3"
    s = Store(path)
    s.db.execute("DELETE FROM kv WHERE key='privacy_scrub'")
    s.db.execute("INSERT INTO examples(ts, description, owner_choice, note) VALUES "
                 "(0, 'Описание: Анне из Москвы идёт платье', '7', 'ей 23'), (0, 'лицо: видно', '5', ''), "
                 "(0, 'лицо: Анна из Москвы; лицо: видно', '6', '')")
    s.db.close()
    s = Store(path)
    assert [r for r in s.db.execute("SELECT description, note FROM examples ORDER BY id")] == [
        ("", "ей"), ("лицо: видно", ""), ("", "")]


async def test_owner_notes_never_keep_an_age(tmp_path):
    agent, clock = build(tmp_path, FakeBibinto(), lambda t, i: {"action": "click", "button_id": "k1c1"}, BIB,
                         mode="supervised")
    await run(agent, 2)
    await agent.handle_owner(OwnerMessage(id=1, chat_id=42, text="3 ей 19, выглядит младше",
                                          reply_to=7000 + len(agent.tg.owner_inbox)))
    assert agent.store.recent_examples(1)[0]["note"] == "ей, выглядит младше"


def test_long_digit_runs_are_not_logged():
    from doppelgram.agent import _loggable

    msg = BotMessage(id=1, text="", buttons=())
    assert _loggable("89161234567", msg) == "(сообщение)" and _loggable("12", msg) == "12"


def test_store_accepts_only_traits_as_descriptions(tmp_path):
    from doppelgram.storage import Pending

    s = Store(":memory:")
    s.add_example(0, 1, "Анна из Москвы", "❤️", "❤️")
    s.add_example(0, 2, "лицо: видно", "❤️", "👎")
    s.add_pending(Pending(9, 1, "r", {"description": "Анна из Москвы", "reason": "x"}, [], 0))
    assert [e["description"] for e in s.recent_examples(5)] == ["", "лицо: видно"]
    assert s.get_pending(9).decision["description"] == "" and s.get_pending(9).decision["reason"] == ""
