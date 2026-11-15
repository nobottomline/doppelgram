"""Owner control from Telegram: slash commands, settings, modes, spot checks, paid offers, formatting."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest
from fakes import Clock, FakeRatingBot, ScriptedLLM, make_config, rate_or_wait
from fakes_dating import FakeBibinto, FakeLeomatch, reply_kb

from doppelgram.agent import Agent
from doppelgram.models import BotMessage, Button, OwnerMessage
from doppelgram.storage import Store
from doppelgram.supervisor import Supervisor


class Inbox:
    def __init__(self):
        self.messages: list[str] = []
        self.images: list[bool] = []

    async def __call__(self, text, image=None):
        self.messages.append(str(text))
        self.images.append(image is not None)
        return 9000 + len(self.messages)


def make(tmp_path, bots=None, **behavior):
    cfg = make_config(tmp_path)
    clock = Clock()
    inbox = Inbox()
    agents = {}
    for name, bot, decide in (bots or [("арт", FakeRatingBot(), rate_or_wait)]):
        bot.notify_owner = inbox
        agents[name] = Agent(cfg, bot, ScriptedLLM(decide), Store(":memory:"), clock=clock, sleep=clock.sleep,
                             rng=random.Random(0), behavior=replace(cfg.behavior, **behavior), name=name,
                             label=name if bots and len(bots) > 1 else "")

    from types import SimpleNamespace

    control = Store(":memory:")
    llm = SimpleNamespace(cfg=replace(cfg.llm, model="model-a"))
    sup = Supervisor(agents, inbox, control=control, llm=llm, allowed_models=["model-a", "model-b"])
    return sup, agents, inbox, clock, control, llm


async def send(sup, text, reply_to=None, saved=False, msg_id=1):
    await sup.handle_owner(OwnerMessage(id=msg_id, chat_id=42, text=text, reply_to=reply_to),
                           from_saved_messages=saved)


async def steps(agents, n):
    for _ in range(n):
        for a in agents.values():
            await a.step_safely()


# ------------------------------------------------------------------ commands
async def test_notes_are_ignored_and_dm_gets_a_rate_limited_hint(tmp_path):
    sup, agents, inbox, clock, *_ = make(tmp_path)
    await send(sup, "стоп", saved=True)
    await send(sup, "купить хлеб", saved=True)
    assert inbox.messages == [] and agents["арт"].running
    await send(sup, "привет")
    await send(sup, "ещё заметка")
    assert len(inbox.messages) == 1 and "/help" in inbox.messages[0]
    await send(sup, "/stop")
    assert not agents["арт"].running


async def test_start_and_stop_are_idempotent(tmp_path):
    sup, agents, inbox, *_ = make(tmp_path)
    await send(sup, "/start")
    assert "Уже работает" in inbox.messages[-1]
    await send(sup, "/stop")
    assert "Остановлен" in inbox.messages[-1]
    await send(sup, "/stop")
    assert "Уже остановлен" in inbox.messages[-1]
    await send(sup, "/старт")
    assert "Запущен" in inbox.messages[-1] and agents["арт"].running


async def test_broadcast_start_reports_already_running(tmp_path):
    bots = [("арт", FakeRatingBot(), rate_or_wait), ("лайки", FakeRatingBot(), rate_or_wait)]
    sup, agents, inbox, *_ = make(tmp_path, bots=bots)
    await send(sup, "/start")
    assert "Уже работает" in inbox.messages[-1]
    await send(sup, "/stop лайки")
    await send(sup, "/start")
    assert "Запущено" in inbox.messages[-1] and agents["лайки"].running


async def test_unknown_command_and_help(tmp_path):
    sup, agents, inbox, *_ = make(tmp_path)
    await send(sup, "/fly")
    assert "Не знаю команду" in inbox.messages[-1]
    await send(sup, "/help")
    assert "<b>Режимы</b>" in inbox.messages[-1] and "/like" in inbox.messages[-1]


# ------------------------------------------------------------------ settings
async def test_settings_are_set_listed_persisted_and_reset(tmp_path):
    sup, agents, inbox, clock, control, llm = make(tmp_path)
    art = agents["арт"]
    await send(sup, "/set порог 0.85")
    assert art.b.confidence_threshold == 0.85 and "0.85" in inbox.messages[-1]
    await send(sup, "/set задержка 5-12")
    assert (art.b.action_delay_min_s, art.b.action_delay_max_s) == (5.0, 12.0)
    await send(sup, "/settings")
    assert "порог" in inbox.messages[-1] and "из лички" in inbox.messages[-1]
    # A new process with the same database keeps the override.
    again = Agent(art.cfg, art.tg, art.llm, art.store, behavior=art._base)
    assert again.b.confidence_threshold == 0.85
    await send(sup, "/reset порог")
    assert art.b.confidence_threshold == art._base.confidence_threshold
    await send(sup, "/set порог 7")
    assert "Не подходит значение" in inbox.messages[-1]
    await send(sup, "/set небо синее")
    assert "Нет настройки" in inbox.messages[-1]


async def test_setting_a_mode_for_one_bot(tmp_path):
    bots = [("дайвинчик", FakeLeomatch(), lambda t, i: {"action": "wait"}),
            ("бибинто", FakeBibinto(), lambda t, i: {"action": "wait"})]
    sup, agents, inbox, *_ = make(tmp_path, bots=bots, mode="supervised")
    await send(sup, "/set бибинто режим авто")
    assert agents["бибинто"].mode == "auto" and agents["дайвинчик"].mode == "supervised"
    await send(sup, "/mode умный дайвинчик")
    assert agents["дайвинчик"].mode == "smart"
    await send(sup, "/reset all")
    assert agents["бибинто"].mode == "supervised" and agents["дайвинчик"].mode == "supervised"


async def test_model_switch_only_to_allowed_models_and_persisted(tmp_path):
    sup, agents, inbox, clock, control, llm = make(tmp_path)
    await send(sup, "/set модель evil/model")
    assert "не разрешена" in inbox.messages[-1] and llm.cfg.model == "model-a"
    await send(sup, "/set модель model-b")
    assert llm.cfg.model == "model-b"
    Supervisor(agents, inbox, control=control, llm=llm, allowed_models=["model-a", "model-b"])
    assert llm.cfg.model == "model-b"
    await send(sup, "/reset модель")
    assert llm.cfg.model == "model-a"


async def test_legacy_mode_value_is_migrated_to_an_override(tmp_path):
    cfg = make_config(tmp_path)
    store = Store(":memory:")
    store.set("mode", "auto")
    agent = Agent(cfg, FakeRatingBot(), ScriptedLLM(rate_or_wait), store,
                  behavior=replace(cfg.behavior, mode="supervised"))
    assert agent.mode == "auto" and store.get("mode") is None


# ------------------------------------------------------------------ modes
def unsure(text, has_image):
    return {"action": "click", "button_id": "r0c3", "confidence": 0.5, "reason": "сомневаюсь"}


async def test_smart_asks_only_when_unsure_and_auto_never_asks(tmp_path):
    sup, agents, inbox, *_ = make(tmp_path, bots=[("арт", FakeRatingBot(), unsure)], mode="smart",
                                  confidence_threshold=0.7, spot_check_rate=0)
    await steps(agents, 2)
    assert agents["арт"].tg.ratings == [] and "Не уверен" in inbox.messages[-1]
    await send(sup, "/mode авто")
    await steps(agents, 3)
    assert agents["арт"].tg.ratings == ["4", "4", "4"]


async def test_smart_timeout_applies_own_choice_only_when_enabled(tmp_path):
    sup, agents, inbox, clock, *_ = make(tmp_path, bots=[("арт", FakeRatingBot(), unsure)], mode="smart",
                                         confidence_threshold=0.7, ask_timeout_min=0, spot_check_rate=0)
    art = agents["арт"]
    await steps(agents, 2)
    clock.t += 10_000
    await steps(agents, 1)
    assert art.tg.ratings == []  # 0 = wait for the owner
    await send(sup, "/set ожидание 30")
    await steps(agents, 1)  # the setting change re-arms nothing by itself; the next card would wait 30 min
    clock.t += 31 * 60
    await steps(agents, 1)
    assert art.tg.ratings == ["4"]
    assert "Если не ответите за 30 мин" not in inbox.messages[0]


# ------------------------------------------------------------------ spot checks
def sure(text, has_image):
    return {"action": "click", "button_id": "k1c1", "confidence": 0.95, "description": "девушка у моря"}


async def test_spot_check_asks_without_anchoring_and_learns_without_clicking(tmp_path):
    bib = FakeBibinto(people=["Аня", "Оля"])
    sup, agents, inbox, *_ = make(tmp_path, bots=[("бибинто", bib, sure)], mode="auto",
                                  spot_check_rate=1.0, spot_checks_per_hour=5)
    agent = agents["бибинто"]
    await steps(agents, 2)
    assert bib.ratings == [("Аня", "7")]
    check = inbox.messages[-1]
    assert "Проверка вкуса" in check and "<b>7</b>" not in check and inbox.images[-1]
    await send(sup, "9", reply_to=9000 + len(inbox.messages))
    assert bib.ratings == [("Аня", "7")]  # the answer never clicks
    assert "Я поставил <b>7</b>, вы — <b>9</b>" in inbox.messages[-1]
    assert agent.store.recent_examples(5)[-1]["owner_choice"] == "9"
    assert agent.store.spot_stats() == {"sent": 1, "answered": 1, "agree": 0, "mae": 2.0}
    await send(sup, "/status")
    assert "Проверки вкуса" in inbox.messages[-1]


async def test_spot_checks_respect_the_hourly_cap_and_outstanding_limit(tmp_path):
    bib = FakeBibinto(people=[f"p{n}" for n in range(8)])
    sup, agents, inbox, *_ = make(tmp_path, bots=[("бибинто", bib, sure)], mode="auto",
                                  spot_check_rate=1.0, spot_checks_per_hour=2)
    await steps(agents, 9)
    assert len(bib.ratings) == 8
    assert sum("Проверка вкуса" in m for m in inbox.messages) == 2


# ------------------------------------------------------------------ paid offers
LEO_PAID = dict(premium_pattern=r"premium|премиум|vip|звёзд|буст",
                decline_buttons=["Нет", "Не сейчас", "Позже", "Назад"])


async def test_paid_offer_is_declined_without_the_model_or_the_owner(tmp_path):
    leo = FakeLeomatch()
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {"action": "ask_owner"})],
                                  mode="auto", **LEO_PAID)
    agent = agents["дайвинчик"]
    leo.messages.append(BotMessage(id=leo._id(), text="Подключи Premium за 99 ⭐ и смотри больше анкет!",
                                   buttons=reply_kb([["Купить Premium", "Нет, спасибо"]]), date=1))
    await steps(agents, 1)
    assert leo.sent[-1] == "Нет, спасибо" and agent.llm.calls == [] and inbox.messages == []
    assert agent.store.recent_actions(1)[0]["action"] == "decline"


async def test_menu_mentioning_premium_goes_to_the_model_which_keeps_browsing(tmp_path):
    leo = FakeLeomatch()
    LEO_BUY = dict(LEO_PAID, deny_buttons=["Купить", "Оформить", "Оплатить", "Подключить"],
                   rating_buttons="❤|👍|👎")  # Leomatch rates with hearts; its menu numbers are not ratings

    def decide(text, has_image):
        if "Премиум" in text:  # the model may try the purchase button first: it is blocked
            return {"action": "click", "button_id": "k0c1"}
        return {"action": "wait"}

    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, decide)], mode="auto", **LEO_BUY)
    agent = agents["дайвинчик"]
    leo.messages.append(BotMessage(id=leo._id(), text="1. Смотреть анкеты\n2. Купить Премиум",
                                   buttons=reply_kb([["1", "Купить Премиум"]]), date=1))
    await steps(agents, 1)
    assert agent.llm.calls, "the model decides when there is no refusal button"
    # The purchase button is blocked; instead of stopping, the agent goes back to the profiles itself.
    assert "Купить Премиум" not in leo.sent and leo.sent[-1] == "/start" and inbox.messages == []


async def test_repeated_paid_offers_go_to_the_owner(tmp_path):
    leo = FakeLeomatch()
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {"action": "wait"})],
                                  mode="auto", **LEO_PAID)
    agent = agents["дайвинчик"]
    for _ in range(6):
        leo.messages.append(BotMessage(id=leo._id(), text="Подключи VIP!",
                                       buttons=reply_kb([["Нет, спасибо"]]), date=1))
        await steps(agents, 1)
    assert "слишком много платных предложений" in inbox.messages[-1] and not agent.running


def test_status_uses_russian_plurals():
    from doppelgram.ui import plural
    assert [plural(n, "оценка", "оценки", "оценок") for n in (0, 1, 2, 5, 11, 21, 22, 112)] == [
        "оценок", "оценка", "оценки", "оценок", "оценок", "оценка", "оценки", "оценок"]


def test_plain_owner_text_shows_commands_as_code():
    from doppelgram.ui import as_html
    assert as_html("отправьте команду /start, затем /lead 123") == \
        "отправьте команду <code>/start</code>, затем <code>/lead 123</code>"
    assert as_html("ссылка https://t.me/x/123") == "ссылка https://t.me/x/123"


# ------------------------------------------------------------------ formatting & answers
async def test_dynamic_content_is_escaped(tmp_path):
    def decide(text, has_image):
        return {"action": "click", "button_id": "r0c0", "reason": "<b>bold</b> & <script>",
                "description": "<i>x</i>"}

    sup, agents, inbox, *_ = make(tmp_path, bots=[("арт", FakeRatingBot(), decide)], mode="supervised")
    await steps(agents, 2)
    card = inbox.messages[-1]
    assert "&lt;b&gt;bold&lt;/b&gt; &amp; &lt;script&gt;" in card and "&lt;i&gt;x&lt;/i&gt;" in card


async def test_answer_by_button_label(tmp_path):
    leo = FakeLeomatch()
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {
        "action": "click", "button_id": "k0c2", "confidence": 0.9})], mode="supervised")
    await steps(agents, 2)
    await send(sup, "❤️", reply_to=9000 + len(inbox.messages))
    assert leo.reactions == [("Аня", "❤️")]


async def test_number_means_rating_when_buttons_are_ratings(tmp_path):
    bib = FakeBibinto()
    sup, agents, inbox, *_ = make(tmp_path, bots=[("бибинто", bib, lambda t, i: {
        "action": "click", "button_id": "k0c0"})], mode="supervised")
    await steps(agents, 2)
    await send(sup, "10", reply_to=9000 + len(inbox.messages))
    assert bib.ratings == [("Аня", "10")]


def test_config_accepts_russian_modes_and_the_old_threshold_name(tmp_path):
    from doppelgram.config import load_config

    (tmp_path / "config.toml").write_text('[behavior]\nmode = "умный"\nauto_min_confidence = 0.8\n', encoding="utf-8")
    cfg = load_config(tmp_path / "config.toml", require_llm=False, require_telegram=False)
    assert cfg.behavior.mode == "smart" and cfg.behavior.confidence_threshold == 0.8


async def test_number_is_the_rating_even_when_buttons_are_not_in_order(tmp_path):
    from fakes_dating import _Base

    bot = _Base()
    bot.messages = [BotMessage(id=1, text="Оцени", media_kind="photo", media_key="p",
                               buttons=reply_kb([["Назад"], ["3", "2", "1"]]))]

    async def send_to_bot(text):
        bot.sent.append(text)

    bot.send_to_bot = send_to_bot
    sup, agents, inbox, *_ = make(tmp_path, bots=[("бот", bot, lambda t, i: {
        "action": "click", "button_id": "k1c0"})], mode="supervised")
    await steps(agents, 1)
    await send(sup, "1", reply_to=9000 + len(inbox.messages))
    assert bot.sent == ["1"]  # the rating "1", not the first button "Назад"


# ---- regressions from the review of the control commit ---------------------------------------
def test_smart_timeout_is_off_by_default():
    from pathlib import Path

    from doppelgram.config import BehaviorConfig
    assert BehaviorConfig(task_file=Path("t"), taste_file=Path("t")).ask_timeout_min == 0


async def test_thumbs_up_reply_presses_the_thumbs_up_button(tmp_path):
    from fakes_dating import _Base

    bot = _Base()
    bot.messages = [BotMessage(id=1, text="Оцени", media_kind="photo", media_key="p",
                               buttons=reply_kb([["👍", "👎"]]))]

    async def send_to_bot(text):
        bot.sent.append(text)

    bot.send_to_bot = send_to_bot
    sup, agents, inbox, *_ = make(tmp_path, bots=[("бот", bot, lambda t, i: {
        "action": "click", "button_id": "k0c1"})], mode="supervised")  # the agent proposes 👎
    await steps(agents, 1)
    await send(sup, "👍", reply_to=9000 + len(inbox.messages))
    assert bot.sent == ["👍"]


@pytest.mark.parametrize("value", ["NaN", "nan", "inf", "-inf"])
async def test_non_finite_setting_values_are_rejected(tmp_path, value):
    sup, agents, inbox, *_ = make(tmp_path)
    before = agents["арт"].b.daily_budget_usd
    await send(sup, f"/set бюджет {value}")
    assert agents["арт"].b.daily_budget_usd == before and "Не подходит значение" in inbox.messages[-1]


async def test_reset_all_releases_cards_held_by_the_old_mode(tmp_path):
    sup, agents, inbox, *_ = make(tmp_path, mode="auto", spot_check_rate=0)
    art = agents["арт"]
    await send(sup, "/mode ручной")
    await steps(agents, 2)
    assert art.store.latest_pending() is not None and art.tg.ratings == []
    await send(sup, "/reset all")
    assert art.mode == "auto" and art.store.latest_pending() is None
    await steps(agents, 3)
    assert art.tg.ratings == ["4", "4", "4"]


def test_long_lines_are_split_without_breaking_entities_or_tags():
    import html as h

    from doppelgram.telegram import _chunks
    from doppelgram.ui import b, esc, join

    body = join(b("Правило"), esc("&" * 1000 + "<x>" * 300), "конец")
    chunks = _chunks(body, 4000)
    assert all(len(c) <= 4000 for c in chunks)
    text = h.unescape("".join(chunks).replace("\n", ""))
    assert text.count("&") == 1000 and text.count("<x>") == 300 and text.endswith("конец")
    for c in chunks:  # every chunk is well-formed: no dangling entity or tag
        assert c.count("<b>") == c.count("</b>")
        assert not c.rstrip().endswith("&") and "&am\n" not in c


async def test_a_button_labelled_like_an_accept_word_wins(tmp_path):
    from fakes_dating import _Base

    bot = _Base()
    bot.messages = [BotMessage(id=1, text="Продолжить?", media_kind="photo", media_key="p",
                               buttons=reply_kb([["Да", "Нет"]]))]

    async def send_to_bot(text):
        bot.sent.append(text)

    bot.send_to_bot = send_to_bot
    sup, agents, inbox, *_ = make(tmp_path, bots=[("бот", bot, lambda t, i: {
        "action": "click", "button_id": "k0c1"})], mode="supervised",
        rating_buttons=r"^(Да|Нет)$")  # the agent proposes "Нет"
    await steps(agents, 1)
    await send(sup, "да", reply_to=9000 + len(inbox.messages))
    assert bot.sent == ["Да"]


async def test_card_from_the_previous_mode_is_released_after_a_config_change(tmp_path):
    sup, agents, inbox, *_ = make(tmp_path, mode="supervised", spot_check_rate=0)
    art = agents["арт"]
    await steps(agents, 2)
    assert art.store.latest_pending() is not None and art.tg.ratings == []
    art.b = replace(art._base, mode="auto")  # config.toml edited and the service restarted
    await steps(agents, 4)
    assert art.store.latest_pending() is None and art.tg.ratings == ["4", "4", "4"]


async def test_purchase_buttons_are_blocked_in_any_case_and_by_menu_number(tmp_path):
    leo = FakeLeomatch()
    deny = dict(LEO_PAID, deny_buttons=["Купить", "Премиум"])
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {"action": "wait"})],
                                  mode="auto", **deny)
    leo.messages.append(BotMessage(id=leo._id(), text="1. Смотреть анкеты\n2. купить премиум",
                                   buttons=reply_kb([["1", "2", "купить премиум"]]), date=1))
    from doppelgram.agent import observe
    view = observe(leo.messages, tuple(deny["deny_buttons"])).current
    assert [(btn.text, btn.blocked) for btn in view.buttons] == [
        ("1", False), ("2", True), ("купить премиум", True)]
    from doppelgram import brain
    d = brain.validate_decision({"action": "send", "text": "Купить премиум"}, view, max_wait_s=60,
                                max_send_chars=50, deny=("купить",))
    assert d.action == "ask_owner"


def test_commands_inside_urls_are_left_alone():
    from doppelgram.ui import as_html
    assert as_html("https://example.com/?next=/start и /help") == \
        "https://example.com/?next=/start и <code>/help</code>"
    assert as_html("путь /opt/doppelgram/data") == "путь /opt/doppelgram/data"


def test_numbered_lists_in_a_profile_do_not_block_ratings():
    from doppelgram.agent import observe

    card = BotMessage(id=1, text="Аня, 22\n1. люблю VIP-ложи\n2. Премиум кофе", media_kind="photo", media_key="p",
                      buttons=reply_kb([["1", "2", "3"]]))
    view = observe([card], ("VIP", "Премиум")).current
    assert not any(btn.blocked for btn in view.buttons)


def test_menu_after_a_photo_is_still_checked_and_urls_with_parentheses_are_kept():
    from doppelgram.agent import observe
    from doppelgram.ui import as_html

    photo = BotMessage(id=1, text="", media_kind="photo", media_key="p", date=0)
    menu = BotMessage(id=2, text="1. Смотреть анкеты\n2. купить премиум", date=1,
                      buttons=reply_kb([["1", "2"]]))
    view = observe([photo, menu], ("Купить",)).current
    assert [(btn.text, btn.blocked) for btn in view.buttons] == [("1", False), ("2", True)]
    assert as_html("https://example.com/(/start) и /help") == "https://example.com/(/start) и <code>/help</code>"


def test_command_formatting_is_linear_on_long_input():
    import time

    from doppelgram.ui import as_html
    start = time.perf_counter()
    as_html("x" * 50_000)
    as_html("/a" * 20_000)
    as_html("https://" + "y" * 50_000)
    assert time.perf_counter() - start < 0.2


def test_urls_in_parentheses_and_many_urls_stay_fast_and_untouched():
    import time

    from doppelgram.ui import as_html
    assert as_html("(www.example.com/(/start)) и (t.me/x/(/help)) /ok") == \
        "(www.example.com/(/start)) и (t.me/x/(/help)) <code>/ok</code>"
    start = time.perf_counter()
    as_html(" ".join(["https://x.y/z", "/a"] * 5000))
    assert time.perf_counter() - start < 0.2


# ---- Bibinto: ads on pictures, "someone rated you N" -------------------------------------------
def vip_picture(bot):
    return BotMessage(id=bot._id(), text="🐨Коала: Хочу, чтобы у тебя было больше оценок! VIP-Status откроет Буст",
                      media_kind="photo", media_key="koala", date=1,
                      buttons=(Button(id="r0c0", row=0, col=0, text="Узнать про вип", kind="callback", data=b"v"),
                               Button(id="r0c1", row=0, col=1, text="Пропустить", kind="callback", data=b"s")))


@pytest.mark.parametrize("mode", ["auto", "supervised"])
async def test_a_picture_ad_is_not_treated_as_a_profile(tmp_path, mode):
    bib = FakeBibinto()
    clicks = []

    async def click(msg_id, button):
        clicks.append(button.text)
        from doppelgram.models import ClickOutcome
        return ClickOutcome(ok=True)

    bib.click = click
    sup, agents, inbox, *_ = make(tmp_path, bots=[("бибинто", bib, lambda t, i: {
        "action": "click", "button_id": "r0c1", "confidence": 0.9})], mode=mode,
        spot_check_rate=1.0, spot_checks_per_hour=10)
    bib.messages.append(vip_picture(bib))
    await steps(agents, 1)
    assert clicks == ["Пропустить"]
    assert not any("Проверка вкуса" in m or "Новая анкета" in m for m in inbox.messages)
    assert agents["бибинто"].store.counts_today(0)["rated"] == 0


INCOMING = dict(incoming_rating_pattern=r"оценил[аи]?\s+вас,?\s+на\s+(\d{1,2})\s*балл",
                incoming_open_buttons=["Посмотреть"], deny_buttons=["Отключить"])


def incoming_message(bot, score=10):
    return BotMessage(id=bot._id(), date=1, text=f"⭐️ Кто-то из вашего города, оценил вас, на {score} баллов 😮\n"
                                                 f"Посмотрим кто это?",
                      buttons=reply_kb([["Посмотреть", "Попозже", "Отключить"]]))


def rater(own):
    def decide(text, has_image):
        if has_image:
            return {"action": "click", "button_id": f"k{0 if own <= 5 else 1}c{(own - 1) % 5}", "confidence": 0.9}
        return {"action": "wait"}
    return decide


@pytest.mark.parametrize("own,expected", [(7, "10"), (6, "10"), (4, "4"), (10, "10")])
async def test_incoming_rating_is_opened_and_a_fitting_rating_is_raised(tmp_path, own, expected):
    bib = FakeBibinto(people=["Аня"])
    sup, agents, inbox, *_ = make(tmp_path, bots=[("бибинто", bib, rater(own))], mode="auto",
                                  spot_check_rate=0, **INCOMING)
    agent = agents["бибинто"]
    bib.messages = [incoming_message(bib)]
    original = bib.send_to_bot

    async def send(text):
        if text == "Посмотреть":
            bib.sent.append(text)
            bib.shown = 0
            bib._screen()
            return
        await original(text)

    bib.send_to_bot = send
    await steps(agents, 1)
    assert bib.sent == ["Посмотреть"] and agent.llm.calls == []  # opened by rule
    await steps(agents, 1)
    assert bib.ratings == [("Аня", expected)]
    assert agent.store.get("reciprocal") is None
    assert "already rated the owner highly" in agent.llm.calls[-1][1]["content"][0]["text"]


async def test_a_stale_incoming_rating_does_not_raise_later_ratings(tmp_path):
    bib = FakeBibinto(people=["Аня"])
    sup, agents, inbox, clock, *_ = make(tmp_path, bots=[("бибинто", bib, rater(7))], mode="auto",
                                         spot_check_rate=0, **INCOMING)
    agents["бибинто"].store.set("reciprocal", '{"score": 10, "ts": %f}' % (clock() - 3600))
    await steps(agents, 2)
    assert bib.ratings == [("Аня", "7")]


def test_disable_notifications_button_is_blocked():
    from doppelgram.agent import observe
    bot = FakeBibinto()
    view = observe([incoming_message(bot)], ("Отключить",)).current
    assert [btn.blocked for btn in view.buttons] == [False, False, True]


def test_task_templates_have_no_minor_rule_and_first_messages_are_off_by_default():
    from pathlib import Path

    from doppelgram.config import ConversationConfig

    root = Path(__file__).parent.parent
    for name in ("task-leomatch.example.md", "task-bibinto.example.md"):
        assert "несовершеннолет" not in (root / name).read_text(encoding="utf-8")
    defaults = ConversationConfig(about_file=Path("a"), goal_file=Path("g"), style_file=Path("s"))
    assert defaults.write_first is False


async def test_reciprocal_raise_applies_only_to_the_profile_opened_after_the_press(tmp_path):
    bib = FakeBibinto(people=["Обычная", "Оценившая"])
    sup, agents, inbox, *_ = make(tmp_path, bots=[("бибинто", bib, rater(7))], mode="auto",
                                  spot_check_rate=0, **INCOMING)
    await steps(agents, 1)  # /start → "Обычная" card on screen (sent before the notification)
    bib.messages.append(incoming_message(bib))
    original = bib.send_to_bot

    async def send(text):
        if text == "Посмотреть":
            bib.sent.append(text)
            bib.shown = 1
            bib._screen()  # the rater's profile arrives after the press
            return
        await original(text)

    bib.send_to_bot = send
    await steps(agents, 2)
    assert bib.ratings == [("Оценившая", "10")]


async def test_failed_open_leaves_no_raise_and_menu_numbers_are_never_raised(tmp_path):
    bib = FakeBibinto(people=["Аня"])
    sup, agents, inbox, clock, *_ = make(tmp_path, bots=[("бибинто", bib, rater(7))], mode="auto",
                                         spot_check_rate=0, rating_buttons=r"^[1-5]$", **INCOMING)
    agent = agents["бибинто"]
    bib.messages = [incoming_message(bib)]

    async def failing(msg_id, button):
        from doppelgram.models import ClickOutcome
        return ClickOutcome(ok=False, answer="MessageIdInvalidError")

    bib.click = failing
    await steps(agents, 1)
    assert agent.store.get("reciprocal") is None
    # A "7" that is not a rating button (rating_buttons = 1-5) is never turned into "10".
    from doppelgram.models import Decision
    agent.store.set("reciprocal", f'{{"score": 10, "ts": {clock():f}, "after": 0}}')
    from doppelgram.agent import observe
    card = BotMessage(id=999, text="", media_kind="photo", media_key="x", buttons=reply_kb([["7", "10"]]))
    obs = observe([card])
    d = agent._apply_reciprocal(obs, Decision(action="click", button_id="k0c0"))
    assert d.button_id == "k0c0"


async def test_text_offer_right_after_a_card_photo_is_still_declined(tmp_path):
    leo = FakeLeomatch()
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {"action": "wait"})],
                                  mode="auto", **LEO_PAID)
    await steps(agents, 1)  # /start → a card photo is on screen
    leo.messages.append(BotMessage(id=leo._id(), text="Подключи Premium!", date=leo.messages[-1].date,
                                   buttons=reply_kb([["Купить", "Нет, спасибо"]])))
    await steps(agents, 1)
    assert leo.sent[-1] == "Нет, спасибо" and len(agents["дайвинчик"].llm.calls) == 0


async def test_reciprocal_score_is_bound_to_cards_after_the_press(tmp_path):
    from doppelgram.agent import observe

    sup, agents, inbox, clock, *_ = make(tmp_path, bots=[("бибинто", FakeBibinto(), rater(7))], **INCOMING)
    agent = agents["бибинто"]
    agent.store.set("reciprocal", f'{{"score": 9, "ts": {clock():f}, "after": 50}}')
    old = BotMessage(id=50, text="", media_kind="photo", media_key="a", buttons=reply_kb([["7", "9"]]))
    new = BotMessage(id=51, text="", media_kind="photo", media_key="b", buttons=reply_kb([["7", "9"]]))
    assert agent._reciprocal_score(observe([old])) is None
    assert agent._reciprocal_score(observe([new])) == 9


# ---- refusals that name the paid feature; numbers 10-18 ----------------------------------------
async def test_refusal_naming_premium_is_pressed_by_rule(tmp_path):
    leo = FakeLeomatch()
    paid = dict(premium_pattern=r"premium|премиум", decline_buttons=["Нет", "Пока без"],
                deny_buttons=["⭐", "Premium", "Активировать"])
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {"action": "ask_owner"})],
                                  mode="auto", **paid)
    leo.messages.append(BotMessage(id=leo._id(), date=1, text="Саша, больше внимания — больше знакомств 👀\n"
                                   "С ⭐ Premium-статусом: ты выше в выдаче\n⭐ Активируй Premium",
                                   buttons=reply_kb([["⭐Активировать", "Пока без Premium"]])))
    await steps(agents, 1)
    assert leo.sent[-1] == "Пока без Premium" and agents["дайвинчик"].llm.calls == []
    from doppelgram.agent import observe
    view = observe(leo.messages, tuple(paid["deny_buttons"]), tuple(paid["decline_buttons"])).current
    assert [(btn.text, btn.blocked) for btn in view.buttons] == [("⭐Активировать", True), ("Пока без Premium", False)]


@pytest.mark.parametrize("text,expected", [
    ("Аня, 23, Москва – рост 165, мне 17, люблю 2 кошек и 15 книг",
     "Аня, 23, Москва – рост 165, мне, люблю 2 кошек и книг"),
    # The age field is not this filter's job.
    ("Аня, 19, Москва – учусь на 12 курсе", "Аня, 19, Москва – учусь на курсе"),
    ("было 10.5 баллов, 120 км, 2015 год", "было 10.5 баллов, 120 км, 2015 год"),  # not standalone 10-20
])
def test_numbers_10_20_are_removed_except_the_age_field(text, expected):
    from doppelgram.brain import hide_numbers_10_18
    assert hide_numbers_10_18(text, r"^[^,\n]+,\s*(\d{1,2}),") == expected


async def test_number_filter_keeps_rating_buttons_and_follows_the_setting(tmp_path):
    bib = FakeBibinto(people=["Аня"])
    sup, agents, inbox, *_ = make(tmp_path, bots=[("бибинто", bib, rater(7))], mode="auto", spot_check_rate=0,
                                  age_in_prompt="hide")
    agent = agents["бибинто"]
    original = bib._screen

    def screen():
        original()
        last = bib.messages[-1]
        bib.messages[-1] = replace(last, text="👩‍🦰: Аня, 22 лет\n✍️: мне 17, учусь в 11 классе")

    bib._screen = screen
    await steps(agents, 2)
    prompt = agent.llm.calls[-1][1]["content"][0]["text"]
    assert '"10"' in prompt and '"7"' in prompt          # rating labels survive even in "hide" mode
    assert "17" not in prompt and "11 классе" not in prompt
    await send(sup, "/set числа выкл")
    assert agent.b.hide_numbers_10_18 is False and "выкл" in inbox.messages[-1]


@pytest.mark.parametrize("label,refusal", [
    ("Пока без Premium", True), ("Нет, спасибо", True), ("👎 Нет", True), ("Не сейчас", True),
    ("Нет, хочу Premium", False), ("Позже купить", False), ("Купить без скидки", False),
    ("Активировать позже", False), ("Сегодня нет", False),
    ("Нет, не хочу Premium", True), ("Пока без подключения Premium", True), ("Нет, но хочу купить", False),
])
def test_refusal_labels(label, refusal):
    from doppelgram.agent import _is_refusal
    assert _is_refusal(label, ("Нет", "Не сейчас", "Позже", "Пока без", "Без")) is refusal


@pytest.mark.parametrize("text,expected", [
    ("Мне,17 и всё", "Мне, и всё"), ("A17 Pro, M16", "A17 Pro, M16"), ("17лет", "лет"),
    ("10,5 и 10.5 и 15", "10,5 и 10.5 и"),
])
def test_numbers_10_20_edge_cases(text, expected):
    from doppelgram.brain import hide_numbers_10_18
    assert hide_numbers_10_18(text) == expected


def test_only_1_to_10_labels_bypass_the_sanitizer():
    from doppelgram import brain
    msg = BotMessage(id=1, text="", buttons=reply_kb([["1", "10", "17", "20"]]))
    clean = lambda t: brain.sanitize_text(t, "keep", "", True)  # noqa: E731
    msgs = brain.build_messages(task="t", taste="", examples=[], history=[], recent_actions=[], current=msg,
                                image_url=None, image_from_previous=False, notes=[], sanitize=clean)
    text = msgs[1]["content"][0]["text"]
    assert '"1"' in text and '"10"' in text and '"17"' not in text and '"20"' in text  # range is 10-18



@pytest.mark.parametrize("text,expected", [("9 10 18 19 20", "9 19 20"), ("мне 18, ей 19", "мне, ей 19")])
def test_number_range_is_10_to_18_inclusive(text, expected):
    from doppelgram.brain import hide_numbers_10_18
    assert hide_numbers_10_18(text) == expected


async def test_picture_without_buttons_waits_for_the_keyboard_before_the_model(tmp_path):
    bib = FakeBibinto(people=["Аня"])
    sup, agents, inbox, clock, *_ = make(tmp_path, bots=[("бибинто", bib, rater(7))], mode="auto",
                                         spot_check_rate=0)
    agent = agents["бибинто"]
    bib.messages = [BotMessage(id=500, text="👩‍🦰: Аня, 22 лет", media_kind="photo", media_key="a", date=1,
                               keyboard_hidden=True)]
    for _ in range(3):
        await steps(agents, 1)
        clock.t += 11
    assert agent.llm.calls == []  # waited three times for the keyboard
    await steps(agents, 1)
    assert len(agent.llm.calls) == 1  # then the model decides


async def test_ask_owner_records_the_visible_buttons(tmp_path):
    sup, agents, inbox, *_ = make(tmp_path, bots=[("арт", FakeRatingBot(), lambda t, i: {
        "action": "ask_owner", "reason": "?"})], mode="supervised")
    await steps(agents, 2)
    row = agents["арт"].store.recent_actions(1)[0]
    assert row["action"] == "ask_owner" and row["button_text"] == "1 | 2 | 3 | 4 | 5" and row["text"] == "картинка"


def test_legacy_config_name_for_the_number_filter(tmp_path):
    from doppelgram.config import load_config
    (tmp_path / "config.toml").write_text("[behavior]\nhide_numbers_10_20 = false\n", encoding="utf-8")
    cfg = load_config(tmp_path / "config.toml", require_llm=False, require_telegram=False)
    assert cfg.behavior.hide_numbers_10_18 is False


async def test_keyboard_wait_state_is_bounded_and_legacy_number_setting_migrates(tmp_path):
    bib = FakeBibinto(people=["Аня"])
    sup, agents, inbox, clock, *_ = make(tmp_path, bots=[("бибинто", bib, rater(7))], mode="auto")
    agent = agents["бибинто"]
    for n in range(20):
        bib.messages = [BotMessage(id=600 + n, text="", media_kind="photo", media_key=f"p{n}", date=1)]
        await steps(agents, 1)
    keys = [k for (k,) in agent.store.db.execute("SELECT key FROM kv WHERE key LIKE 'kbwait%'")]
    assert keys == ["kbwait"]

    store = Store(":memory:")
    store.set("setting:hide_numbers_10_20", "false")
    a = Agent(agent.cfg, FakeRatingBot(), ScriptedLLM(rate_or_wait), store, behavior=agent._base)
    assert a.b.hide_numbers_10_18 is False and store.get("setting:hide_numbers_10_20") is None


async def test_ask_owner_labels_drop_urls_digits_and_long_text(tmp_path):
    from doppelgram.models import Button as Btn
    bot = FakeRatingBot()
    sup, agents, inbox, *_ = make(tmp_path, bots=[("арт", bot, lambda t, i: {"action": "ask_owner"})],
                                  mode="supervised")
    await steps(agents, 1)
    bot.messages[-1] = replace(bot.messages[-1], buttons=(
        Btn("r0c0", 0, 0, "Написать Анне 23 года, она ждёт тебя тут", "callback", b"x"),
        Btn("r0c1", 0, 1, "https://t.me/anna", "url", url="https://t.me/anna")))
    await steps(agents, 1)
    label = agents["арт"].store.recent_actions(1)[0]["button_text"]
    assert "23" not in label and "t.me" not in label and len(label) <= 30


# ---- keep going instead of stopping (auto/smart) ------------------------------------------------
LEO_GO = dict(rating_buttons="❤|👍|👎", continue_buttons=["Продолжить смотреть", "Смотреть анкеты", "🚀"],
              owner_required_pattern="верификац|кружок")
LEO_DIGITS = dict(LEO_GO, rating_buttons=r"^\s*(?:10|[1-9])\s*$|❤")  # a bot that rates with digits


async def test_location_request_is_skipped_with_the_continue_button(tmp_path):
    leo = FakeLeomatch()
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {
        "action": "ask_owner", "reason": "Бот просит отправить местоположение"})], mode="auto", **LEO_GO)
    leo.messages.append(BotMessage(id=leo._id(), date=1, text="Чтобы видеть людей рядом, отправь местоположение",
                                   buttons=reply_kb([["Продолжить смотреть анкеты", "📍 Отправить мои координаты"]])))
    await steps(agents, 1)
    assert leo.sent[-1] == "Продолжить смотреть анкеты" and inbox.messages == []
    assert agents["дайвинчик"].running


async def test_repeated_short_waits_on_an_ad_move_on(tmp_path):
    leo = FakeLeomatch()
    sup, agents, inbox, clock, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {
        "action": "wait", "wait_seconds": 30})], mode="auto", **LEO_GO)
    leo.messages.append(BotMessage(id=leo._id(), date=1, text="Саша, твоя анкета может больше. Заходи @Leomatchglobal",
                                   buttons=reply_kb([["🚀 Смотреть анкеты"]])))
    await steps(agents, 1)
    clock.t += 31
    await steps(agents, 1)
    assert leo.sent[-1] == "🚀 Смотреть анкеты"


async def test_verification_still_goes_to_the_owner_and_continues_are_capped(tmp_path):
    leo = FakeLeomatch()
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {"action": "ask_owner",
                                                                                 "reason": "?"})],
                                  mode="auto", **LEO_GO)
    agent = agents["дайвинчик"]
    leo.messages.append(BotMessage(id=leo._id(), date=1, text="Для верификации анкеты отправь видео-кружок",
                                   buttons=reply_kb([["🚀 Смотреть анкеты"]])))
    await steps(agents, 1)
    assert not agent.running and "Нужна ваша помощь" in inbox.messages[-1]
    # More than a dozen "continues" an hour means something is wrong: then the owner is asked.
    await send(sup, "/start")
    for _n in range(12):
        agent.store.log_action(agent.clock(), None, None, "click", button_text="🚀 Смотреть анкеты", decided_by="rule")
    leo.messages.append(BotMessage(id=leo._id(), date=1, text="Непонятный экран", buttons=reply_kb([["🚀"]])))
    await steps(agents, 1)
    assert not agent.running


async def test_bibinto_waits_for_the_photo_after_the_rating_keyboard(tmp_path):
    bib = FakeBibinto(people=["Аня"])
    sup, agents, inbox, clock, *_ = make(tmp_path, bots=[("бибинто", bib, rater(7))], mode="auto",
                                         spot_check_rate=0)
    agent = agents["бибинто"]
    bib.messages = [BotMessage(id=700, text="🐨", buttons=reply_kb([["1", "2", "3", "4", "5"],
                                                                    ["6", "7", "8", "9", "10"]]), date=1)]
    await steps(agents, 1)
    assert agent.llm.calls == []  # the photo has not arrived yet: wait
    bib.messages.append(BotMessage(id=701, text="👩‍🦰: Аня, 22 лет", media_kind="photo", media_key="a", date=2))
    clock.t += 11
    await steps(agents, 1)
    assert len(agent.llm.calls) == 1 and len(agent.llm.calls[0][1]["content"]) == 2  # rated with the photo


async def test_a_priced_rocket_button_is_never_a_continue(tmp_path):
    leo = FakeLeomatch()
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {"action": "ask_owner"})],
                                  mode="auto", **LEO_GO)
    leo.messages.append(BotMessage(id=leo._id(), date=1, text="Буст анкеты",
                                   buttons=reply_kb([["🚀 Продолжить за 99 ₽", "⭐ Смотреть анкеты"]])))
    await steps(agents, 1)
    assert leo.sent[-1] == "/start"  # no safe continue button: restart instead of paying


@pytest.mark.parametrize("text", ["Ваш аккаунт заблокирован", "Подтвердите личность: отправьте селфи",
                                  "Анкета ограничена за спам"])
async def test_blocks_and_identity_checks_go_to_the_owner(tmp_path, text):
    leo = FakeLeomatch()
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {"action": "ask_owner"})],
                                  mode="auto", **dict(LEO_GO, owner_required_pattern=""))  # built in, not config
    leo.messages.append(BotMessage(id=leo._id(), date=1, text=text, buttons=reply_kb([["🚀 Смотреть анкеты"]])))
    await steps(agents, 1)
    assert not agents["дайвинчик"].running and "Нужна ваша помощь" in inbox.messages[-1]


async def test_restart_loops_by_the_model_are_capped(tmp_path):
    leo = FakeLeomatch()
    sup, agents, inbox, clock, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {
        "action": "send", "text": "/start"})], mode="auto", **LEO_GO)
    agent = agents["дайвинчик"]
    for _n in range(20):
        leo.messages.append(BotMessage(id=leo._id(), date=1, text="Меню", buttons=reply_kb([["2", "3"]])))
        await steps(agents, 1)
        if not agent.running:
            break
    assert not agent.running and "по кругу" in inbox.messages[-1]
    assert agent.store.count_actions(0, "send", "/start") <= 12


async def test_last_wait_does_not_leak_to_another_identical_screen(tmp_path):
    leo = FakeLeomatch()
    sup, agents, inbox, clock, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {
        "action": "wait", "wait_seconds": 30})], mode="auto", **LEO_GO)
    ad = dict(date=1, text="Реклама", buttons=reply_kb([["🚀 Смотреть анкеты"]]))
    leo.messages.append(BotMessage(id=leo._id(), **ad))
    await steps(agents, 1)
    leo.messages.append(BotMessage(id=leo._id(), **ad))  # a new, identical ad message
    clock.t += 31
    await steps(agents, 1)
    assert "🚀 Смотреть анкеты" not in leo.sent  # first wait on the new message: just wait


async def test_an_ad_about_bans_is_still_skipped(tmp_path):
    leo = FakeLeomatch()
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {"action": "ask_owner"})],
                                  mode="auto", **LEO_GO)
    leo.messages.append(BotMessage(id=leo._id(), date=1, text=(
        "Саша, твоя анкета в Дайвинчике может больше. Заходи @Leomatchglobal. Научу делать профиль, "
        "который лайкают, и не попадать в бан, чтобы избежать блокировки"), buttons=reply_kb([["🚀 Смотреть анкеты"]])))
    await steps(agents, 1)
    assert leo.sent[-1] == "🚀 Смотреть анкеты" and agents["дайвинчик"].running


async def test_a_charge_confirmation_is_left_with_a_restart_not_the_rocket(tmp_path):
    leo = FakeLeomatch()
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {"action": "ask_owner"})],
                                  mode="auto", **LEO_GO)
    leo.messages.append(BotMessage(id=leo._id(), date=1, text="Буст: списать 99 звёзд с баланса? Нажмите 🚀",
                                   buttons=reply_kb([["🚀"]])))
    await steps(agents, 1)
    assert leo.sent[-1] == "/start"


@pytest.mark.parametrize("decision", [{"action": "send", "text": "/start"}, {"action": "wait", "wait_seconds": 30},
                                      {"action": "click", "button_id": "k0c0"}])
@pytest.mark.parametrize("text", ["Ваш аккаунт заблокирован", "Пройдите капчу: выберите картинку с котом"])
async def test_blocks_and_captchas_always_reach_the_owner(tmp_path, decision, text):
    leo = FakeLeomatch()
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: dict(decision))],
                                  mode="auto", **LEO_GO)
    leo.messages.append(BotMessage(id=leo._id(), date=1, text=text, buttons=reply_kb([["🚀 Смотреть анкеты"]])))
    await steps(agents, 1)
    assert not agents["дайвинчик"].running and "Нужна ваша помощь" in inbox.messages[-1]
    assert "/start" not in leo.sent[1:] and "🚀 Смотреть анкеты" not in leo.sent


async def test_a_profile_mentioning_selfies_is_still_rated(tmp_path):
    leo = FakeLeomatch()
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {
        "action": "click", "button_id": "k0c0"})], mode="auto", **LEO_GO)
    leo.messages.append(BotMessage(id=leo._id(), date=1, text="Аня, Москва — люблю селфи и верификацию кофе",
                                   buttons=reply_kb([["❤️", "👎", "💤"]]), media_kind="photo", media_key="a"))
    await steps(agents, 1)
    assert leo.sent[-1] == "❤️" and agents["дайвинчик"].running


async def test_model_clicks_on_continue_count_towards_the_loop_cap(tmp_path):
    leo = FakeLeomatch()
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {
        "action": "click", "button_id": "k0c0"})], mode="auto", **LEO_GO)
    agent = agents["дайвинчик"]
    for _ in range(20):
        leo.messages.append(BotMessage(id=leo._id(), date=1, text="Реклама",
                                       buttons=reply_kb([["🚀 Смотреть анкеты"]])))
        await steps(agents, 1)
        if not agent.running:
            break
    assert not agent.running and "по кругу" in inbox.messages[-1]
    assert leo.sent.count("🚀 Смотреть анкеты") <= 12


async def test_rating_a_profile_resets_the_loop_count(tmp_path):
    leo = FakeLeomatch()
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {
        "action": "click", "button_id": "k0c0"})], mode="auto", **LEO_GO)
    agent = agents["дайвинчик"]
    for n in range(30):
        if n % 10 == 9:
            leo.messages.append(BotMessage(id=leo._id(), date=1, text="Аня, Москва", media_kind="photo",
                                           media_key=f"p{n}", buttons=reply_kb([["❤️", "👎", "💤"]])))
        else:
            leo.messages.append(BotMessage(id=leo._id(), date=1, text="Реклама",
                                           buttons=reply_kb([["🚀 Смотреть анкеты"]])))
        await steps(agents, 1)
    assert agent.running and leo.sent.count("❤️") == 3


@pytest.mark.parametrize("decision", [{"action": "click", "button_id": "k0c0"}, {"action": "ask_owner"}])
@pytest.mark.parametrize("text", ["Списать 99 звёзд? Нажмите 🚀", "Подтвердите покупку буста: нажмите 🚀"])
async def test_a_purchase_is_never_confirmed(tmp_path, decision, text):
    leo = FakeLeomatch()
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: dict(decision))],
                                  mode="auto", **LEO_GO)
    leo.messages.append(BotMessage(id=leo._id(), date=1, text=text, buttons=reply_kb([["🚀"]])))
    await steps(agents, 1)
    assert "🚀" not in leo.sent and leo.sent[-1] == "/start"


async def test_a_block_after_a_profile_is_not_hidden_by_its_old_keyboard(tmp_path):
    leo = FakeLeomatch()
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {
        "action": "send", "text": "/start"})], mode="auto", **LEO_GO)
    leo.messages.append(BotMessage(id=leo._id(), date=1, text="Аня, Москва", media_kind="photo", media_key="a",
                                   buttons=reply_kb([["❤️", "👎", "💤"]])))
    leo.messages.append(BotMessage(id=leo._id(), date=2, text="Ваш аккаунт заблокирован"))
    await steps(agents, 1)
    assert not agents["дайвинчик"].running and "Нужна ваша помощь" in inbox.messages[-1]


async def test_a_configured_owner_screen_is_not_skipped_by_the_model(tmp_path):
    leo = FakeLeomatch()
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {
        "action": "send", "text": "/start"})], mode="auto", **LEO_GO)
    leo.messages.append(BotMessage(id=leo._id(), date=1, text="Отправь видео-кружок",
                                   buttons=reply_kb([["🚀 Смотреть анкеты"]])))
    await steps(agents, 1)
    assert not agents["дайвинчик"].running


@pytest.mark.parametrize("reason, stops", [("Это капча", True), ("Реклама, это не капча", False)])
async def test_a_captcha_the_model_recognised_goes_to_the_owner(tmp_path, reason, stops):
    leo = FakeLeomatch()
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {
        "action": "ask_owner", "reason": reason})], mode="auto", **LEO_GO)
    leo.messages.append(BotMessage(id=leo._id(), date=1, text="Выберите все картинки с кошкой",
                                   buttons=reply_kb([["🚀 Смотреть анкеты"]])))
    await steps(agents, 1)
    assert agents["дайвинчик"].running is not stops


async def test_digit_menus_count_towards_the_loop_cap(tmp_path):
    leo = FakeLeomatch()
    sup, agents, inbox, clock, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {
        "action": "click", "button_id": "k0c0"})], mode="auto", **LEO_DIGITS)
    agent = agents["дайвинчик"]

    async def menu_again(text):  # the bot answers every press with the same menu, never a profile
        leo.sent.append(text)
    leo.send_to_bot = menu_again
    for _ in range(20):
        leo.messages.append(BotMessage(id=leo._id(), date=1, text="1. Смотреть анкеты\n2. Моя анкета",
                                       buttons=reply_kb([["1", "2"]])))
        for _ in range(4):  # digit buttons without a picture: it first waits for the photo three times
            await steps(agents, 1)
            clock.t += 11
        if not agent.running:
            break
    assert not agent.running and leo.sent.count("1") <= 12


@pytest.mark.parametrize("text, buttons, decision", [
    ("Буст анкеты", ["🚀 Продолжить за 99 ₽", "Назад"], {"action": "click", "button_id": "k0c0"}),
    ("Списать 99 звёзд?\n1. Да\n2. Нет", ["1", "2"], {"action": "click", "button_id": "k0c0"}),
    ("Списать 99 ₽?\n1. Подтвердить\n2. Нет", ["1", "2"], {"action": "click", "button_id": "k0c0"}),
    ("Списать 99 ₽?\n1. Продолжить\n2. Нет", ["1", "2"], {"action": "send", "text": "1"}),
    ("Оплата 99 ₽\n1. Согласен\n2. Нет", ["1", "2"], {"action": "click", "button_id": "k0c0"}),
    ("Подтвердите покупку: ответьте ДА", ["Назад"], {"action": "send", "text": "ДА"}),
])
async def test_the_model_cannot_confirm_a_purchase(tmp_path, text, buttons, decision):
    leo = FakeLeomatch()
    sup, agents, inbox, clock, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: dict(decision))], mode="auto",
                                         **LEO_DIGITS)

    async def silent(text):  # the confirmation stays on screen
        leo.sent.append(text)
    leo.send_to_bot = silent
    leo.messages.append(BotMessage(id=leo._id(), date=1, text=text, buttons=reply_kb([buttons])))
    for _ in range(4):
        await steps(agents, 1)
        clock.t += 11
    assert not {"🚀 Продолжить за 99 ₽", "1", "ДА"} & set(leo.sent)


async def test_a_block_with_a_premium_offer_still_goes_to_the_owner(tmp_path):
    leo = FakeLeomatch()
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {"action": "ask_owner"})],
                                  mode="auto", **dict(LEO_GO, premium_pattern="Premium", decline_buttons=["Назад"]))
    leo.messages.append(BotMessage(id=leo._id(), date=1, text="Ваш аккаунт заблокирован. Разблокировка с Premium",
                                   buttons=reply_kb([["Назад"]])))
    await steps(agents, 1)
    assert "Назад" not in leo.sent and not agents["дайвинчик"].running


async def test_incoming_rating_presses_count_towards_the_loop_cap(tmp_path):
    leo = FakeLeomatch()
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {
        "action": "click", "button_id": "k0c0"})], mode="auto", **dict(
        LEO_GO, incoming_rating_pattern=r"оценил[аи]?\s+вас,?\s+на\s+(\d{1,2})\s*балл",
        incoming_open_buttons=["Посмотреть"]))
    agent = agents["дайвинчик"]

    async def silent(text):
        leo.sent.append(text)
    leo.send_to_bot = silent
    for _ in range(20):
        leo.messages.append(BotMessage(id=leo._id(), date=1, text="Кто-то оценил вас на 10 баллов",
                                       buttons=reply_kb([["Посмотреть"]])))
        await steps(agents, 1)
        if not agent.running:
            break
    assert not agent.running and leo.sent.count("Посмотреть") <= 12


async def test_an_illustrated_menu_is_not_a_rated_profile(tmp_path):
    leo = FakeLeomatch()
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {
        "action": "click", "button_id": "k0c0"})], mode="auto", **LEO_DIGITS)
    agent = agents["дайвинчик"]

    async def silent(text):
        leo.sent.append(text)
    leo.send_to_bot = silent
    for n in range(20):
        leo.messages.append(BotMessage(id=leo._id(), date=1, text="1. Посмотреть\n2. Пропустить", media_kind="photo",
                                       media_key=f"ad{n}", buttons=reply_kb([["1", "2"]])))
        await steps(agents, 1)
        if not agent.running:
            break
    assert not agent.running and leo.sent.count("1") <= 12


async def test_an_ad_about_not_getting_blocked_keeps_going(tmp_path):
    leo = FakeLeomatch()
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {"action": "ask_owner"})],
                                  mode="auto", **LEO_GO)
    leo.messages.append(BotMessage(id=leo._id(), date=1, text="Узнай, как не быть заблокированным в Дайвинчике",
                                   buttons=reply_kb([["🚀 Смотреть анкеты"]])))
    await steps(agents, 1)
    assert leo.sent[-1] == "🚀 Смотреть анкеты" and agents["дайвинчик"].running


@pytest.mark.parametrize("decision", [{"action": "click", "button_id": "k0c0"}, {"action": "ask_owner"}])
async def test_a_priced_skip_is_not_a_refusal(tmp_path, decision):
    leo = FakeLeomatch()
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: dict(decision))], mode="auto",
                                  **dict(LEO_GO, premium_pattern="Premium", decline_buttons=["Пропустить"]))
    leo.messages.append(BotMessage(id=leo._id(), date=1, text="Premium: больше лайков",
                                   buttons=reply_kb([["Пропустить за 10 ⭐"]])))
    await steps(agents, 1)
    assert "Пропустить за 10 ⭐" not in leo.sent


@pytest.mark.parametrize("text, opened", [("Кто-то оценил вас на 10 баллов. Просмотр за 10 звёзд с баланса", False),
                                          ("⭐️ Кто-то из вашего города, оценил вас, на 10 баллов", True)])
async def test_a_paid_look_at_an_incoming_rating_is_not_opened(tmp_path, text, opened):
    leo = FakeLeomatch()
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {"action": "ask_owner"})],
                                  mode="auto", **dict(LEO_GO, incoming_open_buttons=["Посмотреть"],
                                                      incoming_rating_pattern=r"оценил[аи]?\s+вас,?\s+на\s+(\d{1,2})\s*балл"))
    leo.messages.append(BotMessage(id=leo._id(), date=1, text=text, buttons=reply_kb([["Посмотреть"]])))
    await steps(agents, 1)
    assert ("Посмотреть" in leo.sent) is opened


async def test_a_purchase_prompt_after_a_profile_photo_is_not_a_rating(tmp_path):
    leo = FakeLeomatch()
    sup, agents, inbox, clock, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {
        "action": "click", "button_id": "k0c0"})], mode="auto", **LEO_DIGITS)

    async def silent(text):
        leo.sent.append(text)
    leo.send_to_bot = silent
    leo.messages.append(BotMessage(id=leo._id(), date=1, text="Аня", media_kind="photo", media_key="a"))
    leo.messages.append(BotMessage(id=leo._id(), date=1, text="Подтвердить буст за 10 звёзд? Нажмите 1",
                                   buttons=reply_kb([["1", "2"]])))
    for _ in range(4):
        await steps(agents, 1)
        clock.t += 11
    assert "1" not in leo.sent


# ------------------------------------------------------------ bot limits
LEO_LIMIT = dict(LEO_GO, limit_pattern=r"слишком много\W.{0,12}за сегодня|загляни к нам попозже", limit_retry_h=4,
                 resume_buttons=["Главное меню", "Смотреть анкеты"])
LIMIT_TEXT = "Слишком много ❤️ за сегодня.\n\nПерейди на Premium и получи больше ❤️!"
LIMIT_KB = [["Активируй Premium — будь в топе ⭐"], ["Главное меню"]]


def limited(tmp_path, decide=None, **extra):
    leo = FakeLeomatch()

    async def silent(text):  # the bot's answers are added by the test
        leo.sent.append(text)
    leo.send_to_bot = silent
    sup, agents, inbox, clock, *_ = make(tmp_path, bots=[("дайвинчик", leo, decide or (lambda t, i: {
        "action": "click", "button_id": "k0c0"}))], mode="auto", **dict(LEO_LIMIT, **extra))
    leo.messages.append(BotMessage(id=leo._id(), date=int(clock.t), text=LIMIT_TEXT, buttons=reply_kb(LIMIT_KB)))
    return sup, agents["дайвинчик"], leo, inbox, clock


async def test_a_bot_limit_pauses_without_the_model_and_retries_later(tmp_path):
    sup, agent, leo, inbox, clock = limited(tmp_path)
    await agent.step_safely()
    assert agent.llm.calls == [] and leo.sent == [] and not agent.active() and agent.running
    assert "Лимит анкет" in inbox.messages[-1] and "через 4 ч" in inbox.messages[-1]
    assert "лимит бота" in str(agent.status_text())
    clock.t += 3.9 * 3600
    await agent.step_safely()
    assert leo.sent == []
    clock.t += 0.2 * 3600
    await agent.step_safely()
    assert leo.sent == ["Главное меню"] and agent.llm.calls == []  # never the Premium button


async def test_still_limited_after_a_retry_pauses_again_quietly(tmp_path):
    sup, agent, leo, inbox, clock = limited(tmp_path)
    await agent.step_safely()
    told = len(inbox.messages)
    clock.t += 4 * 3600 + 1
    await agent.step_safely()
    leo.messages.append(BotMessage(id=leo._id(), date=int(clock.t), text=LIMIT_TEXT, buttons=reply_kb(LIMIT_KB)))
    await agent.step_safely()
    assert not agent.active() and len(inbox.messages) == told  # no second message within six hours
    clock.t += 4 * 3600 + 1
    await agent.step_safely()
    assert leo.sent == ["Главное меню", "Главное меню"]


async def test_profiles_after_a_retry_clear_the_limit(tmp_path):
    sup, agent, leo, inbox, clock = limited(tmp_path)
    await agent.step_safely()
    clock.t += 4 * 3600 + 1
    await agent.step_safely()
    leo.messages.append(BotMessage(id=leo._id(), date=2, text="Аня, Москва", media_kind="photo", media_key="a",
                                   buttons=reply_kb([["❤️", "👎", "💤"]])))
    await agent.step_safely()
    assert leo.sent[-1] == "❤️" and agent.store.get("limit") is None and "работает" in str(agent.status_text())


async def test_a_silent_bot_after_a_retry_means_another_interval(tmp_path):
    sup, agent, leo, inbox, clock = limited(tmp_path)
    await agent.step_safely()
    clock.t += 4 * 3600 + 1
    await agent.step_safely()
    clock.t += 30
    await agent.step_safely()
    assert agent.active()  # the bot may still answer
    clock.t += 31
    await agent.step_safely()
    assert not agent.active() and leo.sent == ["Главное меню"]
    clock.t += 4 * 3600 + 1
    await agent.step_safely()
    assert leo.sent == ["Главное меню", "Главное меню"]


async def test_without_auto_resume_it_stops_and_start_retries_at_once(tmp_path):
    sup, agent, leo, inbox, clock = limited(tmp_path, limit_resume=False)
    await agent.step_safely()
    assert not agent.running and "автопродолжение выключено" in inbox.messages[-1] and leo.sent == []
    await send(sup, "/start")
    await agent.step_safely()
    assert leo.sent == ["Главное меню"]


async def test_start_during_a_limit_pause_retries_at_once(tmp_path):
    sup, agent, leo, inbox, clock = limited(tmp_path)
    await agent.step_safely()
    await send(sup, "/start")
    assert "Запущен" in inbox.messages[-1]
    await agent.step_safely()
    assert leo.sent == ["Главное меню"]


async def test_turning_auto_resume_off_during_a_pause_stops_the_agent(tmp_path):
    sup, agent, leo, inbox, clock = limited(tmp_path)
    await agent.step_safely()
    await send(sup, "/set автопродолжение выкл")
    await agent.step_safely()
    assert not agent.running and leo.sent == []


async def test_the_retry_interval_is_a_setting(tmp_path):
    sup, agent, leo, inbox, clock = limited(tmp_path)
    await agent.step_safely()
    await send(sup, "/set после_лимита 0.1")
    assert agent.b.limit_retry_h == 4
    await send(sup, "/set после_лимита 1")
    clock.t += 3601
    await agent.step_safely()
    assert leo.sent == ["Главное меню"]


async def test_a_profile_caption_is_never_a_limit_notice(tmp_path):
    leo = FakeLeomatch()
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {
        "action": "click", "button_id": "k0c0"})], mode="auto", **LEO_LIMIT)
    leo.messages.append(BotMessage(id=leo._id(), date=1, text="Аня. Загляни к нам попозже, в кофейню :)",
                                   media_kind="photo", media_key="a", buttons=reply_kb([["❤️", "👎", "💤"]])))
    await steps(agents, 1)
    assert leo.sent[-1] == "❤️"


async def test_a_full_koala_resumes_with_the_rating_button_not_the_paid_one(tmp_path):
    leo = FakeLeomatch()

    async def silent(text):
        leo.sent.append(text)
    leo.send_to_bot = silent
    sup, agents, inbox, clock, *_ = make(tmp_path, bots=[("бибинто", leo, lambda t, i: {"action": "wait"})],
                                         mode="auto", limit_pattern="накушалась|возвращайся чуть позже",
                                         limit_retry_h=1, resume_buttons=["Оценивать"],
                                         continue_buttons=["Оценивать", "🚀"])
    agent = agents["бибинто"]
    leo.messages.append(BotMessage(id=leo._id(), date=int(clock.t), text="🐨 Коала накушалась твоих оценок!\n"
                                   "Возвращайся чуть позже\n\n⭐ Повышай уровень Коалы",
                                   buttons=reply_kb([["⭐ Улучшить коалу", "❤️ Vip-Status"], ["🚀Оценивать"]])))
    await agent.step_safely()
    assert "через 1 ч" in inbox.messages[-1]
    clock.t += 3601
    await agent.step_safely()
    assert leo.sent == ["🚀Оценивать"]


async def test_a_limit_notice_that_charges_is_left_with_a_restart(tmp_path):
    sup, agent, leo, inbox, clock = limited(tmp_path)
    leo.messages[-1] = replace(leo.messages[-1], text="Слишком много ❤️ за сегодня. Списать 99 звёзд? Нажмите 🚀",
                               buttons=reply_kb([["🚀"]]))
    await agent.step_safely()
    clock.t += 4 * 3600 + 1
    await agent.step_safely()
    assert leo.sent == ["/start"]


async def test_start_retries_at_once_even_on_a_new_notice(tmp_path):
    sup, agent, leo, inbox, clock = limited(tmp_path)
    await agent.step_safely()
    leo.messages.append(BotMessage(id=leo._id(), date=int(clock.t), text=LIMIT_TEXT, buttons=reply_kb(LIMIT_KB)))
    await send(sup, "/start")
    await agent.step_safely()
    assert leo.sent == ["Главное меню"]


async def test_auto_resume_off_stops_even_under_an_incoming_rating(tmp_path):
    sup, agent, leo, inbox, clock = limited(tmp_path, incoming_open_buttons=["Посмотреть"],
                                           incoming_rating_pattern=r"оценил[аи]?\s+вас,?\s+на\s+(\d{1,2})\s*балл")
    await agent.step_safely()
    await send(sup, "/set автопродолжение выкл")
    leo.messages.append(BotMessage(id=leo._id(), date=2, text="Кто-то оценил вас на 10 баллов",
                                   buttons=reply_kb([["Посмотреть"]])))
    await agent.step_safely()
    assert not agent.running and leo.sent == []


async def test_retries_respect_the_hourly_action_cap(tmp_path):
    sup, agent, leo, inbox, clock = limited(tmp_path, limit_retry_h=0.25, max_actions_per_hour=1)
    await agent.step_safely()
    for _ in range(4):
        clock.t += 0.25 * 3600 + 1
        await agent.step_safely()
        clock.t += 61
        await agent.step_safely()
    assert len(leo.sent) <= 1


async def test_start_right_after_a_retry_still_means_try_now(tmp_path):
    sup, agent, leo, inbox, clock = limited(tmp_path)
    await agent.step_safely()
    clock.t += 4 * 3600 + 1
    await agent.step_safely()  # retry: "Главное меню"
    leo.messages.append(BotMessage(id=leo._id(), date=int(clock.t), text=LIMIT_TEXT, buttons=reply_kb(LIMIT_KB)))
    await send(sup, "/start")
    await agent.step_safely()
    assert leo.sent == ["Главное меню", "Главное меню"]


async def test_an_old_notice_is_retried_at_once(tmp_path):
    sup, agent, leo, inbox, clock = limited(tmp_path)
    leo.messages[-1] = replace(leo.messages[-1], date=int(clock.t - 5 * 3600))  # the agent was stopped meanwhile
    await agent.step_safely()
    assert leo.sent == ["Главное меню"] and not any("Лимит анкет" in m for m in inbox.messages)


async def test_a_match_right_before_the_limit_is_still_reported(tmp_path):
    sup, agent, leo, inbox, clock = limited(tmp_path, match_pattern="Начинай общаться")
    leo.messages.insert(-1, BotMessage(id=leo._id(), date=int(clock.t), text="Есть взаимная симпатия! "
                                       "Начинай общаться 👉 @someone", links=("https://t.me/someone",)))
    await agent.step_safely()
    assert any("Взаимная симпатия" in m for m in inbox.messages)


# ------------------------------------------------- working hours, daily norm, warnings
from datetime import datetime  # noqa: E402
from zoneinfo import ZoneInfo  # noqa: E402

MSK = ZoneInfo("Europe/Moscow")


def at_msk(clock, day, hour, minute=0):
    clock.t = datetime(2027, 1, day, hour, minute, tzinfo=MSK).timestamp()


def cards(leo, n, start=0):
    for k in range(n):
        leo.messages.append(BotMessage(id=leo._id(), date=1, text=f"Анкета {start + k}", media_kind="photo",
                                       media_key=f"p{start + k}", buttons=reply_kb([["❤️", "👎", "💤"]])))


def liker(tmp_path, **extra):
    leo = FakeLeomatch()

    async def silent(text):
        leo.sent.append(text)
    leo.send_to_bot = silent
    sup, agents, inbox, clock, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {
        "action": "click", "button_id": "k0c0"})], mode="auto", **dict(LEO_GO, **extra))
    return sup, agents["дайвинчик"], leo, inbox, clock


async def test_working_hours_keep_the_bot_quiet_at_night(tmp_path):
    sup, agent, leo, inbox, clock = liker(tmp_path, active_hours_enabled=True)
    at_msk(clock, 15, 3)
    cards(leo, 1)
    await agent.step_safely()
    assert leo.sent == [] and agent.running and not agent.active()
    assert "нерабочее время" in str(agent.status_text()) and "в 10:00 MSK" in str(agent.status_text())
    at_msk(clock, 15, 10, 1)
    await agent.step_safely()
    assert leo.sent == ["❤️"]
    at_msk(clock, 15, 23, 0)
    cards(leo, 1, 1)
    await agent.step_safely()
    assert leo.sent == ["❤️"] and "завтра в 10:00" in str(agent.status_text())


async def test_working_hours_can_be_turned_off_and_changed(tmp_path):
    sup, agent, leo, inbox, clock = liker(tmp_path, active_hours_enabled=True)
    at_msk(clock, 15, 3)
    cards(leo, 1)
    await send(sup, "/set часы 22-04")
    assert agent.b.active_hours == "22:00-04:00"
    await agent.step_safely()
    assert leo.sent == ["❤️"]  # 03:00 is inside a window that crosses midnight
    await send(sup, "/set часы 9-9")
    assert agent.b.active_hours == "22:00-04:00"  # rejected
    at_msk(clock, 15, 12)
    cards(leo, 1, 1)
    await agent.step_safely()
    assert leo.sent == ["❤️"]
    await send(sup, "/set часы_работы выкл")
    await agent.step_safely()
    assert leo.sent == ["❤️", "❤️"]


async def test_the_time_zone_is_a_setting(tmp_path):
    sup, agent, leo, inbox, clock = liker(tmp_path, active_hours_enabled=True)
    at_msk(clock, 15, 8)  # 08:00 in Moscow is 10:00 at UTC+5
    cards(leo, 1)
    await agent.step_safely()
    assert leo.sent == []
    await send(sup, "/set пояс +5")
    await agent.step_safely()
    assert leo.sent == ["❤️"]
    await send(sup, "/set пояс Марс/Олимп")
    assert agent.b.timezone == "+5"


async def test_the_daily_norm_is_off_by_default(tmp_path):
    sup, agent, leo, inbox, clock = liker(tmp_path, daily_ratings=2)
    assert agent.b.daily_ratings_enabled is False
    at_msk(clock, 15, 12)
    for k in range(4):
        cards(leo, 1, k)
        await agent.step_safely()
    assert leo.sent.count("❤️") == 4


async def test_the_daily_norm_stops_until_the_next_working_day(tmp_path):
    sup, agent, leo, inbox, clock = liker(tmp_path, active_hours_enabled=True, daily_ratings_enabled=True,
                                          daily_ratings=3)
    at_msk(clock, 15, 20)
    for k in range(5):
        cards(leo, 1, k)
        await agent.step_safely()
        clock.t += 60
    assert leo.sent.count("❤️") == 3 and agent.running and not agent.active()
    told = [m for m in inbox.messages if "Дневная норма" in m]
    assert len(told) == 1 and "завтра в 10:00 MSK" in told[0]
    assert "дневная норма выполнена" in str(agent.status_text())
    at_msk(clock, 16, 1)  # a new day, but night
    await agent.step_safely()
    assert leo.sent.count("❤️") == 3
    at_msk(clock, 16, 10, 1)
    await agent.step_safely()
    assert leo.sent.count("❤️") == 4


async def test_the_daily_norm_is_a_setting(tmp_path):
    sup, agent, leo, inbox, clock = liker(tmp_path)
    at_msk(clock, 15, 12)
    await send(sup, "/set норма 2")
    await send(sup, "/set дневная_норма вкл")
    for k in range(3):
        cards(leo, 1, k)
        await agent.step_safely()
    assert leo.sent.count("❤️") == 2
    await send(sup, "/set норма 0")
    assert agent.b.daily_ratings == 2  # rejected
    await send(sup, "/set дневная_норма выкл")
    await agent.step_safely()
    assert leo.sent.count("❤️") == 3


@pytest.mark.parametrize("mode", ["auto", "supervised"])
async def test_a_bot_warning_pauses_a_few_seconds_then_goes_on(tmp_path, mode):
    sup, agent, leo, inbox, clock = liker(tmp_path)
    agent.b = replace(agent.b, mode=mode)
    leo.messages.append(BotMessage(id=leo._id(), date=1, text="Вы ставите лайки слишком быстро, отдохните",
                                   buttons=reply_kb([["🚀 Смотреть анкеты"]])))
    await agent.step_safely()
    assert leo.sent == [] and not agent.active() and agent.running and agent.llm.calls == []
    clock.t += 4
    await agent.step_safely()
    assert leo.sent == []
    clock.t += 5
    await agent.step_safely()
    assert leo.sent == ["🚀 Смотреть анкеты"] and agent.running
    assert not any("Нужна ваша помощь" in m for m in inbox.messages)


async def test_a_warning_that_asks_for_identity_still_goes_to_the_owner(tmp_path):
    sup, agent, leo, inbox, clock = liker(tmp_path)
    agent.llm.decide = lambda t, i: {"action": "ask_owner"}
    leo.messages.append(BotMessage(id=leo._id(), date=1, text="Подозрительная активность: подтвердите личность",
                                   buttons=reply_kb([["🚀 Смотреть анкеты"]])))
    await agent.step_safely()
    assert not agent.running and "Нужна ваша помощь" in inbox.messages[-1]


async def test_the_limit_notice_wins_over_a_warning(tmp_path):
    sup, agent, leo, inbox, clock = limited(tmp_path)
    leo.messages[-1] = replace(leo.messages[-1], text="Слишком много лайков за сегодня")
    await agent.step_safely()
    assert "Лимит анкет" in inbox.messages[-1]


def test_defaults_hours_on_norm_off():
    from pathlib import Path

    from doppelgram.config import BehaviorConfig as BC
    from doppelgram.schedule import parse_tz, parse_window
    d = BC(task_file=Path("t"), taste_file=Path("t"))
    assert d.active_hours_enabled and not d.daily_ratings_enabled
    assert parse_window(d.active_hours) == (600, 1380) and parse_tz(d.timezone)


async def test_an_edited_warning_is_a_new_warning(tmp_path):
    sup, agent, leo, inbox, clock = liker(tmp_path)
    agent.b = replace(agent.b, mode="supervised")
    agent.llm.decide = lambda t, i: {"action": "ask_owner"}
    leo.messages.append(BotMessage(id=leo._id(), date=1, text="Слишком быстро!",
                                   buttons=reply_kb([["🚀 Смотреть анкеты"]])))
    await agent.step_safely()
    clock.t += 9
    await agent.step_safely()
    leo.messages[-1] = replace(leo.messages[-1], text="Снова слишком быстро, притормози")
    await agent.step_safely()
    clock.t += 9
    await agent.step_safely()
    assert leo.sent == ["🚀 Смотреть анкеты"] * 2 and agent.running


async def test_a_cancelled_continue_after_a_warning_is_tried_again(tmp_path):
    sup, agent, leo, inbox, clock = liker(tmp_path)
    agent.b = replace(agent.b, mode="supervised")
    agent.llm.decide = lambda t, i: {"action": "ask_owner"}  # the model would stop: the rule must retry
    leo.messages.append(BotMessage(id=leo._id(), date=1, text="Слишком быстро!",
                                   buttons=reply_kb([["🚀 Смотреть анкеты"]])))
    await agent.step_safely()
    clock.t += 9
    original = agent._execute

    async def cancelled(*a, **k):
        return False
    agent._execute = cancelled
    await agent.step_safely()
    agent._execute = original
    await agent.step_safely()
    assert leo.sent == ["🚀 Смотреть анкеты"]


async def test_warnings_respect_the_hourly_cap(tmp_path):
    sup, agent, leo, inbox, clock = liker(tmp_path, max_actions_per_hour=1)
    agent.store.log_action(clock(), None, None, "click", button_text="❤️")
    leo.messages.append(BotMessage(id=leo._id(), date=1, text="Слишком быстро!",
                                   buttons=reply_kb([["🚀 Смотреть анкеты"]])))
    await agent.step_safely()
    clock.t += 9
    await agent.step_safely()
    assert leo.sent == []


# ------------------------------------------------------------- open-source review regressions
async def test_a_profile_caption_cannot_fake_a_match(tmp_path):
    leo = FakeLeomatch()
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {
        "action": "click", "button_id": "k0c0"})], mode="auto", **dict(LEO_GO, match_pattern="Начинай общаться"))
    leo.messages.append(BotMessage(id=leo._id(), date=1, text="Начинай общаться 👉 @stranger_example",
                                   links=("https://t.me/stranger_example",), media_kind="photo", media_key="x",
                                   buttons=reply_kb([["❤️", "👎", "💤"]])))
    await steps(agents, 1)
    assert agents["дайвинчик"].store.db.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 0
    assert not any("Взаимная симпатия" in m for m in inbox.messages)


async def test_stop_cancels_an_owner_confirmed_press_during_the_delay(tmp_path):
    sup, agents, inbox, *_ = make(tmp_path, mode="supervised", action_delay_min_s=1, action_delay_max_s=1)
    agent, bot = agents["арт"], agents["арт"].tg
    await steps(agents, 2)
    card = agent.store.latest_pending().control_msg_id
    real_sleep = agent.sleep

    async def stop_during_delay(seconds):
        agent.stop_now()
        await real_sleep(seconds)
    agent.sleep = stop_during_delay
    await send(sup, "2", reply_to=card)
    assert bot.ratings == []


async def test_a_reply_keyboard_far_back_is_still_seen(tmp_path):
    leo = FakeLeomatch()
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {
        "action": "click", "button_id": "k0c0"})], mode="auto", **LEO_GO)
    leo.messages.append(BotMessage(id=leo._id(), date=1, text="Меню", buttons=reply_kb([["🚀 Смотреть анкеты"]])))
    for k in range(12):  # the bot keeps the keyboard and sends more messages without markup
        leo.messages.append(BotMessage(id=leo._id(), date=1, text=f"Новость {k}"))
    await steps(agents, 1)
    assert leo.sent[-1:] == ["🚀 Смотреть анкеты"]


@pytest.mark.parametrize("text, sent", [("1", True), ("2", False)])
async def test_a_free_item_of_a_text_menu_with_a_paid_one_is_typed(tmp_path, text, sent):
    leo = FakeLeomatch()

    async def silent(t):
        leo.sent.append(t)
    leo.send_to_bot = silent
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {"action": "send", "text": text})],
                                  mode="auto", **dict(LEO_GO, deny_buttons=["Премиум"]))
    leo.messages.append(BotMessage(id=leo._id(), date=1, text="1. Смотреть анкеты\n2. Купить Премиум за 99 ₽"))
    await steps(agents, 1)
    assert (text in leo.sent) is sent


async def test_an_owner_answer_after_working_hours_still_presses(tmp_path):
    sup, agents, inbox, clock, *_ = make(tmp_path, mode="supervised", active_hours_enabled=True,
                                         active_hours="0-24")
    agent, bot = agents["арт"], agents["арт"].tg
    await steps(agents, 2)
    card = agent.store.latest_pending().control_msg_id
    agent.b = replace(agent.b, active_hours="01:00-01:01")  # now outside working hours
    await send(sup, "2", reply_to=card)
    assert bot.ratings == ["2"]


async def test_a_digit_rating_is_the_score_passed_with_a_match(tmp_path):
    leo = FakeLeomatch()

    async def silent(text):
        leo.sent.append(text)
    leo.send_to_bot = silent
    sup, agents, inbox, *_ = make(tmp_path, bots=[("бибинто", leo, lambda t, i: {
        "action": "click", "button_id": "k0c6", "score": 3})], mode="auto",
        **dict(LEO_DIGITS, match_pattern="Начинай общаться"))
    agent = agents["бибинто"]
    got = []

    async def on_match(*args, **kwargs):
        got.append(args)
    agent.on_match = on_match
    leo.messages.append(BotMessage(id=leo._id(), date=1, text="Аня, Москва", media_kind="photo", media_key="a",
                                   buttons=reply_kb([[str(n) for n in range(1, 11)]])))
    await agent.step_safely()
    assert leo.sent == ["7"]
    leo.messages.append(BotMessage(id=leo._id(), date=2, text="Начинай общаться 👉 Аня",
                                   links=("https://t.me/anya_example",)))
    await agent.step_safely()
    assert got and got[0][-1] == 7  # the pressed rating, not the model's separate estimate


async def test_the_score_survives_an_owner_confirmation(tmp_path):
    leo = FakeLeomatch()

    async def silent(text):
        leo.sent.append(text)
    leo.send_to_bot = silent
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", leo, lambda t, i: {
        "action": "click", "button_id": "k0c0", "score": 9})], mode="supervised",
        **dict(LEO_GO, match_pattern="Начинай общаться"))
    agent = agents["дайвинчик"]
    got = []

    async def on_match(*args, **kwargs):
        got.append(args)
    agent.on_match = on_match
    leo.messages.append(BotMessage(id=leo._id(), date=1, text="Аня, Москва", media_kind="photo", media_key="a",
                                   buttons=reply_kb([["❤️", "👎", "💤"]])))
    await agent.step_safely()
    await send(sup, "ок", reply_to=agent.store.latest_pending().control_msg_id)
    leo.messages.append(BotMessage(id=leo._id(), date=2, text="Начинай общаться 👉 Аня",
                                   links=("https://t.me/anya_example",)))
    await agent.step_safely()
    assert leo.sent == ["❤️"] and got and got[0][-1] == 9


async def test_a_taste_check_offers_only_the_rating_buttons(tmp_path):
    bib = FakeBibinto(people=["Аня"])
    sup, agents, inbox, *_ = make(tmp_path, bots=[("бибинто", bib, sure)], mode="auto",
                                  spot_check_rate=1.0, spot_checks_per_hour=5)
    await steps(agents, 2)
    check = next(m for m in inbox.messages if "Проверка вкуса" in m)
    assert "Варианты: 1 · 2 · 3 · 4 · 5 · 6 · 7 · 8 · 9 · 10" in check
    assert "Назад" not in check and "11" not in check


def test_leomatch_options_are_like_and_dislike_only(tmp_path):
    from doppelgram.models import BotMessage
    sup, agents, inbox, *_ = make(tmp_path, bots=[("дайвинчик", FakeLeomatch(), sure)], mode="supervised")
    msg = BotMessage(id=1, text="Аня, 22", media_kind="photo", media_key="p",
                     buttons=reply_kb([["❤️", "💌 / 📹", "👎", "💤"]]))
    choices, labels, listing = agents["дайвинчик"]._options(msg)
    assert labels == ["❤️", "👎"] and str(listing) == "1 ❤️ · 2 👎"
    menu = BotMessage(id=2, text="Меню", buttons=reply_kb([["1", "2", "3"], ["Назад"]]))
    assert agents["дайвинчик"]._options(menu)[1] == ["1", "2", "3"]
    other = BotMessage(id=3, text="?", buttons=reply_kb([["Да", "Нет"]]))
    assert agents["дайвинчик"]._options(other)[1] == ["Да", "Нет"]  # no ratings: everything is offered
