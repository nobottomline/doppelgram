from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest
from fakes import png_bytes
from telethon.tl import types

from doppelgram import brain
from doppelgram.config import LLMConfig
from doppelgram.control import parse_command, parse_duration
from doppelgram.llm import LLMClient, LLMError, extract_json
from doppelgram.media import image_data_url, to_jpeg
from doppelgram.models import BotMessage, Button
from doppelgram.telegram import normalize, parse_markup


# ------------------------------------------------------------------ control
@pytest.mark.parametrize("text,seconds", [("2ч", 7200), ("30м", 1800), ("1h30m", 5400), ("90", 5400),
                                          ("1,5ч", 5400), ("2 часа", 7200), ("", None)])
def test_parse_duration(text, seconds):
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("text,name,arg", [
    ("старт", "start", None), ("СТОП", "stop", None), ("пауза 2ч", "pause", 7200),
    ("пауза", "pause", 3600), ("режим авто", "mode", "auto"), ("режим учёба", "mode", "supervised"),
    ("режим чего", "unknown", "режим чего"), ("журнал 5", "log", 5), ("7", "choose", 7),
    ("ок", "accept", None), ("+", "accept", None), ("пропуск", "skip", None), ("привет", "unknown", "привет"),
])
def test_parse_command(text, name, arg):
    cmd = parse_command(text)
    assert (cmd.name, cmd.arg) == (name, arg)


def test_choose_keeps_note():
    assert parse_command("3 руки кривые").note == "руки кривые"


# --------------------------------------------------------- telethon markup
def _now():
    return datetime(2026, 11, 7, tzinfo=UTC)


def test_parse_inline_markup_current_layer():
    markup = types.ReplyInlineMarkup(rows=[
        types.KeyboardInlineButtonRow(buttons=[
            types.KeyboardInlineButton(text="⭐", type=types.InlineButtonTypeCallback(data=b"r:1")),
            types.KeyboardInlineButton(text="pw", type=types.InlineButtonTypeCallback(
                data=b"p", requires_password=True)),
        ]),
        types.KeyboardInlineButtonRow(buttons=[
            types.KeyboardInlineButton(text="site", type=types.InlineButtonTypeUrl(url="https://a")),
        ]),
    ])
    b = parse_markup(markup)
    assert [(x.id, x.kind, x.text, x.data, x.clickable) for x in b] == [
        ("r0c0", "callback", "⭐", b"r:1", True),
        ("r0c1", "callback", "pw", b"p", False),
        ("r1c0", "url", "site", None, False),
    ]
    assert b[2].url == "https://a"


def test_parse_reply_keyboard_never_shares_phone():
    markup = types.ReplyKeyboardMarkup(rows=[types.KeyboardButtonRow(buttons=[
        types.KeyboardButton(text="Смотреть анкеты", type=types.ButtonTypeDefault()),
        types.KeyboardButton(text="Отправить номер", type=types.ButtonTypeRequestPhone()),
    ])])
    b = parse_markup(markup)
    assert [(x.kind, x.clickable) for x in b] == [("reply", True), ("other", False)]


def test_normalize_message_with_photo_and_edit_changes_revision():
    photo = types.Photo(id=77, access_hash=1, file_reference=b"", date=_now(), sizes=[], dc_id=2)
    kb = types.ReplyInlineMarkup(rows=[types.KeyboardInlineButtonRow(buttons=[
        types.KeyboardInlineButton(text="1", type=types.InlineButtonTypeCallback(data=b"a"))])])
    m = types.Message(id=5, peer_id=types.PeerUser(1), date=_now(), message="Карточка",
                      media=types.MessageMediaPhoto(photo=photo), reply_markup=kb)
    n = normalize(m)
    assert (n.id, n.text, n.media_kind, n.has_image, n.outgoing) == (5, "Карточка", "photo", True, False)
    m2 = types.Message(id=5, peer_id=types.PeerUser(1), date=_now(), message="Карточка",
                       media=types.MessageMediaPhoto(photo=types.Photo(
                           id=78, access_hash=1, file_reference=b"", date=_now(), sizes=[], dc_id=2)),
                       reply_markup=kb, edit_date=_now())
    assert normalize(m2).revision != n.revision  # same text, new picture → new revision


def test_sticker_classification():
    doc = types.Document(id=9, access_hash=1, file_reference=b"", date=_now(), mime_type="application/x-tgsticker",
                         size=10, dc_id=2, attributes=[types.DocumentAttributeSticker(
                             alt="😀", stickerset=types.InputStickerSetEmpty())])
    m = types.Message(id=6, peer_id=types.PeerUser(1), date=_now(), message="",
                      media=types.MessageMediaDocument(document=doc))
    assert normalize(m).media_kind == "sticker"


# -------------------------------------------------------------------- brain
def _msg():
    return BotMessage(id=1, text="t", buttons=(
        Button("r0c0", 0, 0, "1", "callback", b"1"), Button("r0c1", 0, 1, "url", "url")))


def test_validate_click_and_rejections():
    m = _msg()
    d = brain.validate_decision({"action": "click", "button_id": "r0c0", "confidence": 3}, m,
                                max_wait_s=100, max_send_chars=10)
    assert (d.action, d.button_id, d.confidence) == ("click", "r0c0", 1.0)
    for bad in ({"action": "click", "button_id": "nope"}, {"action": "click", "button_id": "r0c1"},
                {"action": "fly"}, {"action": "send", "text": ""}, {"action": "send", "text": "x" * 11}):
        assert brain.validate_decision(bad, m, max_wait_s=100, max_send_chars=10).action == "ask_owner"
    w = brain.validate_decision({"action": "wait", "wait_seconds": 99999}, m, max_wait_s=100, max_send_chars=10)
    assert w.wait_seconds == 100


def test_prompt_keeps_callback_data_away_from_model():
    msgs = brain.build_messages(task="t", taste="", examples=[], history=[], recent_actions=[],
                                current=_msg(), image_url=None, image_from_previous=False, notes=[])
    assert "callback" in msgs[1]["content"][0]["text"]
    assert "b'1'" not in json.dumps(msgs, ensure_ascii=False)


# ---------------------------------------------------------------------- llm
@pytest.mark.parametrize("raw", ['{"a": 1}', '```json\n{"a": 1}\n```', 'Sure! {"a": 1} done'])
def test_extract_json(raw):
    assert extract_json(raw) == {"a": 1}


def test_extract_json_rejects_garbage():
    with pytest.raises(LLMError):
        extract_json("no json here")


async def test_llm_fallback_and_cost():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if body["model"] == "primary":
            return httpx.Response(503, text="overloaded")
        return httpx.Response(200, json={"model": "backup", "choices": [{"message": {
            "content": '{"action": "wait"}'}}], "usage": {"prompt_tokens": 1000, "completion_tokens": 200}})

    cfg = LLMConfig(base_url="http://llm.test/v1", api_key="k", model="primary", fallback_models=["backup"],
                    price_in_per_mtok=0.1, price_out_per_mtok=0.5)
    client = LLMClient(cfg, http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    r = await client.complete_json([{"role": "user", "content": "hi"}])
    assert r.data == {"action": "wait"} and r.model == "backup"
    assert r.cost_usd == pytest.approx((1000 * 0.1 + 200 * 0.5) / 1e6)
    assert [b["model"] for b in seen] == ["primary", "backup"]
    assert seen[0]["response_format"] == {"type": "json_object"}
    assert "usage" not in seen[0]  # only sent to OpenRouter


def test_openrouter_reported_cost_wins():
    cfg = LLMConfig(base_url="https://openrouter.ai/api/v1", api_key="k", model="m", price_in_per_mtok=99)
    c = LLMClient(cfg, http=httpx.AsyncClient())
    assert c._cost({"cost": 0.0002, "prompt_tokens": 10}) == 0.0002
    assert c._body("m", [])["usage"] == {"include": True}


# -------------------------------------------------------------------- media
def test_images_are_normalized_to_jpeg():
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGBA", (4000, 2000), (0, 0, 0, 0)).save(buf, format="WEBP")
    out = to_jpeg(buf.getvalue(), 1000)
    img = Image.open(io.BytesIO(out))
    assert img.format == "JPEG" and max(img.size) == 1000
    assert image_data_url(png_bytes(), 512).startswith("data:image/jpeg;base64,")


def test_example_config_loads(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path

    from doppelgram.config import load_config

    root = Path(__file__).parent.parent
    shutil.copy(root / "config.example.toml", tmp_path / "config.toml")
    monkeypatch.setenv("TG_API_ID", "123")
    monkeypatch.setenv("TG_API_HASH", "h")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    cfg = load_config(tmp_path / "config.toml")
    assert cfg.telegram.api_id == 123 and cfg.llm.model == "openai/gpt-6-luna"
    assert cfg.behavior.start_command == "/start"
    leo, bib = cfg.targets
    assert (leo.name, leo.bot, leo.behavior.mode, leo.behavior.task_file.name) == (
        "дайвинчик", "@leomatchbot", "auto", "task-leomatch.md")
    assert bib.behavior.mode == "auto" and cfg.behavior.mode == "auto"
    assert (bib.name, bib.db_path.name) == ("бибинто", "бибинто.sqlite3")
    assert leo.behavior.taste_profile == bib.behavior.taste_profile == "default"
    import re
    assert leo.behavior.age_in_prompt == "18" and leo.behavior.min_age == 0
    assert re.search(leo.behavior.age_pattern, "Аня, 17, Москва – люблю котов", re.M).group(1) == "17"
    assert re.search(leo.behavior.age_pattern, "Аня, 21, 📍3 км", re.M).group(1) == "21"
    assert re.search(bib.behavior.age_pattern, "👩‍🦰: Аня, 19 лет (✅)\n🌇Город: Москва").group(1) == "19"
    assert "💌" in leo.behavior.deny_buttons
    assert re.search(bib.behavior.incoming_rating_pattern, "⭐️ Кто-то из вашего города, оценил вас, на 10 баллов 😮",
                     re.I).group(1) == "10"
    assert bib.behavior.incoming_open_buttons == ["Посмотреть"] and "Отключить" in bib.behavior.deny_buttons
    assert cfg.conversations.write_first is False and cfg.conversations.write_first_min_score == 9
    likes = bib
    conv = cfg.conversations
    assert conv is not None and not conv.enabled and conv.mode == "approve"
    assert conv.about_file.name == "about-me.md" and conv.timezone == "Europe/Moscow"
    assert likes.behavior.daily_budget_usd == cfg.behavior.daily_budget_usd
    assert cfg.telegram.session == (tmp_path / "data" / "account").resolve()


async def test_unparseable_paid_reply_is_billed():
    def handler(request):
        body = json.loads(request.content)
        if body["model"] == "primary":
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"action": "cl'}}],
                                             "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.25}})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"action": "wait"}'}}],
                                         "usage": {"cost": 0.001}})

    cfg = LLMConfig(base_url="https://openrouter.ai/api/v1", api_key="k", model="primary",
                    fallback_models=["backup"])
    client = LLMClient(cfg, http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    r = await client.complete_json([])
    assert r.cost_usd == 0.001
    assert [f.cost_usd for f in r.failures] == [0.25]


async def test_owner_filter_rejects_groups_and_strangers():
    from types import SimpleNamespace

    from doppelgram.telegram import TelegramAccount

    seen = []

    class RouterStub:
        async def handle_owner(self, om, from_saved_messages=False):
            seen.append(om.text)

        def on_bot_event(self, name):
            seen.append(f"bot:{name}")

    cfg = SimpleNamespace(owner_ids=[42])
    t = TelegramAccount.__new__(TelegramAccount)
    t.cfg, t.router, t.owner_peer, t._tasks = cfg, RouterStub(), object(), set()
    t.me = SimpleNamespace(id=7)
    t.channels = {500: SimpleNamespace(target=SimpleNamespace(name="likes"))}

    def ev(text, private, sender, chat):
        return SimpleNamespace(is_private=private, sender_id=sender, chat_id=chat,
                               message=SimpleNamespace(id=1, message=text, reply_to_msg_id=None))

    await t._on_owner(ev("стоп", False, 99, -42))      # stranger in a group whose raw id is 42
    await t._on_owner(ev("стоп", True, 99, 99))        # stranger in DM (should not even match the filter)
    await t._on_owner(ev("старт", True, 42, 42))       # the owner
    await t._on_saved(ev("статус", False, 7, -100))    # own message in some group
    await t._on_saved(ev("пауза", True, 7, 7))         # Saved Messages
    await t._on_bot(ev("", False, 1, -500))            # group with the bot's raw id
    await t._on_bot(ev("", True, 500, 500))            # the bot chat
    import asyncio
    await asyncio.sleep(0)
    assert seen == ["bot:likes", "старт", "пауза"]


async def test_concurrent_calls_keep_their_own_failures():
    import asyncio

    gate = asyncio.Event()

    async def handler(request):
        body = json.loads(request.content)
        if body["messages"][0]["content"] == "A" and body["model"] == "primary":
            await gate.wait()  # A's first attempt is slow and then fails
            return httpx.Response(200, json={"choices": [{"message": {"content": "garbage"}}],
                                             "usage": {"cost": 0.25}})
        if body["messages"][0]["content"] == "B":
            gate.set()
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"action": "wait"}'}}],
                                         "usage": {"cost": 0.01}})

    cfg = LLMConfig(base_url="https://openrouter.ai/api/v1", api_key="k", model="primary",
                    fallback_models=["backup"])
    client = LLMClient(cfg, http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    a, b = await asyncio.gather(client.complete_json([{"role": "user", "content": "A"}]),
                                client.complete_json([{"role": "user", "content": "B"}]))
    assert [f.cost_usd for f in a.failures] == [0.25] and b.failures == []


async def test_same_bot_under_two_spellings_is_rejected():
    from types import SimpleNamespace

    from doppelgram.telegram import TelegramAccount
    from doppelgram.transport import TransportError

    class Client:
        async def connect(self): pass
        async def is_user_authorized(self): return True
        async def get_me(self): return SimpleNamespace(id=1, username="me", first_name="Me")
        async def get_entity(self, ref): return SimpleNamespace(id=777, bot=True)
        async def get_input_entity(self, e): return e

    targets = [SimpleNamespace(name="a", bot="@same_bot"), SimpleNamespace(name="b", bot="https://t.me/same_bot")]
    acc = TelegramAccount(SimpleNamespace(targets=targets, owner_ids=[]), client=Client())
    with pytest.raises(TransportError, match="same Telegram bot"):
        await acc.start()


@pytest.mark.parametrize("text", ["30 м", "1 час", "2 часа 30 минут", "1h30m"])
def test_spaced_durations_are_valid(text):
    assert parse_duration(text) is not None


@pytest.mark.parametrize("text", ["2ч арт3", "арт3", "завтра"])
def test_non_durations_are_rejected(text):
    assert parse_duration(text) is None


def test_nan_confidence_is_treated_as_unknown():
    d = brain.validate_decision({"action": "click", "button_id": "r0c0", "confidence": "NaN"}, _msg(),
                                max_wait_s=100, max_send_chars=10)
    assert d.confidence is None


def test_duration_parsing_is_linear_on_adversarial_input():
    import time

    start = time.perf_counter()
    for n in (24, 60, 5000):
        assert parse_duration("1" * n + "x") is None
        assert parse_command("пауза " + "1" * n + "x").name == "pause"
    assert time.perf_counter() - start < 0.05


def test_extract_json_is_fast_on_unclosed_fence():
    import time

    start = time.perf_counter()
    with pytest.raises(LLMError):
        extract_json("```json" + " " * 200_000 + "{")
    assert extract_json("```JSON\n{\"a\": 1}\n```") == {"a": 1}
    assert time.perf_counter() - start < 0.05


# ---- OpenAI-compatible providers --------------------------------------------------------------
def _client(handler, **cfg):
    base = dict(base_url="http://localhost:11434/v1", api_key="", model="m")
    base.update(cfg)
    return LLMClient(LLMConfig(**base), http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def test_rejected_params_are_adapted_and_remembered():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "response_format" in body:
            return httpx.Response(400, json={"error": {"message": "response_format is not supported"}})
        if "max_tokens" in body:
            return httpx.Response(400, json={"error": {"message": "Unsupported parameter: 'max_tokens'. "
                                                                  "Use 'max_completion_tokens' instead."}})
        if "temperature" in body:
            return httpx.Response(400, json={"error": {"message": "temperature does not support 0.2"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"action": "wait"}'}}]})

    c = _client(handler)
    r = await c.complete_json([{"role": "user", "content": "x"}])
    assert r.data == {"action": "wait"} and r.failures == []
    assert bodies[-1]["max_completion_tokens"] == 1500 and "temperature" not in bodies[-1]
    n = len(bodies)
    await c.complete_json([{"role": "user", "content": "x"}])
    assert len(bodies) == n + 1  # adaptations are remembered per model


async def test_unrelated_400_is_not_retried_forever():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={"error": {"message": "invalid image"}})

    with pytest.raises(LLMError):
        await _client(handler).complete_json([])
    assert len(calls) == 1


async def test_content_parts_think_blocks_and_no_key():
    seen_headers = []

    def handler(request):
        seen_headers.append(dict(request.headers))
        return httpx.Response(200, json={"choices": [{"message": {"content": [
            {"type": "text", "text": "<think>hmm {not json}</think>"}, {"type": "text", "text": '{"a": 1}'}]}}]})

    r = await _client(handler).complete_json([])
    assert r.data == {"a": 1}
    assert "authorization" not in seen_headers[0]
    assert str(_client(handler, base_url="http://x/v1/chat/completions/")._url) == "http://x/v1/chat/completions"


def test_llm_config_without_key(tmp_path, monkeypatch):
    from doppelgram.config import load_config

    (tmp_path / "config.toml").write_text('[llm]\nbase_url = "http://localhost:11434/v1"\napi_key_env = ""\n'
                                          'model = "qwen3-vl"\n', encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = load_config(tmp_path / "config.toml", require_telegram=False)
    assert cfg.llm.api_key == "" and cfg.llm.model == "qwen3-vl"


async def test_restriction_is_never_swallowed_by_transport_handlers():
    from types import SimpleNamespace

    from doppelgram.telegram import PeopleChannel, TelegramAccount
    from doppelgram.transport import AccountRestricted

    class Client:
        async def get_entity(self, ref):
            raise AccountRestricted("PeerFloodError")

        async def download_media(self, *a, **k):
            raise AccountRestricted("PeerFloodError")

    acc = TelegramAccount.__new__(TelegramAccount)
    acc.client, acc.cfg, acc.me = Client(), SimpleNamespace(owner_ids=[]), None
    with pytest.raises(AccountRestricted):
        await PeopleChannel(acc).resolve(["https://t.me/someone_here"], [])

    tripped = []

    class Router:
        async def restrict_all(self, reason):
            tripped.append(reason)

    acc.router = Router()
    from datetime import datetime

    from telethon.tl import types
    p = types.Photo(id=1, access_hash=1, file_reference=b"", date=datetime.now(UTC), sizes=[], dc_id=2)
    msg = types.Message(id=1, peer_id=types.PeerUser(1), date=datetime.now(UTC), message="+",
                        media=types.MessageMediaPhoto(photo=p))
    await acc._deliver(SimpleNamespace(message=msg, chat_id=42), from_saved_messages=False)
    assert tripped == ["PeerFloodError"]


# ---- ages shown to the LLM provider -----------------------------------------------------------
LEO_AGE = r"^[^,\n]+,\s*(\d{1,2}),"
BIB_AGE = r"(\d{1,2})\s*лет"


@pytest.mark.parametrize("text,mode,pattern,expected", [
    ("Аня, 23, Москва – люблю горы, мне 23 года", "keep", LEO_AGE, "Аня, 23, Москва – люблю горы, мне 23 года"),
    ("Аня, 23, Москва – люблю горы, мне 23 года", "18", LEO_AGE, "Аня, 18, Москва – люблю горы, мне 18 года"),
    ("Аня, 23, Москва – рост 170, 2 кота", "18", LEO_AGE, "Аня, 18, Москва – рост 170, 2 кота"),
    ("👩‍🦰: Аня, 22 лет (✅)\n🌇Город: Москва", "18", BIB_AGE, "👩‍🦰: Аня, 18 лет (✅)\n🌇Город: Москва"),
    ("Аня, 23, Москва – рост 170, 2 кота", "hide", LEO_AGE, "Аня, Москва – рост , кота"),
    ("Аня, 23, 📍3 км", "hide", LEO_AGE, "Аня, 📍 км"),
    ("", "hide", LEO_AGE, ""),
])
def test_sanitize_age(text, mode, pattern, expected):
    from doppelgram.brain import sanitize_age
    assert sanitize_age(text, mode, pattern) == expected


def test_invalid_age_mode_is_rejected(tmp_path):
    from doppelgram.config import ConfigError, load_config

    (tmp_path / "config.toml").write_text('[behavior]\nage_in_prompt = "maybe"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="age_in_prompt"):
        load_config(tmp_path / "config.toml", require_llm=False, require_telegram=False)


@pytest.mark.parametrize("text,expected", [
    ("мне 23 годика", "мне 18 годика"), ("23y, love travel", "18y, love travel"), ("23 years old", "18 years old"),
    ("Кому-то понравилась твоя анкета:\n\nМаша, 24, Тула – привет\n\nОля, 25, Тверь – хай",
     "Кому-то понравилась твоя анкета:\n\nМаша, 18, Тула – привет\n\nОля, 18, Тверь – хай"),
    ("рост 170, 2 кота", "рост 170, 2 кота"),
])
def test_sanitize_age_variants(text, expected):
    from doppelgram.brain import sanitize_age
    assert sanitize_age(text, "18", LEO_AGE) == expected


def test_every_prompt_text_goes_through_the_sanitizer():
    from doppelgram import brain
    from doppelgram.models import BotMessage, Button

    clean = lambda t: brain.sanitize_age(t, "18", LEO_AGE)  # noqa: E731
    cur = BotMessage(id=2, text="Аня, 23, Москва", buttons=(
        Button("k0c0", 0, 0, "Аня, 23 года", "reply", inline=False),))
    msgs = brain.build_messages(
        task="t", taste="", current=cur, image_url=None, image_from_previous=False,
        history=[BotMessage(id=1, text="Маша, 24, Тула – хай")],
        examples=[{"description": "девушка 23 лет", "model_choice": "❤️", "owner_choice": "❤️", "note": "ей 23 года"}],
        recent_actions=[{"action": "send", "button_text": None, "text": "мне 25 лет", "msg_id": 1,
                         "result": "answer: Ане 23 года"}],
        notes=['After the last click the bot showed: "Ане 23 года"'], sanitize=clean)
    text = msgs[1]["content"][0]["text"]
    for real in ("23", "24", "25"):
        assert real not in text, real


def test_example_choices_are_sanitized_too():
    from doppelgram import brain
    from doppelgram.models import BotMessage

    clean = lambda t: brain.sanitize_age(t, "18", LEO_AGE)  # noqa: E731
    msgs = brain.build_messages(
        task="t", taste="", current=BotMessage(id=1, text="x"), image_url=None, image_from_previous=False,
        history=[], recent_actions=[], notes=[], sanitize=clean,
        examples=[{"description": "", "model_choice": "Маша, 23 года", "owner_choice": "Маша, 23 года", "note": ""}])
    assert "23" not in msgs[1]["content"][0]["text"]


# ------------------------------------------------------------- open-source review regressions
def test_the_model_may_type_only_allowed_commands_or_menu_numbers():
    from doppelgram import brain
    from doppelgram.models import BotMessage
    m = BotMessage(id=1, text="Меню")
    ok = lambda text: brain.validate_decision({"action": "send", "text": text}, m, max_wait_s=60,  # noqa: E731
                                              max_send_chars=200, allowed_commands=("/start",)).action == "send"
    assert ok("/start") and ok("2") and ok("12")
    assert not ok("OWNER_PRIVATE_MARKER") and not ok("/start мой вкус: блондинки") and not ok("/myprofile")


def test_session_files_follow_telethon_naming(tmp_path):
    from doppelgram.cli import _session_files
    assert _session_files(tmp_path / "account") == [tmp_path / "account.session", tmp_path / "account.session-journal"]
    assert _session_files(tmp_path / "a.session") == [tmp_path / "a.session", tmp_path / "a.session-journal"]


@pytest.mark.parametrize("name", ["aux", "con", "nul", "com1", "lpt9"])
def test_windows_reserved_bot_names_are_rejected(tmp_path, name):
    from doppelgram.config import ConfigError, load_config
    (tmp_path / "config.toml").write_text(f'[[bots]]\nname = "{name}"\nusername = "@x_bot"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="reserved"):
        load_config(tmp_path / "config.toml", require_llm=False, require_telegram=False)


def test_exif_rotation_is_applied():
    import io

    from PIL import Image

    from doppelgram.media import to_jpeg
    img = Image.new("RGB", (40, 20))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90° to display
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    with Image.open(io.BytesIO(to_jpeg(buf.getvalue()))) as out:
        assert out.size == (20, 40)


def test_malformed_chat_replies_escalate_instead_of_crashing():
    from doppelgram.conversation import classify_outgoing, parse_reply
    assert parse_reply({"action": "reply", "messages": 42})[0] == "escalate"
    assert parse_reply({"action": "reply", "messages": [1, 2]})[0] == "escalate"
    assert parse_reply({"action": "dance"})[0] == "escalate"
    assert parse_reply([1])[0] == "escalate"
    assert parse_reply({"action": "reply", "messages": "привет"})[:2] == ("reply", ["привет"])
    assert classify_outgoing("приеду завтра в восемь") == "meeting"
    assert classify_outgoing("я не бот, честно") == "ai_question"
    assert classify_outgoing("как прошёл день?") is None


def test_a_menu_number_for_a_denied_item_is_refused():
    from doppelgram import brain
    from doppelgram.models import BotMessage
    m = BotMessage(id=1, text="1. Смотреть анкеты\n2. Премиум")
    pick = lambda n: brain.validate_decision({"action": "send", "text": n}, m, max_wait_s=60,  # noqa: E731
                                             max_send_chars=200, deny=("Премиум",)).action
    assert pick("1") == "send" and pick("2") == "ask_owner"


def test_the_removed_owner_age_option_is_ignored_in_old_configs(tmp_path):
    from doppelgram.config import load_config
    (tmp_path / "config.toml").write_text('[conversations]\nowner_age = 17\nmode = "auto"\n', encoding="utf-8")
    conv = load_config(tmp_path / "config.toml", require_llm=False, require_telegram=False).conversations
    assert conv.mode == "auto" and not hasattr(conv, "owner_age")
