"""What people send besides text (stickers, photos, voice) and people who write first without a match."""

from __future__ import annotations

import io
from dataclasses import replace
from datetime import UTC

import pytest
from test_conversation import ChatMessage, ConversationManager, PersonRef, classify_incoming, make, match

from doppelgram.llm import LLMResult
from doppelgram.speech import SpeechError, Transcript


def png() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 100, 50)).save(buf, format="PNG")
    return buf.getvalue()


class FakeTranscriber:
    def __init__(self, text="давай встретимся в субботу", fail=False):
        self.text, self.fail, self.calls = text, fail, 0

    async def transcribe(self, raw, failures=None):
        self.calls += 1
        if self.fail:
            raise SpeechError("no credit")
        return Transcript(self.text, 3.0, 0.0002)


def prompt_of(m) -> tuple[str, list]:
    content = m.llm.calls[-1][1]["content"]
    if isinstance(content, str):
        return content, []
    return content[0]["text"], [p for p in content if p.get("type") == "image_url"]


async def answered(m, people, clock, user=555):
    m.on_message(user)
    clock.t += 11
    await m.tick()


# ------------------------------------------------------------------ media
async def test_a_sticker_is_read_as_its_emoji(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve", write_first=False)
    await match(m)
    people.incoming(555, "", "sticker", emoji="😂")
    await answered(m, people, clock)
    text, images = prompt_of(m)
    assert "[стикер 😂]" in text and not images and "Черновик" in inbox[-1]
    assert people.downloads == []  # a sticker is never downloaded


async def test_a_photo_is_shown_to_the_model(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve", write_first=False)
    await match(m)
    people.incoming(555, "это я в горах", "photo", blob=png())
    await answered(m, people, clock)
    text, images = prompt_of(m)
    assert "[фото 1] это я в горах" in text and len(images) == 1 and "## Pictures" in text
    assert "Черновик" in inbox[-1]


async def test_a_gif_and_a_video_show_a_preview_frame(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve", write_first=False)
    await match(m)
    people.incoming(555, "", "gif", blob=png())
    people.incoming(555, "", "video", duration=12, blob=png())
    await answered(m, people, clock)
    text, images = prompt_of(m)
    assert "[гифка" in text and "[видео" in text and len(images) == 2
    assert all(thumb for _, thumb in people.downloads)


async def test_photos_can_be_switched_off(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve", write_first=False, see_photos=False)
    await match(m)
    people.incoming(555, "", "photo", blob=png())
    await answered(m, people, clock)
    assert "медиа" in inbox[-1] and m.llm.calls == []


async def test_a_voice_message_is_transcribed_and_checked_like_text(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False)
    m.transcriber = FakeTranscriber("давай встретимся в субботу")
    await match(m)
    people.incoming(555, "", "voice", duration=4, blob=b"ogg")
    m.llm.reply = {"action": "reply", "messages": ["давай, в субботу на кг?"]}
    await answered(m, people, clock)
    assert "речь о встрече" in inbox[-1] and "[голосовое] давай встретимся в субботу" in inbox[-1]
    assert people.sent_texts(555) == []


async def test_an_ordinary_voice_message_gets_an_answer(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve", write_first=False)
    m.transcriber = FakeTranscriber("привет, как прошёл день?")
    await match(m)
    people.incoming(555, "", "voice", duration=4, blob=b"ogg")
    await answered(m, people, clock)
    text, _ = prompt_of(m)
    assert "[голосовое] привет, как прошёл день?" in text and "Черновик" in inbox[-1]
    m.on_message(555)  # read again later: the transcript comes from memory
    await m.tick()
    assert m.transcriber.calls == 1


@pytest.mark.parametrize("transcriber, duration", [(None, 4), (FakeTranscriber(fail=True), 4),
                                                   (FakeTranscriber(), 999)])
async def test_a_voice_message_the_agent_cannot_hear_goes_to_the_owner(tmp_path, transcriber, duration):
    m, people, inbox, clock = make(tmp_path, mode="approve", write_first=False)
    m.transcriber = transcriber
    await match(m)
    people.incoming(555, "", "voice", duration=duration, blob=b"ogg")
    await answered(m, people, clock)
    assert "медиа" in inbox[-1] and m.llm.calls == []


async def test_a_video_note_is_heard_and_seen(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve", write_first=False)
    m.transcriber = FakeTranscriber("смотри какой закат")
    await match(m)
    people.incoming(555, "", "video_note", duration=8, blob=png())
    await answered(m, people, clock)
    text, images = prompt_of(m)
    assert "[кружок 1] смотри какой закат" in text and len(images) == 1


async def test_files_and_unknown_media_still_go_to_the_owner(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve", write_first=False)
    await match(m)
    people.incoming(555, "", "other")
    await answered(m, people, clock)
    assert "медиа" in inbox[-1] and m.llm.calls == []


async def test_transcripts_and_pictures_are_never_stored(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve", write_first=False)
    m.transcriber = FakeTranscriber("секретная фраза про кота")
    await match(m)
    people.incoming(555, "", "voice", duration=4, blob=b"ogg")
    await answered(m, people, clock)
    dump = "\n".join(m.store.db.iterdump())
    assert "секретная фраза" not in dump and "base64" not in dump


# ------------------------------------------------------------------ strangers
async def stranger_writes(m, people, clock, text="привет) я тебя в бибинто видела", user=777, name="Лена"):
    people.incoming(user, text)
    await m.on_stranger(PersonRef(user, "lena_example", name))
    clock.t += 11
    await m.tick()


async def test_a_stranger_goes_to_the_owner_by_default(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve")
    await stranger_writes(m, people, clock)
    assert "Вам впервые написал(а) Лена (@lena_example)" in inbox[-1] and "«веди»" in inbox[-1]
    assert m.llm.calls == []  # in ask mode nothing is sent to the model before the owner decides
    await m.answer_card(9000 + len(inbox), "веди")
    clock.t += 1
    await m.tick()
    assert "Черновик" in inbox[-1] and m.store.person(777).state == "active"


async def test_the_owner_can_ignore_a_stranger(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve")
    await stranger_writes(m, people, clock)
    await m.answer_card(9000 + len(inbox), "игнор")
    people.incoming(777, "ау")
    m.on_message(777)
    clock.t += 11
    await m.tick()
    assert m.store.person(777).state == "blocked" and m.llm.calls == []


async def test_the_owner_can_answer_a_stranger_in_their_own_words(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve")
    await stranger_writes(m, people, clock)
    await m.answer_card(9000 + len(inbox), "привет) да, было дело")
    assert people.sent_texts(777) == ["привет) да, было дело"] and m.store.person(777).state == "active"


def classifier(kind):
    def reply(messages):
        if "unknown Telegram user wrote first" in messages[0]["content"]:
            return {"kind": kind, "reason": "тест"}
        return {"action": "reply", "messages": ["привет) да, видел тебя там"]}
    return reply


async def test_in_lead_mode_an_acquaintance_is_taken_up(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", strangers="lead")
    m.llm.reply = classifier("dating")
    await stranger_writes(m, people, clock)
    assert "похоже на знакомство" in inbox[-1]
    for _ in range(3):
        clock.t += 31
        await m.tick()
    assert people.sent_texts(777) == ["привет) да, видел тебя там"]


@pytest.mark.parametrize("kind", ["known", "business", "unclear"])
async def test_in_lead_mode_anything_else_goes_to_the_owner(tmp_path, kind):
    m, people, inbox, clock = make(tmp_path, mode="auto", strangers="lead")
    m.llm.reply = classifier(kind)
    await stranger_writes(m, people, clock, text="привет, это Саша с курса, скинешь конспект?")
    assert "Вам впервые написал(а)" in inbox[-1] and people.sent_texts(777) == []


async def test_spam_and_existing_chats_are_dropped_silently(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", strangers="lead")
    await stranger_writes(m, people, clock, text="заработок от 5000 в день, пиши", user=801)
    people.owner_writes(802, "привет, как ты")  # the owner already talked with this person
    await stranger_writes(m, people, clock, text="привет!", user=802)
    assert m.store.person(801).state == "blocked" and m.store.person(802).state == "blocked"
    assert inbox == [] and m.llm.calls == []


async def test_strangers_can_be_ignored_entirely(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", strangers="ignore")
    await stranger_writes(m, people, clock)
    assert m.store.person(777) is None and inbox == []


async def test_a_stranger_who_writes_several_messages_is_triaged_once(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve")
    people.incoming(777, "привет")
    await m.on_stranger(PersonRef(777, None, "Лена"))
    clock.t += 5
    people.incoming(777, "я из бибинто)")
    m.on_message(777)
    await m.on_stranger(PersonRef(777, None, "Лена"))  # a second event for the same person
    clock.t += 11
    await m.tick()
    assert sum("Вам впервые написал(а)" in t for t in inbox) == 1 and "я из бибинто" in inbox[-1]


async def test_strangers_mode_is_controlled_from_telegram(tmp_path):
    from doppelgram.control import parse_owner
    m, people, inbox, clock = make(tmp_path)
    assert parse_owner("/chat незнакомые вести").arg == "strangers:lead"
    await m.command("chat_mode", "strangers:ignore")
    assert m.strangers == "ignore"


def test_a_bio_is_kept_without_digits(tmp_path):
    import asyncio

    async def run():
        m, people, inbox, clock = make(tmp_path, mode="approve")
        people.bio = "Лена, 21, люблю джаз"
        await stranger_writes(m, people, clock)
        return m.store.person(777).card
    assert asyncio.run(run()) == "Лена, люблю джаз"


# ------------------------------------------------------------------ speech and telegram plumbing
def opus_voice(seconds=1.0) -> bytes:
    import array
    import math

    import av
    buf = io.BytesIO()
    out = av.open(buf, "w", format="ogg")
    stream = out.add_stream("libopus", rate=48000)
    stream.layout = "mono"
    total, step = int(48000 * seconds), 960
    for start in range(0, total, step):
        pcm = array.array("h", (int(8000 * math.sin(2 * math.pi * 440 * (start + i) / 48000)) for i in range(step)))
        frame = av.AudioFrame(format="s16", layout="mono", samples=step)
        frame.planes[0].update(pcm.tobytes())
        frame.sample_rate = 48000
        frame.pts = start
        for packet in stream.encode(frame):
            out.mux(packet)
    for packet in stream.encode(None):
        out.mux(packet)
    out.close()
    return buf.getvalue()


def test_voice_is_decoded_to_wav_and_cut():
    import wave

    from doppelgram.speech import to_wav
    raw = opus_voice(2.0)
    wav, seconds = to_wav(raw, 1.0)
    with wave.open(io.BytesIO(wav)) as w:
        assert (w.getframerate(), w.getnchannels()) == (16000, 1)
    assert 0.9 <= seconds <= 1.0
    with pytest.raises(SpeechError):
        to_wav(b"not audio", 10)


async def test_the_transcriber_sends_wav_as_input_audio_without_reasoning():
    from doppelgram.config import LLMConfig
    from doppelgram.speech import Transcriber

    class Client:
        async def complete_json(self, messages, failures=None):
            self.messages = messages
            return LLMResult(data={"text": " привет "}, model="m", prompt_tokens=1, completion_tokens=1,
                             cost_usd=0.0001)

    cfg = LLMConfig(base_url="https://openrouter.ai/api/v1", api_key="k", model="main")
    t = Transcriber(cfg, "google/gemini-3.8-flash")
    assert t.client.cfg.extra_body["reasoning"] == {"enabled": False} and t.client.cfg.model.startswith("google/")
    await t.client.aclose()
    t.client = Client()
    result = await t.transcribe(opus_voice(1.0))
    audio = t.client.messages[0]["content"][1]
    assert result.text == "привет" and audio["type"] == "input_audio" and audio["input_audio"]["format"] == "wav"


def test_telegram_media_kinds_and_stranger_filter():
    from datetime import datetime

    from telethon.tl import types

    from doppelgram.telegram import chat_media, is_stranger_candidate

    def doc(*attrs, mime="application/octet-stream"):
        return types.Message(id=1, peer_id=types.PeerUser(1), date=datetime(2026, 1, 1), message="", media=(
            types.MessageMediaDocument(document=types.Document(
                id=1, access_hash=1, file_reference=b"", date=datetime(2026, 1, 1), mime_type=mime, size=10,
                dc_id=2, attributes=list(attrs)))))
    assert chat_media(doc(types.DocumentAttributeSticker(alt="😂", stickerset=types.InputStickerSetEmpty())))[:2] == \
        ("sticker", "😂")
    assert chat_media(doc(types.DocumentAttributeAudio(duration=7, voice=True)))[::2] == ("voice", 7.0)
    assert chat_media(doc(types.DocumentAttributeVideo(duration=5, w=1, h=1, round_message=True)))[0] == "video_note"
    assert chat_media(doc(types.DocumentAttributeVideo(duration=5, w=1, h=1),
                          types.DocumentAttributeAnimated()))[0] == "gif"
    assert chat_media(doc(mime="image/png"))[0] == "image"
    user = types.User(id=5, first_name="Лена")
    assert is_stranger_candidate(user)
    for flag in ("bot", "contact", "mutual_contact", "deleted", "support", "scam", "fake"):
        assert not is_stranger_candidate(types.User(id=5, **{flag: True}))
    assert not is_stranger_candidate(types.Channel(id=1, title="x", photo=types.ChatPhotoEmpty(),
                                                   date=datetime(2026, 1, 1)))


def test_chat_message_defaults_are_backward_compatible():
    m = ChatMessage(1, False, "привет", 0.0)
    assert (m.media_kind, m.emoji, m.duration) == (None, "", 0.0)
    assert ConversationManager.MAX_IMAGES == 3


async def test_a_strangers_burst_is_waited_out_before_triage(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve")
    people.incoming(777, "привет")
    await m.on_stranger(PersonRef(777, None, "Лена"))
    clock.t += 8
    people.incoming(777, "ты тут?")
    m.on_message(777)  # settle again from the newest message
    clock.t += 4
    await m.tick()
    assert inbox == []
    clock.t += 7
    await m.tick()
    assert "ты тут?" in inbox[-1]


async def test_the_transcript_memory_is_bounded(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve", write_first=False)
    m.transcriber = FakeTranscriber("привет")
    await match(m)
    for i in range(510):
        m._transcripts[-i - 1] = "x"
    people.incoming(555, "", "voice", duration=2, blob=b"ogg")
    await answered(m, people, clock)
    assert len(m._transcripts) == 500


# ------------------------------------------------------------------ review regressions


async def test_the_owner_taking_a_stranger_during_triage_wins(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", strangers="lead")
    m.llm.reply = classifier("dating")
    real_profile = people.profile

    async def owner_takes_over(user_id):
        await m.command("self", user_id)
        return await real_profile(user_id)
    people.profile = owner_takes_over
    await stranger_writes(m, people, clock)
    for _ in range(3):
        clock.t += 31
        await m.tick()
    assert m.store.person(777).state == "owner" and people.sent_texts(777) == []


async def test_a_chat_the_owner_wrote_in_long_ago_is_never_a_stranger(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", strangers="lead", history_messages=30)
    m.llm.reply = classifier("dating")
    people.owner_writes(777, "давно не виделись")
    for i in range(60):
        people.incoming(777, f"сообщение {i}")
    await m.on_stranger(PersonRef(777, None, "Знакомый"))
    clock.t += 11
    await m.tick()
    assert m.store.person(777).state == "blocked" and m.llm.calls == [] and inbox == []


@pytest.mark.parametrize("text", ["хочу сосать", "скинь нюдс", "давай секстинг"])
async def test_graphic_sexting_goes_to_the_owner_even_with_intimacy(tmp_path, text):
    from doppelgram.conversation import classify_outgoing
    assert classify_outgoing(text, intimacy=True) == "explicit"
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, intimacy=True)
    await match(m)
    people.incoming(555, "ты такой милый)")
    m.llm.reply = {"action": "reply", "messages": [text]}
    await answered(m, people, clock)
    clock.t += 31
    await m.tick()
    assert people.sent_texts(555) == [] and "интимная тема" in inbox[-1]


async def test_transcription_respects_the_daily_budget(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve", write_first=False, daily_budget_usd=0.1)
    m.transcriber = FakeTranscriber("привет")
    await match(m)
    m.store.log_spend(clock(), 0.2)
    people.incoming(555, "", "voice", duration=3, blob=b"ogg")
    await answered(m, people, clock)
    assert m.transcriber.calls == 0


async def test_a_video_note_with_neither_speech_nor_picture_goes_to_the_owner(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False)
    m.transcriber = FakeTranscriber(fail=True)
    await match(m)
    people.incoming(555, "", "video_note", duration=5, blob=None)  # no preview available
    await answered(m, people, clock)
    assert "медиа" in inbox[-1] and m.llm.calls == []


@pytest.mark.parametrize("text", ["ок, оплачу такси", "заплачу за двоих", "скину 500"])
def test_any_promise_to_pay_goes_to_the_owner(text):
    from doppelgram.conversation import classify_outgoing
    assert classify_outgoing(text) == "money"


@pytest.mark.parametrize("error", ["flood", "restricted"])
async def test_telegram_limits_while_reading_a_bio_are_not_swallowed(tmp_path, error):
    from doppelgram.transport import AccountRestricted, RateLimited
    m, people, inbox, clock = make(tmp_path, mode="auto", strangers="lead")
    m.llm.reply = classifier("dating")

    async def failing(user_id):
        raise RateLimited(60) if error == "flood" else AccountRestricted("PeerFloodError")
    people.profile = failing
    await stranger_writes(m, people, clock)
    assert m.store.person(777).state == "new" and m.llm.calls == []
    if error == "flood":
        assert m.store.next_due() >= clock.t + 60
    else:
        assert m.stopped


async def test_a_paid_answer_without_a_transcript_counts_against_the_budget():
    from doppelgram.config import LLMConfig
    from doppelgram.speech import Transcriber

    class Client:
        async def complete_json(self, messages, failures=None):
            return LLMResult(data={"text": None}, model="m", prompt_tokens=1, completion_tokens=1, cost_usd=0.5)

    t = Transcriber(LLMConfig(base_url="https://openrouter.ai/api/v1", api_key="k", model="x"), "m")
    await t.client.aclose()
    t.client = Client()
    failures: list = []
    with pytest.raises(SpeechError):
        await t.transcribe(opus_voice(0.5), failures=failures)
    assert [f.cost_usd for f in failures] == [0.5]


# ------------------------------------------------------------------ second review regressions
async def test_wrote_before_scans_the_history_for_the_accounts_own_messages():
    from types import SimpleNamespace

    from doppelgram.telegram import PeopleChannel

    class Client:
        def __init__(self, outs):
            self.outs = outs

        async def get_messages(self, peer, limit):
            self.limit = limit
            return [SimpleNamespace(out=o) for o in self.outs]

    for outs, expected in (([False] * 999 + [True], True), ([False, False], False), ([], False)):
        account = SimpleNamespace(client=Client(outs))
        assert await PeopleChannel(account).wrote_before(777) is expected
    assert account.client.limit == PeopleChannel.WROTE_BEFORE_SCAN


@pytest.mark.parametrize("text", ["я закончила универ", "смотрю канал про путешествия", "без страха",
                                  "работаю аналитиком", "член клуба любителей кино"])
def test_ordinary_words_are_not_graphic(text):
    from doppelgram.conversation import classify_outgoing
    assert classify_incoming(text, None, intimacy=True) is None
    assert classify_outgoing(text, intimacy=True) is None


def test_graphic_words_are_caught_with_intimacy_off_too():
    from doppelgram.conversation import classify_outgoing
    assert classify_outgoing("хочу сосать", intimacy=False) == "explicit"
    assert classify_incoming("хочу сосать", None, intimacy=False) == "explicit"


async def test_an_unrelated_command_does_not_abort_triage(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve")
    real_profile = people.profile

    async def busy_owner(user_id):
        await m.command("chat_mode", "auto")  # bumps the epoch, unrelated to this person
        return await real_profile(user_id)
    people.profile = busy_owner
    await stranger_writes(m, people, clock)
    assert "Вам впервые написал(а)" in inbox[-1]


async def test_lead_during_the_existing_chat_check_wins(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve")
    people.incoming(777, "привет")
    await m.on_stranger(PersonRef(777, None, "Лена"))

    async def owner_steps_in(user_id):
        people.owner_writes(777, "о, привет")
        await m.command("lead", 777)
        return True
    people.wrote_before = owner_steps_in
    clock.t += 11
    await m.tick()
    assert m.store.person(777).state == "active"


@pytest.mark.parametrize("text, money", [("не заплачу ни копейки", False), ("билет стоит 300 рублей", False),
                                         ("ладно, заплачу", True), ("скину 500 на карту", True)])
def test_refusals_and_prices_are_not_promises_to_pay(text, money):
    from doppelgram.conversation import classify_outgoing
    assert (classify_outgoing(text) == "money") is money


async def test_a_match_with_an_old_chat_stays_with_the_owner(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=True)
    people.owner_writes(555, "привет, давно не общались")
    for i in range(40):
        people.incoming(555, f"сообщение {i}")
    await match(m)
    await m.tick()
    assert m.store.person(555).state == "owner" and "Вы уже переписываетесь" in inbox[-1] and m.llm.calls == []


@pytest.mark.parametrize("text, kind", [("возьму твой член в рот", "explicit"), ("член клуба", None),
                                        ("я пошла спать", None), ("это пошлость", "explicit")])
def test_explicit_words_in_context(text, kind):
    from doppelgram.conversation import classify_outgoing
    assert classify_outgoing(text, intimacy=False) == kind


@pytest.mark.parametrize("text, money", [("скину тебе 500 на карту", True), ("не заплачу меньше тысячи", True)])
def test_payment_promises_with_words_between(text, money):
    from doppelgram.conversation import classify_outgoing
    assert (classify_outgoing(text) == "money") is money


async def test_an_unchecked_match_chat_is_checked_before_the_first_answer(tmp_path):
    from doppelgram.transport import RateLimited
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, history_messages=5)
    people.owner_writes(555, "привет, давно не общались")
    for i in range(10):
        people.incoming(555, f"сообщение {i}")
    real = people.wrote_before
    calls = {"n": 0}

    async def flaky(user_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimited(30)
        return await real(user_id)
    people.wrote_before = flaky
    await match(m)
    m.on_message(555)
    clock.t += 11
    await m.tick()
    assert m.store.person(555).state == "owner" and people.sent_texts(555) == ["привет, давно не общались"]


async def test_a_takeover_while_a_meeting_draft_is_written_wins(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False)
    await match(m)
    people.incoming(555, "давай встретимся в субботу?")

    def draft(messages):
        m.store.update(555, state="owner")  # /self while the model writes the draft
        return {"action": "reply", "messages": ["давай, в субботу на кг"]}
    m.llm.reply = draft
    await answered(m, people, clock)
    assert m.store.person(555).state == "owner" and not any("речь о встрече" in t for t in inbox)


async def test_a_reply_stopped_at_send_time_is_replaced_by_a_new_one(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, intimacy=True)
    await match(m)
    people.incoming(555, "привет)")
    m.llm.reply = {"action": "reply", "messages": ["хочу тебя, секс вечером?"]}
    await answered(m, people, clock)  # queued
    real_typing = people.typing

    async def typing_then_off(user_id, seconds):
        people.typing = real_typing  # once
        await m.command("chat_mode", "intimacy:off")
        await real_typing(user_id, seconds)
    people.typing = typing_then_off
    m.llm.reply = {"action": "reply", "messages": ["как прошёл день?"]}
    for _ in range(4):
        clock.t += 31
        await m.tick()
    people.typing = real_typing
    assert people.sent_texts(555) == ["как прошёл день?"]


async def test_lead_after_a_failed_earlier_chat_check_is_respected(tmp_path):
    from doppelgram.transport import RateLimited
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, history_messages=5)
    people.owner_writes(555, "привет, давно не общались")
    for i in range(10):
        people.incoming(555, f"сообщение {i}")
    real = people.wrote_before
    calls = {"n": 0}

    async def flaky(user_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimited(30)
        return await real(user_id)
    people.wrote_before = flaky
    await match(m)
    await m.command("lead", 555)
    m.llm.reply = {"action": "reply", "messages": ["ахах да, рад слышать"]}
    for _ in range(3):
        clock.t += 31
        await m.tick()
    assert m.store.person(555).state == "active" and "ахах да, рад слышать" in people.sent_texts(555)


async def test_lead_while_the_earlier_chat_check_is_running_wins(tmp_path):
    from doppelgram.transport import RateLimited
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, history_messages=5)
    people.owner_writes(555, "привет, давно не общались")
    for i in range(10):
        people.incoming(555, f"сообщение {i}")
    real = people.wrote_before
    calls = {"n": 0}

    async def check(user_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimited(30)
        if calls["n"] == 2:
            await m.command("lead", 555)  # the owner hands the chat over while the check runs
        return await real(user_id)
    people.wrote_before = check
    await match(m)
    m.on_message(555)
    m.llm.reply = {"action": "reply", "messages": ["ахах да, рад слышать"]}
    for _ in range(4):
        clock.t += 31
        await m.tick()
    assert m.store.person(555).state == "active" and "ахах да, рад слышать" in people.sent_texts(555)


async def test_nothing_is_sent_to_a_match_until_the_earlier_chat_check_succeeds(tmp_path):
    from doppelgram.transport import RateLimited
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, history_messages=5)
    people.owner_writes(555, "привет, давно не общались")
    for i in range(10):
        people.incoming(555, f"сообщение {i}")
    real = people.wrote_before
    started = {"n": 0}

    async def slow_then_fail(user_id):
        started["n"] += 1
        if started["n"] == 1:  # while the match's check waits, the person writes and the agent prepares
            people.incoming(555, "ау?")
            m.on_message(555)
            raise RateLimited(30)
        return await real(user_id)
    people.wrote_before = slow_then_fail
    m.llm.reply = {"action": "reply", "messages": ["привет)"]}
    await match(m)
    for _ in range(4):
        clock.t += 61
        await m.tick()
    assert people.sent_texts(555) == ["привет, давно не общались"] and m.store.person(555).state == "owner"


async def test_the_rest_of_a_series_survives_a_command_about_another_chat(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False)
    await match(m)
    people.incoming(555, "привет)")
    m.llm.reply = {"action": "reply", "messages": ["привет)", "как день прошёл?"]}
    await answered(m, people, clock)
    real_send = people.send

    async def send_then_command(user_id, text, reply_to=None):
        people.send = real_send
        msg_id = await real_send(user_id, text, reply_to=reply_to)
        await m.command("self", 999)  # about some other chat: bumps the shared epoch
        return msg_id
    people.send = send_then_command
    for _ in range(3):
        clock.t += 31
        await m.tick()
    assert people.sent_texts(555) == ["привет)", "как день прошёл?"]


async def test_an_approved_series_survives_a_command_about_another_chat(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve", write_first=False)
    await match(m)
    people.incoming(555, "привет)")
    m.llm.reply = {"action": "reply", "messages": ["привет)", "давай в субботу на кг?"]}
    await answered(m, people, clock)
    card = 9000 + len(inbox)
    real_send = people.send

    async def send_then_command(user_id, text, reply_to=None):
        people.send = real_send
        msg_id = await real_send(user_id, text, reply_to=reply_to)
        await m.command("self", 999)
        return msg_id
    people.send = send_then_command
    await m.answer_card(card, "ок")
    for _ in range(2):
        clock.t += 31
        await m.tick()
    assert people.sent_texts(555) == ["привет)", "давай в субботу на кг?"]


async def test_a_restart_while_typing_resumes_the_queued_reply_without_repeats(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False)
    await match(m)
    people.incoming(555, "привет)")
    m.llm.reply = {"action": "reply", "messages": ["привет)", "как день?"]}
    await answered(m, people, clock)  # queued
    real_send = people.send
    sends = {"n": 0}

    class Crash(Exception):
        pass

    async def crash_on_second(user_id, text, reply_to=None):
        sends["n"] += 1
        if sends["n"] == 2:
            raise Crash()  # the process dies while typing the second message
        return await real_send(user_id, text, reply_to=reply_to)
    people.send = crash_on_second
    clock.t += 31
    with pytest.raises(Crash):
        await m._tick()
    people.send = real_send
    restarted = ConversationManager(m.cfg, m.llm, people, m.store, m._notify, clock=clock, sleep=clock.sleep)
    clock.t += 1
    await restarted.tick()
    assert people.sent_texts(555) == ["привет)", "как день?"]


async def owner_ok_card(tmp_path, messages):
    m, people, inbox, clock = make(tmp_path, mode="approve", write_first=False, intimacy=True)
    await match(m)
    people.incoming(555, "привет)")
    m.llm.reply = {"action": "reply", "messages": messages}
    await answered(m, people, clock)
    return m, people, inbox, clock, 9000 + len(inbox)


async def test_an_approved_series_survives_a_setting_change_in_the_middle(tmp_path):
    m, people, inbox, clock, card = await owner_ok_card(tmp_path, ["привет)", "как день прошёл?"])
    real_send = people.send

    async def send_then_setting(user_id, text, reply_to=None):
        people.send = real_send
        msg_id = await real_send(user_id, text, reply_to=reply_to)
        await m.command("chat_mode", "first:off")
        return msg_id
    people.send = send_then_setting
    await m.answer_card(card, "ок")
    assert "остальное отправлю следом" in inbox[-1]
    clock.t += 1
    await m.tick()
    assert people.sent_texts(555) == ["привет)", "как день прошёл?"]


async def test_an_approved_answer_is_kept_if_the_process_dies_before_sending(tmp_path):
    m, people, inbox, clock, card = await owner_ok_card(tmp_path, ["привет)", "как день?"])

    class Crash(Exception):
        pass

    async def die(user_id, seconds):
        raise Crash()
    real_typing, people.typing = people.typing, die
    with pytest.raises(Crash):
        await m.answer_card(card, "ок")
    people.typing = real_typing
    restarted = ConversationManager(m.cfg, m.llm, people, m.store, m._notify, clock=clock, sleep=clock.sleep)
    clock.t += 61
    await restarted.tick()
    assert people.sent_texts(555) == ["привет)", "как день?"]


def test_a_send_is_recorded_and_dequeued_in_one_transaction(tmp_path):
    from test_conversation import ConversationStore
    store = ConversationStore(":memory:")
    store.add_person(PersonRef(1, None, "a"), "b", "", 0.0)
    store.set_outbox(1, 5, ["x", "y"], 0.0)

    def boom(user_id, remaining, expected=None):
        raise RuntimeError("crash")
    store.trim_outbox = boom
    with pytest.raises(RuntimeError):
        store.commit_send(1, 77, 1.0, 5, ["y"])
    assert store.sent_ids(1) == set() and store.outbox_is(1, 5, ["x", "y"])


async def test_the_owners_own_text_stays_their_text_after_a_setting_change(tmp_path):
    m, people, inbox, clock, card = await owner_ok_card(tmp_path, ["черновик агента"])
    real_typing = people.typing

    async def typing_then_setting(user_id, seconds):
        people.typing = real_typing
        await m.command("chat_mode", "first:off")
        people.incoming(555, "ау?")  # the person writes before the retry
        await real_typing(user_id, seconds)
    people.typing = typing_then_setting
    await m.answer_card(card, "мой собственный ответ")
    for _ in range(2):
        clock.t += 31
        await m.tick()
    assert "мой собственный ответ" in people.sent_texts(555)


async def test_a_restored_card_with_the_owners_text_still_sends_it_as_theirs(tmp_path):
    from doppelgram.transport import TransportError
    m, people, inbox, clock, card = await owner_ok_card(tmp_path, ["черновик"])
    real_typing = people.typing

    async def failing(uid, seconds):
        people.typing = real_typing
        raise TransportError("network")
    people.typing = failing
    await m.answer_card(card, "мой текст")  # fails before sending: the card comes back
    assert "Ответьте на карточку ещё раз" in inbox[-1]
    people.incoming(555, "ау?")  # the person writes meanwhile
    await m.answer_card(card, "ок")
    assert people.sent_texts(555) == ["мой текст"]


async def test_a_new_message_does_not_replace_the_owners_unsent_text(tmp_path):
    from doppelgram.transport import TransportError
    m, people, inbox, clock, card = await owner_ok_card(tmp_path, ["черновик"])
    real_history = people.history

    async def failing(uid, limit):
        people.history = real_history
        raise TransportError("network")
    people.history = failing
    await m.answer_card(card, "мой текст")
    people.incoming(555, "ау?")
    m.on_message(555)
    clock.t += 11
    await m.tick()  # must not draft over the owner's text
    await m.answer_card(card, "ок")
    assert people.sent_texts(555) == ["мой текст"]


async def test_the_owners_queued_text_is_not_replaced_by_a_new_draft(tmp_path):
    from doppelgram.transport import TransportError
    m, people, inbox, clock, card = await owner_ok_card(tmp_path, ["черновик"])
    real_typing, real_history = people.typing, people.history

    async def typing_then_setting(user_id, seconds):
        people.typing = real_typing
        await m.command("chat_mode", "first:off")  # the rest goes to the queue as the owner's text
        await real_typing(user_id, seconds)
    people.typing = typing_then_setting
    await m.answer_card(card, "мой текст")
    people.incoming(555, "ау?")
    m.on_message(555)

    async def failing_once(uid, limit):
        people.history = real_history
        raise TransportError("network")
    people.history = failing_once
    for _ in range(4):
        clock.t += 130
        await m.tick()
    assert "мой текст" in people.sent_texts(555)


async def test_self_during_lead_wins(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False)
    await match(m)
    await m.command("self", 555)
    real_history = people.history

    async def history_then_self(uid, limit):
        people.history = real_history
        await m.command("self", 555)
        return await real_history(uid, limit)
    people.history = history_then_self
    await m.command("lead", 555)
    assert m.store.person(555).state == "owner"


async def test_lead_on_an_escalation_survives_a_network_error(tmp_path):
    from doppelgram.transport import TransportError
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False)
    await match(m)
    people.incoming(555, "ты бот?")
    await answered(m, people, clock)
    card = 9000 + len(inbox)
    real_history = people.history

    async def failing(uid, limit):
        people.history = real_history
        raise TransportError("network")
    people.history = failing
    await m.answer_card(card, "веди")
    assert "ещё раз" in inbox[-1]
    await m.answer_card(card, "веди")
    assert m.store.person(555).state == "active"


async def test_lead_on_a_waiting_stranger_lets_the_agent_answer(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto")
    await stranger_writes(m, people, clock)
    await m.command("lead", 777)
    m.llm.reply = {"action": "reply", "messages": ["привет) да, было дело"]}
    for _ in range(3):
        clock.t += 31
        await m.tick()
    people.incoming(777, "как дела?")
    m.on_message(777)
    m.llm.reply = {"action": "reply", "messages": ["норм, ты как?"]}
    for _ in range(3):
        clock.t += 31
        await m.tick()
    assert people.sent_texts(777) == ["привет) да, было дело", "норм, ты как?"]


async def test_a_message_sent_just_before_a_crash_is_not_sent_again(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False)
    await match(m)
    people.incoming(555, "привет)")
    m.llm.reply = {"action": "reply", "messages": ["привет)", "как день?"]}
    await answered(m, people, clock)

    class Crash(Exception):
        pass
    real_commit = m.store.commit_send

    def crash(*a, **k):
        m.store.commit_send = real_commit
        raise Crash()  # Telegram took the message, the process died before recording it
    m.store.commit_send = crash
    clock.t += 31
    with pytest.raises(Crash):
        await m._tick()
    restarted = ConversationManager(m.cfg, m.llm, people, m.store, m._notify, clock=clock, sleep=clock.sleep)
    clock.t += 1
    await restarted.tick()
    assert people.sent_texts(555) == ["привет)", "как день?"]


async def test_an_old_owner_message_with_the_same_words_is_not_taken_for_a_lost_send(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve", write_first=False)
    await match(m)
    people.owner_writes(555, "хорошо")  # the owner wrote this themselves earlier
    await m.command("lead", 555)
    people.incoming(555, "ну так что, в субботу?")
    m.llm.reply = {"action": "reply", "messages": ["черновик"]}
    await answered(m, people, clock)
    await m.answer_card(9000 + len(inbox), "хорошо")
    assert people.sent_texts(555) == ["хорошо", "хорошо"]


def test_status_shows_the_total_spend(tmp_path):
    import random as _r

    from fakes import Clock, FakeRatingBot, ScriptedLLM, make_config, rate_or_wait

    from doppelgram.agent import Agent
    from doppelgram.storage import Store
    clock = Clock()
    store = Store(":memory:")
    for day in range(3):
        store.log_llm(clock() - 86400 * day, "m", 1, 1, 0.5, ok=True)
    agent = Agent(make_config(tmp_path), FakeRatingBot(), ScriptedLLM(rate_or_wait), store, clock=clock,
                  sleep=clock.sleep, rng=_r.Random(0))
    text = str(agent.status_text())
    assert "$1.50</b> всего" in text and "сегодня $0.50" in text


async def test_a_lost_send_is_recognised_even_if_the_owner_wrote_more_after_it(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False)
    await match(m)
    people.incoming(555, "давай встретимся?")
    m.llm.reply = {"action": "reply", "messages": ["в субботу?"]}
    await answered(m, people, clock)  # an escalation: the owner writes their own text
    card = 9000 + len(inbox)

    class Crash(Exception):
        pass
    real_commit = m.store.commit_send

    def crash(*a, **k):
        m.store.commit_send = real_commit
        raise Crash()
    m.store.commit_send = crash
    with pytest.raises(Crash):
        await m.answer_card(card, "!в субботу в шесть")  # the owner's own words, as they are
    people.owner_writes(555, "у входа в парк")  # written directly while the agent was down
    restarted = ConversationManager(m.cfg, m.llm, people, m.store, m._notify, clock=clock, sleep=clock.sleep)
    clock.t += 61
    await restarted.tick()
    assert people.sent_texts(555).count("в субботу в шесть") == 1


@pytest.mark.parametrize("text", ["когда встречаемся?", "давай встречаться", "встретимся в субботу?"])
def test_meeting_words_in_all_forms(text):
    assert classify_incoming(text, None) == "meeting"


# ------------------------------------------------------------------ pace, read receipts, backlog
def test_the_reply_delay_follows_the_persons_pace_and_is_never_instant(tmp_path):
    import random as _r

    from doppelgram.conversation import _their_pace
    m, people, inbox, clock = make(tmp_path, reply_delay_min_s=60, reply_delay_max_s=5400, long_delay_chance=0)
    m.rng = _r.Random(1)
    quick = [ChatMessage(1, True, "a", 1000.0), ChatMessage(2, False, "b", 1060.0),
             ChatMessage(3, True, "c", 1100.0), ChatMessage(4, False, "d", 1190.0)]
    slow = [ChatMessage(1, True, "a", 100.0), ChatMessage(2, False, "b", 3100.0)]
    assert _their_pace(quick) == 90 and _their_pace(slow) == 3000
    assert all(60 <= m._reply_delay(quick) <= 126 for _ in range(50))
    assert all(1800 <= m._reply_delay(slow) <= 4200 for _ in range(50))
    assert all(60 <= m._reply_delay([]) <= 5400 for _ in range(50))


def test_sometimes_the_agent_takes_a_long_pause(tmp_path):
    import random as _r
    m, people, inbox, clock = make(tmp_path, long_delay_chance=1.0, long_delay_min_s=1800, long_delay_max_s=9000)
    m.rng = _r.Random(2)
    assert all(1800 <= m._reply_delay([]) <= 9000 for _ in range(20))


async def test_messages_are_read_only_when_the_agent_answers(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False)
    await match(m)
    msg_id = people.incoming(555, "привет)")
    m.llm.reply = {"action": "reply", "messages": ["привет, как ты?"]}
    await answered(m, people, clock)
    assert people.read == {}  # queued for later: not read yet, like a busy person
    clock.t += 31
    await m.tick()
    assert people.read == {555: msg_id} and people.sent_texts(555) == ["привет, как ты?"]


async def test_read_receipts_can_be_switched_off(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, read_receipts=False)
    await match(m)
    people.incoming(555, "привет)")
    m.llm.reply = {"action": "reply", "messages": ["привет"]}
    await answered(m, people, clock)
    clock.t += 31
    await m.tick()
    assert people.read == {} and people.sent_texts(555) == ["привет"]


async def test_the_prompt_says_how_long_ago_they_wrote(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve", write_first=False)
    await match(m)
    people.chats[555] = [ChatMessage(10, False, "привет, ты тут?", clock() - 3 * 3600)]
    await answered(m, people, clock)
    text, _ = prompt_of(m)
    system = m.llm.calls[-1][0]["content"]
    assert "Their last message was 3 h ago" in text and "busy person" in system


async def test_unanswered_chats_are_picked_up_at_start(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", strangers="lead")
    m.llm.reply = classifier("dating")
    people.incoming(901, "привет) мы в дайвинчике совпали")
    people.incoming(902, "привет, это Саша с курса")
    people.backlog_refs = [PersonRef(901, "a_example", "Аня"), PersonRef(902, None, "Саша")]
    await m.startup()
    assert m.store.person(901).state == "new" and m.store.person(902).state == "new"
    assert m.store.person(902).due_at > m.store.person(901).due_at  # one after another
    m.llm.reply = lambda msgs: ({"kind": "dating" if "дайвинчик" in msgs[1]["content"] else "known", "reason": "т"}
                                if "unknown Telegram user" in msgs[0]["content"]
                                else {"action": "reply", "messages": ["привет) сорри, пропал"]})
    for _ in range(4):
        clock.t += 200
        await m.tick()
    assert m.store.person(901).state == "active" and m.store.person(902).state == "new"
    assert any("Вам впервые написал(а) Саша" in t for t in inbox)


async def test_switching_chats_on_picks_up_waiting_chats(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve", enabled=False)
    people.incoming(901, "привет)")
    people.backlog_refs = [PersonRef(901, None, "Аня")]
    await m.startup()
    assert m.store.person(901) is None  # chats are off
    await m.command("chat_mode", "on")
    assert m.store.person(901).state == "new"


async def test_chats_the_agent_leads_are_looked_at_again_at_start(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve", write_first=False)
    await match(m)
    people.incoming(555, "ты где пропал?")  # written while the agent was down: no event
    await m.startup()
    clock.t += 1
    await m.tick()
    assert "Черновик" in inbox[-1]


async def test_backlog_keeps_only_unread_chats_people_started_recently():
    import time
    from datetime import datetime
    from types import SimpleNamespace

    from telethon.tl import types

    from doppelgram.telegram import PeopleChannel

    def at(days_ago):
        return datetime.fromtimestamp(time.time() - days_ago * 86400, UTC)

    def dialog(uid, *, out=False, unread=1, days_ago=1, pinned=False, **flags):
        return SimpleNamespace(is_user=True, entity=types.User(id=uid, first_name=f"u{uid}", **flags),
                               message=SimpleNamespace(out=out, date=at(days_ago)), unread_count=unread,
                               date=at(days_ago), pinned=pinned)

    other = SimpleNamespace(is_user=False, entity=None, message=None, unread_count=0, date=at(0.1), pinned=False)

    class Client:
        def iter_dialogs(self):
            # an old pinned chat first, 400 group-like dialogs, the candidates, then an old chat ends the scan
            items = [dialog(9, days_ago=90, pinned=True), *[other] * 400,
                     dialog(1), dialog(2, out=True), dialog(3, unread=0), dialog(5, contact=True),
                     dialog(6, bot=True), dialog(42), dialog(7), dialog(4, days_ago=40), dialog(8)]

            async def gen():
                for d in items:
                    yield d
            return gen()
    account = SimpleNamespace(client=Client(), cfg=SimpleNamespace(owner_ids=[42]), channels={7: object()})
    refs = await PeopleChannel(account).backlog(30)
    assert [r.user_id for r in refs] == [1, 3]  # 3 was read but never answered; 8 comes after the old chat 4


async def test_nothing_is_marked_read_when_the_send_cannot_happen_now(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, max_sends_per_hour=0)
    await match(m)
    people.incoming(555, "привет)")
    m.llm.reply = {"action": "reply", "messages": ["привет"]}
    await answered(m, people, clock)
    clock.t += 31
    await m.tick()
    assert people.read == {} and people.sent_texts(555) == []


async def test_a_triage_cut_short_by_a_restart_resumes_at_start(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve")
    people.incoming(777, "привет) я из бибинто")
    await m.on_stranger(PersonRef(777, None, "Лена"))
    m.store.update(777, due_at=None)  # the process died in the middle of the triage
    await m.startup()
    clock.t += 1
    await m.tick()
    assert "Вам впервые написал(а) Лена" in inbox[-1]


async def test_a_reply_held_by_quiet_hours_is_written_again_in_the_morning(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, quiet_hours="00:00-09:00")
    clock.t = 1_800_000_000 - (1_800_000_000 % 86400) + 3600  # 01:00 UTC
    await match(m)
    people.incoming(555, "привет)")
    m.llm.reply = {"action": "reply", "messages": ["ночной ответ"]}
    await answered(m, people, clock)
    clock.t += 31
    await m.tick()  # quiet hours: not sent
    m.llm.reply = {"action": "reply", "messages": ["утренний ответ"]}
    clock.t += 9 * 3600
    for _ in range(3):
        await m.tick()
        clock.t += 120
    assert people.sent_texts(555) == ["утренний ответ"]


async def test_nothing_is_read_when_the_person_wrote_more_before_the_reply(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False)
    await match(m)
    people.incoming(555, "привет)")
    m.llm.reply = {"action": "reply", "messages": ["привет"]}
    await answered(m, people, clock)
    people.incoming(555, "давай встретимся?")  # arrives before the queued reply goes out
    clock.t += 31
    await m.tick()
    assert people.read == {} and people.sent_texts(555) == []


async def test_ignore_mode_does_not_resume_a_cut_short_triage(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve")
    people.incoming(777, "привет)")
    await m.on_stranger(PersonRef(777, None, "Лена"))
    m.store.update(777, due_at=None)
    await m.command("chat_mode", "strangers:ignore")
    await m.startup()
    clock.t += 60
    await m.tick()
    assert not any("Вам впервые написал(а)" in t for t in inbox)


async def test_a_queued_stranger_is_left_alone_once_strangers_are_ignored(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve")
    people.incoming(777, "привет)")
    await m.on_stranger(PersonRef(777, None, "Лена"))  # queued for triage
    await m.command("chat_mode", "strangers:ignore")
    clock.t += 60
    await m.tick()
    assert not any("Вам впервые написал(а)" in t for t in inbox)


# ------------------------------------------------------------------ lively pace, bursts, quotes
def test_a_lively_exchange_gets_quick_answers(tmp_path):
    import random as _r
    m, people, inbox, clock = make(tmp_path, reply_delay_min_s=5, reply_delay_max_s=5400, long_delay_chance=0)
    m.rng = _r.Random(3)
    lively = [ChatMessage(1, True, "a", 1000.0), ChatMessage(2, False, "b", 1012.0),
              ChatMessage(3, True, "c", 1020.0), ChatMessage(4, False, "d", 1030.0)]
    assert all(5 <= m._reply_delay(lively) <= 17 for _ in range(50))


async def test_a_turn_can_be_a_few_quick_messages(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False)
    await match(m)
    people.incoming(555, "ахах а ты что делаешь?")
    m.llm.reply = {"action": "reply", "messages": ["ахах", "да ничего, с пар иду", "а ты?", "кстати видел твой пост"]}
    await answered(m, people, clock)
    clock.t += 31
    await m.tick()
    assert people.sent_texts(555) == ["ахах", "да ничего, с пар иду", "а ты?", "кстати видел твой пост"]


async def test_an_earlier_question_can_be_answered_as_a_quote(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False)
    await match(m)
    asked = people.incoming(555, "а ты где учишься?")
    people.incoming(555, "ой, и ещё: ты кино любишь?")
    m.llm.reply = {"action": "reply", "messages": ["в плехе", "кино да, особенно в кинотеатре"],
                   "reply_to": asked}
    await answered(m, people, clock)
    text, _ = prompt_of(m)
    assert f"[#{asked}]" in text and "reply_to" in m.llm.calls[-1][0]["content"]
    clock.t += 31
    await m.tick()
    assert people.replies == [asked, None]  # only the first message quotes the earlier question


@pytest.mark.parametrize("quote", [None, 12345, "abc", "newest"])
async def test_plain_messages_unless_a_valid_earlier_message_is_quoted(tmp_path, quote):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False)
    await match(m)
    newest = people.incoming(555, "привет)")
    m.llm.reply = {"action": "reply", "messages": ["привет"], "reply_to": newest if quote == "newest" else quote}
    await answered(m, people, clock)
    clock.t += 31
    await m.tick()
    assert people.replies == [None]


async def test_the_owners_own_text_is_never_sent_as_the_agents_quote(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve", write_first=False)
    await match(m)
    asked = people.incoming(555, "а ты где учишься?")
    people.incoming(555, "ау")
    m.llm.reply = {"action": "reply", "messages": ["в плехе"], "reply_to": asked}
    await answered(m, people, clock)
    await m.answer_card(9000 + len(inbox), "в рэу)")
    assert people.replies == [None]


async def test_a_meeting_draft_carries_its_own_quote_and_four_messages(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False)
    await match(m)
    old = people.incoming(555, "а ты где учишься?")
    m.store.set("reply_to:555", str(old))  # left from an earlier, cancelled answer
    people.incoming(555, "давай встретимся?")
    m.llm.reply = {"action": "reply", "messages": ["давай", "в субботу", "на кг", "в шесть"]}
    await answered(m, people, clock)
    assert "в шесть" in inbox[-1] and m.store.get("reply_to:555") is None
    await m.answer_card(9000 + len(inbox), "ок")
    assert people.sent_texts(555) == ["давай", "в субботу", "на кг", "в шесть"] and people.replies[0] is None


# ------------------------------------------------------------------ plan, ending, follow-ups, meetings
async def test_the_prompt_asks_for_a_plan_and_psychology(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve", write_first=False)
    await match(m)
    people.incoming(555, "привет)")
    m.llm.reply = {"action": "reply", "messages": ["привет"], "memory": "план: " + "а" * 3000}
    await answered(m, people, clock)
    system = m.llm.calls[-1][0]["content"]
    assert "Your plan for this person" in system and "End on a high" in system and "hold" in system
    assert len(m.store.person(555).memory) <= 2500


async def test_a_quiet_chat_gets_one_light_follow_up(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, follow_up_min_h=24,
                                   follow_up_max_h=24)
    await match(m)
    people.incoming(555, "привет)")
    m.llm.reply = {"action": "reply", "messages": ["привет, как день?"]}
    await answered(m, people, clock)
    clock.t += 31
    await m.tick()
    assert people.sent_texts(555) == ["привет, как день?"]
    m.llm.reply = lambda msgs: ({"action": "reply", "messages": ["кстати, как прошёл экзамен?"]}
                                if "## Follow-up" in msgs[1]["content"] else {"action": "wait"})
    clock.t += 23 * 3600
    await m.tick()
    assert people.sent_texts(555) == ["привет, как день?"]  # not yet
    clock.t += 2 * 3600
    for _ in range(3):
        await m.tick()
        clock.t += 130
    assert people.sent_texts(555) == ["привет, как день?", "кстати, как прошёл экзамен?"]
    clock.t += 5 * 86400
    for _ in range(3):
        await m.tick()
        clock.t += 130
    assert len(people.sent_texts(555)) == 2  # never two in a row without an answer


async def test_no_follow_up_when_they_answered_or_the_model_lets_it_be(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, follow_up_min_h=24,
                                   follow_up_max_h=24)
    await match(m)
    people.incoming(555, "привет)")
    m.llm.reply = {"action": "reply", "messages": ["привет"]}
    await answered(m, people, clock)
    clock.t += 31
    await m.tick()
    m.llm.reply = {"action": "wait"}  # a cold chat: let it be
    clock.t += 25 * 3600
    for _ in range(2):
        await m.tick()
        clock.t += 130
    assert people.sent_texts(555) == ["привет"] and m.store.get("follow_up:555") is None


async def test_a_promise_to_write_later_is_kept(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False)
    await match(m)
    people.incoming(555, "ахах, ладно")
    m.llm.reply = {"action": "reply", "messages": ["ладно, побежал на тренировку, вечером напишу"], "next_check_h": 4}
    await answered(m, people, clock)
    clock.t += 31
    await m.tick()
    due = float(m.store.get("follow_up:555"))
    assert 3.9 * 3600 < due - clock() < 4.1 * 3600


async def test_a_meeting_gets_a_holding_reply_and_free_times_for_the_owner(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False)
    await match(m)
    people.incoming(555, "давай встретимся на выходных?")
    m.llm.reply = {"action": "escalate", "category": "meeting", "hold": ["давай, гляну расписание и напишу"],
                   "slots": "суббота после 14, воскресенье весь день", "messages": ["давай в субботу на кг?"]}
    await answered(m, people, clock)
    card = 9000 + len(inbox)
    assert "Свободно по графику: суббота после 14" in inbox[-1] and "гляну расписание" in inbox[-1]
    clock.t += 31
    await m.tick()
    assert people.sent_texts(555) == ["давай, гляну расписание и напишу"]
    m.llm.reply = lambda msgs: ({"action": "reply", "messages": ["в субботу в 18 на кг?"]}
                                if "owner's instruction" in msgs[1]["content"] else {"action": "wait"})
    await m.answer_card(card, "суббота в 18, кг")
    assert people.sent_texts(555)[-1] == "в субботу в 18 на кг?" and "Пишу" in " ".join(inbox)


async def test_a_holding_reply_never_names_a_time_or_place(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False)
    await match(m)
    people.incoming(555, "давай встретимся?")
    m.llm.reply = {"action": "escalate", "category": "meeting", "hold": ["давай, встретимся в субботу"],
                   "messages": ["в субботу?"]}
    await answered(m, people, clock)
    clock.t += 31
    await m.tick()
    assert people.sent_texts(555) == [] and "Собеседнику агент ответит" not in inbox[-1]


async def test_the_owner_can_still_send_their_own_words_on_a_meeting_card(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve", write_first=False)
    await match(m)
    people.incoming(555, "давай встретимся?")
    m.llm.reply = {"action": "escalate", "category": "meeting", "messages": ["давай"]}
    await answered(m, people, clock)
    await m.answer_card(9000 + len(inbox), "!давай в субботу в шесть у метро")
    assert people.sent_texts(555) == ["давай в субботу в шесть у метро"]


@pytest.mark.parametrize("text, meeting", [("выбираюсь в центр, погулять или в кофейню", False),
                                           ("люблю погулять вечером", False), ("давай погуляем в субботу", True),
                                           ("может сходим в кино?", True)])
def test_only_invitations_count_as_meetings(text, meeting):
    from doppelgram.conversation import classify_outgoing
    assert (classify_outgoing(text) == "meeting") is meeting
    assert (classify_incoming(text, None) == "meeting") is meeting


# ------------------------------------------------------------------ review of plans, follow-ups, meetings
async def meeting_with_hold(tmp_path, hold="давай, гляну расписание и напишу", **over):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, intimacy=True, **over)
    await match(m)
    people.incoming(555, "давай встретимся?")
    m.llm.reply = {"action": "escalate", "category": "meeting", "hold": [hold], "slots": "суббота",
                   "messages": ["давай в субботу?"]}
    await answered(m, people, clock)
    return m, people, inbox, clock, 9000 + len(inbox)


async def test_a_holding_reply_stops_when_the_mode_turns_to_approve(tmp_path):
    m, people, inbox, clock, card = await meeting_with_hold(tmp_path)
    await m.command("chat_mode", "approve")
    clock.t += 31
    await m.tick()
    assert people.sent_texts(555) == []


@pytest.mark.parametrize("hold", ["давай в субботу в 18 у метро", "давай завтра", "ок, в 19:00"])
async def test_a_holding_reply_naming_a_time_is_not_sent(tmp_path, hold):
    m, people, inbox, clock, card = await meeting_with_hold(tmp_path, hold=hold)
    clock.t += 31
    await m.tick()
    assert people.sent_texts(555) == []


async def test_the_owner_taking_over_during_an_instruction_wins(tmp_path):
    m, people, inbox, clock, card = await meeting_with_hold(tmp_path)

    def answer(msgs):
        if "owner's instruction" in msgs[1]["content"]:
            m.store.update(555, state="owner")  # /self while the agent writes
            return {"action": "reply", "messages": ["в субботу в 18?"]}
        return {"action": "wait"}
    m.llm.reply = answer
    await m.answer_card(card, "суббота в 18")
    assert "в субботу в 18?" not in people.sent_texts(555) and "забрали" in inbox[-1]


@pytest.mark.parametrize("reply", [{"action": "escalate", "category": "money", "messages": ["куплю тебе айфон"]},
                                   {"action": "reply", "messages": ["куплю тебе айфон, только приходи"]}])
async def test_an_instruction_never_sends_what_the_owner_must_decide(tmp_path, reply):
    m, people, inbox, clock, card = await meeting_with_hold(tmp_path)
    m.llm.reply = lambda msgs: reply if "owner's instruction" in msgs[1]["content"] else {"action": "wait"}
    await m.answer_card(card, "суббота в 18")
    assert "айфон" not in " ".join(people.sent_texts(555)) and "ещё раз" in inbox[-1]


async def test_an_instruction_respects_the_daily_budget(tmp_path):
    m, people, inbox, clock, card = await meeting_with_hold(tmp_path, daily_budget_usd=0.1)
    m.store.log_spend(clock(), 0.5)
    calls = len(m.llm.calls)
    await m.answer_card(card, "суббота в 18")
    assert len(m.llm.calls) == calls and "ещё раз" in inbox[-1]


async def test_a_holding_reply_does_not_replace_the_owners_queued_answer(tmp_path):
    m, people, inbox, clock, card = await meeting_with_hold(tmp_path)
    real_typing = people.typing

    async def owner_answers_in_quiet_hours(user_id, seconds):
        people.typing = real_typing
        m.store.set_outbox(555, m.store.outbox_row(555)[0], ["в субботу в 18"], clock() + 9 * 3600, approved=True,
                           owner_text=True, keep_cards=True)  # the owner's answer, waiting for the morning
        m._bump(555)
        await real_typing(user_id, seconds)
    people.typing = owner_answers_in_quiet_hours
    clock.t += 31
    await m.tick()
    assert m.store.outbox_row(555)[1] == ["в субботу в 18"]


async def test_a_follow_up_held_by_quiet_hours_comes_in_the_morning(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, quiet_hours="00:00-09:00",
                                   follow_up_min_h=1, follow_up_max_h=1)
    clock.t = 1_800_000_000 - (1_800_000_000 % 86400) + 20 * 3600  # 20:00 UTC
    await match(m)
    people.incoming(555, "привет)")
    m.llm.reply = {"action": "reply", "messages": ["привет"]}
    await answered(m, people, clock)
    clock.t += 31
    await m.tick()  # sent at about 20:00
    m.llm.reply = lambda msgs: ({"action": "reply", "messages": ["как вечер прошёл?"]}
                                if "## Follow-up" in msgs[1]["content"] else {"action": "wait"})
    clock.t = clock.t - (clock.t % 86400) + 86400 - 600  # 23:50: the follow-up is written...
    m.store.set("follow_up:555", str(clock.t))
    await m.tick()
    clock.t += 1200  # ...but quiet hours started before it went out
    await m.tick()
    assert people.sent_texts(555) == ["привет"]
    clock.t += 10 * 3600
    for _ in range(4):
        await m.tick()
        clock.t += 130
    assert people.sent_texts(555) == ["привет", "как вечер прошёл?"]


async def test_a_promised_follow_up_counts_from_when_the_answer_went_out(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, reply_delay_min_s=7200,
                                   reply_delay_max_s=7200)
    await match(m)
    people.incoming(555, "ахах ладно")
    m.llm.reply = {"action": "reply", "messages": ["побежал, через час напишу"], "next_check_h": 1}
    await answered(m, people, clock)
    assert m.store.get("follow_up:555") is None  # not counting yet: the answer waits two hours
    clock.t += 7201
    await m.tick()
    assert abs(float(m.store.get("follow_up:555")) - (clock() + 3600)) < 60


async def test_finishing_an_old_send_never_removes_a_newer_queued_answer(tmp_path):
    from test_conversation import ConversationStore
    store = ConversationStore(":memory:")
    store.add_person(PersonRef(1, None, "a"), "b", "", 0.0)
    store.set_outbox(1, 5, ["новый ответ владельца"], 0.0, approved=True, owner_text=True)
    store.commit_send(1, 77, 1.0, 5, [], ["старый hold"])  # the old series ends after the new one was queued
    assert store.outbox_row(1) == (5, ["новый ответ владельца"])


@pytest.mark.parametrize("hold", ["давай в шесть у парка", "гляну, может в кафе", "давай около метро"])
async def test_holding_replies_with_times_or_places_in_words_are_not_sent(tmp_path, hold):
    m, people, inbox, clock, card = await meeting_with_hold(tmp_path, hold=hold)
    clock.t += 31
    await m.tick()
    assert people.sent_texts(555) == []


async def test_switching_to_approve_keeps_the_meeting_card(tmp_path):
    m, people, inbox, clock, card = await meeting_with_hold(tmp_path)
    await m.command("chat_mode", "approve")
    clock.t += 31
    await m.tick()
    m.llm.reply = lambda msgs: ({"action": "reply", "messages": ["в субботу в 18?"]}
                                if "owner's instruction" in msgs[1]["content"] else {"action": "wait"})
    await m.answer_card(card, "суббота в 18")
    assert people.sent_texts(555) == ["в субботу в 18?"]


@pytest.mark.parametrize("event", ["self", "intimacy"])
async def test_events_while_the_owner_is_told_what_the_agent_writes_cancel_the_send(tmp_path, event):
    m, people, inbox, clock, card = await meeting_with_hold(tmp_path)
    m.llm.reply = lambda msgs: ({"action": "reply", "messages": ["в субботу в 18?"]}
                                if "owner's instruction" in msgs[1]["content"] else {"action": "wait"})
    real_notify = m._notify

    async def notify(text, image=None):
        if text.startswith("Пишу"):
            m._notify = real_notify
            if event == "self":
                await m.command("self", 555)
            else:
                await m.command("chat_mode", "intimacy:off")
        return await real_notify(text, image)
    m._notify = notify
    await m.answer_card(card, "суббота в 18")
    assert "в субботу в 18?" not in people.sent_texts(555)


async def test_self_while_the_owner_is_told_keeps_the_chat_with_the_owner(tmp_path):
    m, people, inbox, clock, card = await meeting_with_hold(tmp_path)
    m.llm.reply = lambda msgs: ({"action": "reply", "messages": ["в субботу в 18?", "на кг"]}
                                if "owner's instruction" in msgs[1]["content"] else {"action": "wait"})
    real_notify = m._notify

    async def notify(text, image=None):
        if text.startswith("Пишу"):
            m._notify = real_notify
            await m.command("self", 555)
        return await real_notify(text, image)
    m._notify = notify
    await m.answer_card(card, "суббота в 18")
    for _ in range(3):
        clock.t += 31
        await m.tick()
    assert m.store.person(555).state == "owner" and people.sent_texts(555) == []


async def test_a_chat_paused_without_a_card_is_picked_up_again_at_start(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve", write_first=False)
    await match(m)
    people.incoming(555, "ты бот?")
    m.store.update(555, state="paused", due_at=None)  # the process died while the card was being sent
    await m.startup()
    clock.t += 1
    await m.tick()
    assert "бот" in inbox[-1] and m.store.open_card(555) is not None


@pytest.mark.parametrize("event", ["self", "intimacy"])
async def test_events_during_a_quiet_hours_instruction_cancel_it(tmp_path, event):
    from dataclasses import replace as _replace
    m, people, inbox, clock, card = await meeting_with_hold(tmp_path)
    m.cfg = _replace(m.cfg, quiet_hours="00:00-23:59")  # the owner answers the card at night
    m.llm.reply = lambda msgs: ({"action": "reply", "messages": ["в субботу в 18?"]}
                                if "owner's instruction" in msgs[1]["content"] else {"action": "wait"})
    real_notify = m._notify

    async def notify(text, image=None):
        if text.startswith("Пишу"):
            m._notify = real_notify
            if event == "self":
                await m.command("self", 555)
            else:
                await m.command("chat_mode", "intimacy:off")
        return await real_notify(text, image)
    m._notify = notify
    await m.answer_card(card, "суббота в 18")
    assert m.store.outbox_row(555) is None and "ничего не отправляю" in inbox[-1]


async def test_a_cancelled_answer_gives_the_card_back(tmp_path):
    m, people, inbox, clock, card = await meeting_with_hold(tmp_path)
    m.llm.reply = lambda msgs: ({"action": "reply", "messages": ["в субботу в 18?"]}
                                if "owner's instruction" in msgs[1]["content"] else {"action": "wait"})
    real_notify = m._notify

    async def notify(text, image=None):
        if text.startswith("Пишу"):
            m._notify = real_notify
            await m.command("chat_mode", "intimacy:off")
        return await real_notify(text, image)
    m._notify = notify
    await m.answer_card(card, "суббота в 18")
    assert m.store.open_card(555) is not None  # the owner can still answer, the chat is not stuck
    await m.answer_card(card, "!давай в субботу")
    assert people.sent_texts(555)[-1] == "давай в субботу"


async def test_messages_surely_not_sent_after_a_failed_send_stay_on_the_card(tmp_path):
    from doppelgram.transport import TransportError
    m, people, inbox, clock, card = await owner_ok_card(tmp_path, ["раз", "два", "три"])
    real_send = people.send
    count = {"n": 0}

    async def flaky(user_id, text, reply_to=None):
        count["n"] += 1
        if count["n"] == 2:
            raise TransportError("timeout")
        return await real_send(user_id, text, reply_to=reply_to)
    people.send = flaky
    await m.answer_card(card, "ок")
    people.send = real_send
    assert people.sent_texts(555) == ["раз"] and "Не отправлено: «три»" in inbox[-1]
    await m.answer_card(card, "ок")
    assert people.sent_texts(555) == ["раз", "три"]


async def test_a_send_cut_by_a_crash_is_found_even_after_many_new_messages(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, history_messages=30)
    await match(m)
    people.incoming(555, "давай встретимся?")
    m.llm.reply = {"action": "reply", "messages": ["в субботу?"]}
    await answered(m, people, clock)
    card = 9000 + len(inbox)

    class Crash(Exception):
        pass
    real_commit = m.store.commit_send

    def crash(*a, **k):
        m.store.commit_send = real_commit
        raise Crash()
    m.store.commit_send = crash
    with pytest.raises(Crash):
        await m.answer_card(card, "!в субботу в шесть")  # the owner's own text: it is sent even if stale
    for i in range(31):
        people.incoming(555, f"сообщение {i}")
    restarted = ConversationManager(m.cfg, m.llm, people, m.store, m._notify, clock=clock, sleep=clock.sleep)
    clock.t += 61
    await restarted.tick()
    assert people.sent_texts(555).count("в субботу в шесть") == 1


async def test_an_explicit_wait_revisits_the_handled_message_at_the_requested_time(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False)
    await match(m)
    people.incoming(555, "сейчас занята, напиши позже")
    m.llm.reply = {"action": "wait", "next_check_h": 1}
    await answered(m, people, clock)
    assert len(m.llm.calls) == 1 and people.sent_texts(555) == []
    m.llm.reply = {"action": "reply", "messages": ["как прошёл день?"]}
    clock.t += 3500
    await m.tick()
    assert len(m.llm.calls) == 1
    clock.t += 101
    await m.tick()
    assert len(m.llm.calls) == 2
    assert "You chose to wait before replying" in m.llm.calls[-1][1]["content"]
    clock.t += 121
    await m.tick()
    assert people.sent_texts(555) == ["как прошёл день?"]
    clock.t += 5 * 86400
    await m.tick()
    assert len(m.llm.calls) == 2  # one deferred reply, never a repeated nudge


@pytest.mark.parametrize("event", ["incoming", "self"])
async def test_a_later_message_or_takeover_cancels_the_deferred_check(tmp_path, event):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False)
    await match(m)
    people.incoming(555, "я занята")
    m.llm.reply = {"action": "wait", "next_check_h": 1}
    await answered(m, people, clock)
    if event == "incoming":
        m.llm.reply = {"action": "wait"}
        people.incoming(555, "я пока ещё занята")
        await answered(m, people, clock)
    else:
        await m.command("self", 555)
    calls = len(m.llm.calls)
    clock.t += 7200
    await m.tick()
    assert len(m.llm.calls) == calls and people.sent_texts(555) == []
    assert m.store.get("follow_up:555") is None


async def test_an_old_instruction_does_not_delete_a_newer_approved_answer(tmp_path):
    m, people, inbox, clock, card = await meeting_with_hold(tmp_path)
    m.llm.reply = lambda msgs: ({"action": "reply", "messages": ["в субботу в шесть?"]}
                                if "owner's instruction" in msgs[1]["content"] else
                                {"action": "reply", "messages": ["новый черновик"]})
    real_notify = m._notify
    newer = {}

    async def notify(text, image=None):
        if text.startswith("Пишу"):
            m._notify = real_notify
            await m.command("lead", 555)
            await m.command("chat_mode", "approve")
            people.incoming(555, "как прошёл день?")
            await answered(m, people, clock)
            m.cfg = replace(m.cfg, quiet_hours="00:00-23:59")
            await m.answer_card(m.store.open_card(555)["control_msg_id"], "ок")
            newer["row"] = m.store.outbox_row(555)
        return await real_notify(text, image)
    m._notify = notify
    await m.answer_card(card, "суббота в шесть")
    assert newer["row"] is not None and newer["row"][1] == ["новый черновик"]
    assert m.store.outbox_row(555) == newer["row"]
    m.cfg = replace(m.cfg, quiet_hours="")
    clock.t += 2 * 86400
    await m.tick()
    assert people.sent_texts(555) == ["новый черновик"]


@pytest.mark.parametrize("deferred_reply", [False, True])
async def test_a_scheduled_check_survives_a_temporary_model_error(tmp_path, deferred_reply):
    from doppelgram.llm import LLMError
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False)
    await match(m)
    people.incoming(555, "привет")
    m.llm.reply = ({"action": "wait", "next_check_h": 1} if deferred_reply else
                   {"action": "reply", "messages": ["позже напишу"], "next_check_h": 1})
    await answered(m, people, clock)
    if not deferred_reply:
        clock.t += 31
        await m.tick()
    calls = len(m.llm.calls)

    def unavailable(messages):
        raise LLMError("temporary failure")
    m.llm.reply = unavailable
    clock.t += 3601
    await m.tick()
    assert len(m.llm.calls) == calls + 1
    m.llm.reply = {"action": "reply", "messages": ["как прошёл день?"]}
    clock.t += m.RETRY_S + 1
    await m.tick()
    clock.t += 121
    await m.tick()
    assert len(m.llm.calls) == calls + 2
    assert people.sent_texts(555)[-1] == "как прошёл день?"


# ------------------------------------------------------------------ late evening and night
def at_moscow(hour, minute=0):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime(2027, 1, 15, hour, minute, tzinfo=ZoneInfo("Europe/Moscow")).timestamp()


async def test_a_quiet_chat_written_to_late_is_answered_in_the_morning(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, timezone="Europe/Moscow",
                                   late_hours="23:30-01:30", quiet_hours="01:30-09:00")
    await match(m)
    clock.t = at_moscow(0, 10)
    people.chats[555] = [ChatMessage(1, False, "привет, не спишь?", clock.t)]
    m.on_message(555)
    clock.t += 11
    await m.tick()
    assert m.llm.calls == [] and m.store.person(555).due_at >= at_moscow(9) + 60
    clock.t = at_moscow(9, 40)
    m.llm.reply = {"action": "reply", "messages": ["сорри, уснул) привет"]}
    for _ in range(3):
        await m.tick()
        clock.t += 60
    assert people.sent_texts(555) == ["сорри, уснул) привет"]


async def test_a_lively_chat_late_in_the_evening_goes_on(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, timezone="Europe/Moscow",
                                   late_hours="23:30-01:30", quiet_hours="01:30-09:00")
    await match(m)
    clock.t = at_moscow(0, 10)
    people.chats[555] = [ChatMessage(1, False, "ахах", clock.t - 300), ChatMessage(2, True, "ну да", clock.t - 200),
                         ChatMessage(3, False, "а ты завтра рано встаёшь?", clock.t)]
    m.store.record_sent(555, 2, clock.t - 200)  # "ну да" was the agent's
    m.on_message(555)
    m.llm.reply = {"action": "reply", "messages": ["да, пары с утра, скоро спать"]}
    clock.t += 11
    await m.tick()
    clock.t += 31
    await m.tick()
    assert people.sent_texts(555)[-1] == "да, пары с утра, скоро спать"
    assert "say good night" in m.llm.calls[-1][0]["content"]


async def test_no_follow_up_at_night(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, timezone="Europe/Moscow",
                                   late_hours="23:30-01:30", quiet_hours="01:30-09:00")
    await match(m)
    clock.t = at_moscow(3)
    people.chats[555] = [ChatMessage(1, False, "привет", clock.t - 86400),
                         ChatMessage(2, True, "привет)", clock.t - 80000)]
    m.store.record_sent(555, 2, clock.t - 80000)
    m.store.update(555, handled_upto=1)
    m.store.set("follow_up:555", str(clock.t - 1))
    m.store.update(555, due_at=clock.t)
    await m.tick()
    assert m.llm.calls == [] and float(m.store.get("follow_up:555")) >= at_moscow(9)


async def night_chat(tmp_path, when, their_gap, our_gap):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, timezone="Europe/Moscow",
                                   late_hours="23:30-01:30", quiet_hours="01:30-09:00")
    await match(m)
    clock.t = when
    people.chats[555] = [ChatMessage(1, True, "ладно, мне завтра рано, спокойной ночи", clock.t - our_gap),
                         ChatMessage(2, False, "ну ещё пять минут(", clock.t - their_gap)]
    m.store.record_sent(555, 1, clock.t - our_gap)
    m.on_message(555)
    m.llm.reply = {"action": "reply", "messages": ["ахах ладно, пять минут"]}
    clock.t += 11
    await m.tick()
    clock.t += 200
    await m.tick()
    return m, people, inbox, clock


async def test_right_after_good_night_a_quick_answer_still_gets_a_reply(tmp_path):
    m, people, inbox, clock = await night_chat(tmp_path, at_moscow(1, 50), their_gap=60, our_gap=180)
    assert people.sent_texts(555)[-1] == "ахах ладно, пять минут"
    assert "## Night" in m.llm.calls[-1][1]["content"]


async def test_a_late_answer_to_good_night_waits_for_the_morning(tmp_path):
    m, people, inbox, clock = await night_chat(tmp_path, at_moscow(1, 50), their_gap=1200, our_gap=1500)
    assert people.sent_texts(555) == ["ладно, мне завтра рано, спокойной ночи"] and m.llm.calls == []


async def test_deep_in_the_night_even_a_lively_chat_stops(tmp_path):
    m, people, inbox, clock = await night_chat(tmp_path, at_moscow(3, 30), their_gap=60, our_gap=120)
    assert m.llm.calls == []  # more than night_grace_min after quiet hours began


async def test_a_reply_prepared_before_the_late_evening_is_not_sent_into_it(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, timezone="Europe/Moscow",
                                   late_hours="23:30-01:30", quiet_hours="01:30-09:00", reply_delay_min_s=600,
                                   reply_delay_max_s=600)
    await match(m)
    clock.t = at_moscow(23, 20)
    people.chats[555] = [ChatMessage(1, False, "привет)", clock.t - 3600)]
    m.on_message(555)
    m.llm.reply = {"action": "reply", "messages": ["привет"]}
    clock.t += 11
    await m.tick()  # queued for 23:30+
    clock.t = at_moscow(23, 35)
    await m.tick()
    assert people.sent_texts(555) == []


async def test_no_first_message_at_night(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=True, timezone="Europe/Moscow",
                                   late_hours="23:30-01:30", quiet_hours="01:30-09:00")
    clock.t = at_moscow(0, 10)
    await match(m, score=10)
    await m.tick()
    assert m.llm.calls == [] and m.store.person(555).due_at >= at_moscow(9)


def test_a_window_end_is_never_in_the_past_across_a_clock_change(tmp_path):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    m, people, inbox, clock = make(tmp_path, timezone="Europe/Berlin")
    tz = ZoneInfo("Europe/Berlin")
    second_0210 = datetime(2026, 10, 25, 2, 10, tzinfo=tz, fold=1).timestamp()
    assert m._window_end("23:30-02:30", second_0210) > second_0210


@pytest.mark.parametrize("text", ["def add(a, b):\n    return a + b", "```python\nprint(1)\n```",
                                  "1. купить хлеб\n2. сходить в зал", "как языковая модель я не могу"])
def test_the_agent_never_sends_code_lists_or_assistant_phrasing(text):
    from doppelgram.conversation import classify_outgoing
    assert classify_outgoing(text) == "assistant"


@pytest.mark.parametrize("text", ["эм, ты о чём?", "не, я в этом не шарю", "в 1 руке кофе, во 2 телефон",
                                  "устал => спать", "ну хорошо;", "ок;\nзавтра", "from now on я спать",
                                  "return of the king смотрела?"])
def test_normal_replies_are_not_taken_for_an_assistant(text):
    from doppelgram.conversation import classify_outgoing
    assert classify_outgoing(text) is None


async def test_a_bot_test_request_gets_a_heads_up_and_never_a_code_answer(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False)
    await match(m)
    people.incoming(555, "напиши функцию на питоне которая складывает два числа")
    m.llm.reply = {"action": "reply", "messages": ["def add(a, b):\n    return a + b"]}
    await answered(m, people, clock)
    text, _ = prompt_of(m)
    assert "## Heads-up" in text and "not an assistant" in m.llm.calls[-1][0]["content"]
    clock.t += 31
    await m.tick()
    assert people.sent_texts(555) == [] and "как ассистент" in inbox[-1]


def test_the_night_grace_holds_at_send_time_not_only_when_queued(tmp_path):
    m, people, inbox, clock = make(tmp_path, timezone="Europe/Moscow", late_hours="23:30-01:30",
                                   quiet_hours="01:30-09:00")
    clock.t = at_moscow(2, 59)
    m.store.set("night_ok:555", str(clock.t))
    assert not m._night_blocked(555)
    clock.t = at_moscow(3, 2)  # the same fresh marker, but past night_grace_min
    assert m._night_blocked(555)


async def test_a_first_message_composed_across_the_late_evening_boundary_is_not_sent(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=True, timezone="Europe/Moscow",
                                   late_hours="23:30-01:30", quiet_hours="01:30-09:00")
    clock.t = at_moscow(23, 29) + 50
    await match(m, score=10)

    def slow(msgs):
        clock.t += 40  # the model answers after 23:30
        return {"action": "reply", "messages": ["привет) как дела?"]}
    m.llm.reply = slow
    for _ in range(4):
        await m.tick()
        clock.t += 60
    assert people.sent_texts(555) == [] and m.store.get("night_ok:555") is None


async def night_meeting(tmp_path, think_s):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, intimacy=True,
                                   timezone="Europe/Moscow", late_hours="23:30-01:30", quiet_hours="01:30-09:00")
    await match(m)
    clock.t = at_moscow(0, 20)
    people.chats[555] = [ChatMessage(1, True, "ахах да", clock.t - 120),
                         ChatMessage(2, False, "давай встретимся?", clock.t - 30)]
    m.store.record_sent(555, 1, clock.t - 120)

    def think(msgs):
        clock.t += think_s
        return {"action": "escalate", "category": "meeting", "hold": ["давай, гляну расписание и напишу"],
                "slots": "суббота", "messages": ["давай в субботу?"]}
    m.llm.reply = think
    await answered(m, people, clock)
    for _ in range(3):
        clock.t += 61
        await m.tick()
    return m, people, inbox


async def test_a_holding_reply_goes_out_in_a_lively_night_chat(tmp_path):
    m, people, inbox = await night_meeting(tmp_path, think_s=5)
    assert people.sent_texts(555)[-1] == "давай, гляну расписание и напишу"


async def test_no_holding_reply_is_promised_when_nobody_waits_at_night(tmp_path):
    m, people, inbox = await night_meeting(tmp_path, think_s=1800)  # the chat went quiet meanwhile
    assert m.llm.calls and people.sent_texts(555) == ["ахах да"]
    assert any("давай в субботу?" in n for n in inbox) and not any("гляну расписание" in n for n in inbox)


def test_night_grace_is_counted_in_real_minutes_across_a_clock_change(tmp_path):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    m, people, inbox, clock = make(tmp_path, timezone="Europe/Berlin")
    second_0245 = datetime(2026, 10, 25, 2, 45, tzinfo=ZoneInfo("Europe/Berlin"), fold=1).timestamp()
    assert m._minutes_into("01:30-09:00", second_0245) == 135


@pytest.mark.parametrize("text", ["from math import sqrt\nsqrt(9)", "import numpy as np", "import os.path"])
def test_imports_are_taken_for_code(text):
    from doppelgram.conversation import classify_outgoing
    assert classify_outgoing(text) == "assistant"


async def test_a_holding_reply_promised_at_the_end_of_the_night_grace_still_goes_out(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, intimacy=True,
                                   timezone="Europe/Moscow", late_hours="23:30-01:30", quiet_hours="01:30-09:00")
    await match(m)
    clock.t = at_moscow(2, 59) + 30
    people.chats[555] = [ChatMessage(1, True, "ахах да", clock.t - 60),
                         ChatMessage(2, False, "давай встретимся?", clock.t - 20)]
    m.store.record_sent(555, 1, clock.t - 60)
    m.llm.reply = {"action": "escalate", "category": "meeting", "hold": ["давай, гляну расписание и напишу"],
                   "slots": "суббота", "messages": ["давай в субботу?"]}
    await answered(m, people, clock)
    assert any("гляну расписание" in n for n in inbox)  # the owner is told it goes out
    clock.t = at_moscow(3, 2)  # the notification took a while: past the grace
    for _ in range(2):
        await m.tick()
        clock.t += 61
    assert people.sent_texts(555)[-1] == "давай, гляну расписание и напишу"
    people.incoming(555, "ок")
    await answered(m, people, clock)
    assert people.sent_texts(555)[-1] == "давай, гляну расписание и напишу"  # the rest waits for the morning


def test_a_quiet_window_starting_in_a_skipped_hour_starts_at_the_clock_change(tmp_path):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    m, people, inbox, clock = make(tmp_path, timezone="Europe/Berlin")
    tz = ZoneInfo("Europe/Berlin")
    jump = datetime(2027, 3, 28, 3, 0, tzinfo=tz).timestamp()  # 02:00 CET jumps to 03:00 CEST
    assert m._minutes_into("02:30-09:00", jump + 300) == 5
    assert m._minutes_into("01:30-09:00", jump + 300) == 35


@pytest.mark.parametrize("zone,day,hhmm,expect_utc", [
    ("Europe/Berlin", (2026, 10, 25), (2, 45), "2026-10-25T00:45"),  # repeated hour: the first 02:45
    ("Europe/Berlin", (2027, 3, 28), (2, 30), "2027-03-28T01:00"),  # skipped hour: the jump
    ("Australia/Lord_Howe", (2026, 4, 5), (1, 45), "2026-04-04T14:45"),  # half-hour autumn change
    ("Asia/Kolkata", (2027, 1, 15), (1, 30), "2027-01-14T20:00"),  # no clock changes, +05:30
    ("Europe/Berlin", (2027, 3, 28), (2, 0), "2027-03-28T01:00:00"),  # exactly at the jump, to the second
])
def test_the_first_instant_of_a_wall_time(zone, day, hhmm, expect_utc):
    from datetime import UTC, date, datetime
    from zoneinfo import ZoneInfo

    from doppelgram.conversation import _first_instant_at
    ts = _first_instant_at(date(*day), *hhmm, ZoneInfo(zone))
    assert datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%S").startswith(expect_utc)


def test_only_the_promised_holding_reply_may_miss_the_night_grace(tmp_path):
    import json
    m, people, inbox, clock = make(tmp_path, timezone="Europe/Moscow", late_hours="23:30-01:30",
                                   quiet_hours="01:30-09:00")
    clock.t = at_moscow(3, 1)
    m.store.set("night_ok:555", str(clock.t - 60))
    m.store.set("hold:555", json.dumps(["гляну расписание и напишу"], ensure_ascii=False))
    m.store.set_outbox(555, 1, ["гляну расписание и напишу"], clock.t, keep_cards=True)
    assert not m._night_blocked(555)
    m.store.set_outbox(555, 2, ["а ты где учишься?"], clock.t)  # a regular reply replaced it
    assert m._night_blocked(555)


async def test_a_two_message_holding_reply_goes_out_whole_in_a_paused_chat(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, intimacy=True)
    await match(m)
    people.incoming(555, "давай встретимся?")
    m.llm.reply = {"action": "escalate", "category": "meeting", "hold": ["гляну расписание", "и напишу"],
                   "slots": "суббота", "messages": ["давай в субботу?"]}
    await answered(m, people, clock)
    assert m.store.person(555).state == "paused"
    for _ in range(4):
        clock.t += 61
        await m.tick()
    assert people.sent_texts(555)[-2:] == ["гляну расписание", "и напишу"]
