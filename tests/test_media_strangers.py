"""What people send besides text (stickers, photos, voice) and people who write first without a match."""

from __future__ import annotations

import io

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
async def test_a_minor_match_is_never_flirted_with(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, intimacy=True)
    await m.on_match("дайвинчик", ["https://t.me/masha_example"], [], "Лена, 16, Москва", 10, minor=True)
    person = m.store.person(555)
    assert person.minor and person.score is None  # never written to first
    people.incoming(555, "привет)")
    await answered(m, people, clock)
    system = m.llm.calls[-1][0]["content"]
    text, _ = prompt_of(m)
    assert "Intimacy is allowed" not in system and "under 18: keep it friendly" in text
    assert "16" not in m.store.person(555).card  # the age itself is never stored


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


async def test_a_later_minor_match_drops_what_was_queued(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=True)
    await match(m)  # score 10: a first message gets queued
    await m.tick()
    await m.on_match("дайвинчик", ["https://t.me/masha_example"], [], "Маша, 16, Москва", 10, minor=True)
    clock.t += 31
    await m.tick()
    person = m.store.person(555)
    assert person.minor and person.score is None and people.sent_texts(555) == []


async def test_a_bio_saying_under_18_flags_a_stranger(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve", strangers="lead")
    people.bio = "мне 16, люблю аниме"
    await stranger_writes(m, people, clock)
    assert m.store.person(777).minor and m.llm.calls == []  # never classified as a date


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


async def test_a_queued_reply_is_dropped_when_the_person_turns_out_to_be_a_minor(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, intimacy=True)
    await match(m)
    people.incoming(555, "привет)")
    m.llm.reply = {"action": "reply", "messages": ["привет, красотка"]}
    await answered(m, people, clock)  # queued for 30 s
    await m.on_match("дайвинчик", ["https://t.me/masha_example"], [], "Маша, 16, Москва", 10, minor=True)
    clock.t += 31
    await m.tick()
    assert people.sent_texts(555) == []


async def test_a_late_minor_flag_cancels_the_answer_being_composed(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, intimacy=True)
    await match(m)
    people.incoming(555, "привет)")

    def answer(messages):
        m.store.update(555, minor=1)  # a later match notice arrives while the model is answering
        return {"action": "reply", "messages": ["ты такая сексуальная"]}
    m.llm.reply = answer
    await answered(m, people, clock)
    m.llm.reply = {"action": "reply", "messages": ["привет) как день?"]}
    for _ in range(3):
        clock.t += 31
        await m.tick()
    assert "ты такая сексуальная" not in people.sent_texts(555)
    assert "under 18" in prompt_of(m)[0]


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


async def test_a_minor_flag_during_typing_stops_the_send(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, intimacy=True)
    await match(m)
    people.incoming(555, "привет)")
    m.llm.reply = {"action": "reply", "messages": ["ты такая сексуальная"]}
    await answered(m, people, clock)  # queued
    real_typing = people.typing

    async def typing_then_flag(user_id, seconds):
        m.store.update(555, minor=1)
        await real_typing(user_id, seconds)
    people.typing = typing_then_flag
    m.llm.reply = {"action": "reply", "messages": ["как прошёл день?"]}
    for _ in range(3):
        clock.t += 31
        await m.tick()
    assert "ты такая сексуальная" not in people.sent_texts(555)


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


async def test_a_later_minor_match_cancels_a_flirt_being_typed(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, intimacy=True)
    await match(m)
    people.incoming(555, "привет)")
    m.llm.reply = {"action": "reply", "messages": ["хочу тебя"]}
    await answered(m, people, clock)  # queued
    real_typing = people.typing

    async def typing_then_minor(user_id, seconds):
        people.typing = real_typing  # once
        await m.on_match("дайвинчик", ["https://t.me/masha_example"], [], "Маша, 16, Москва", 10, minor=True)
        await real_typing(user_id, seconds)
    people.typing = typing_then_minor
    m.llm.reply = {"action": "reply", "messages": ["как прошёл день?"]}
    clock.t += 31
    await m.tick()
    people.typing = real_typing
    for _ in range(3):
        clock.t += 31
        await m.tick()
    assert "хочу тебя" not in people.sent_texts(555) and people.sent_texts(555) == ["как прошёл день?"]


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


async def test_a_minor_learnt_while_another_chat_types_stops_the_queued_flirt(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, intimacy=True)

    async def resolve(links, ids):
        return PersonRef(int(links[0]), None, f"p{links[0]}")
    people.resolve = resolve
    for uid in (1, 2):
        await m.on_match("дайвинчик", [str(uid)], [], "", 10)
        people.incoming(uid, "привет)")
        m.on_message(uid)
    m.llm.reply = {"action": "reply", "messages": ["хочу тебя"]}
    clock.t += 11
    await m.tick()  # both queued
    real_typing = people.typing

    async def typing(user_id, seconds):
        if user_id == 1:
            people.typing = real_typing
            await m.on_match("дайвинчик", ["2"], [], "Аня, 16, Москва", 10, minor=True)
        await real_typing(user_id, seconds)
    people.typing = typing
    m.llm.reply = {"action": "reply", "messages": ["как день?"]}
    for _ in range(4):
        clock.t += 31
        await m.tick()
    assert "хочу тебя" not in people.sent_texts(2)


async def test_a_late_minor_flag_keeps_the_owners_escalation_card(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False)
    await match(m)
    people.incoming(555, "ты бот?")
    await answered(m, people, clock)
    card = 9000 + len(inbox)
    assert m.store.person(555).state == "paused"
    await m.on_match("дайвинчик", ["https://t.me/masha_example"], [], "Маша, 16, Москва", 10, minor=True)
    await m.answer_card(card, "нет, я живой) просто быстро печатаю")
    assert people.sent_texts(555) == ["нет, я живой) просто быстро печатаю"]


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


async def test_a_network_error_after_a_late_minor_flag_does_not_restore_the_flirt(tmp_path):
    from doppelgram.transport import TransportError
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, intimacy=True)
    await match(m)
    people.incoming(555, "привет)")
    m.llm.reply = {"action": "reply", "messages": ["хочу тебя"]}
    await answered(m, people, clock)  # queued
    real_history = people.history

    async def history_fails(user_id, limit):
        people.history = real_history
        await m.on_match("дайвинчик", ["https://t.me/masha_example"], [], "Маша, 16, Москва", 10, minor=True)
        raise TransportError("network")
    people.history = history_fails
    m.llm.reply = {"action": "reply", "messages": ["как день?"]}
    for _ in range(6):
        clock.t += 61
        await m.tick()
    assert "хочу тебя" not in people.sent_texts(555)


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

    async def send_then_command(user_id, text):
        people.send = real_send
        msg_id = await real_send(user_id, text)
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

    async def send_then_command(user_id, text):
        people.send = real_send
        msg_id = await real_send(user_id, text)
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

    async def crash_on_second(user_id, text):
        sends["n"] += 1
        if sends["n"] == 2:
            raise Crash()  # the process dies while typing the second message
        return await real_send(user_id, text)
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


async def test_an_approved_rest_is_not_sent_after_a_minor_flag(tmp_path):
    m, people, inbox, clock, card = await owner_ok_card(tmp_path, ["привет)", "хочу тебя"])
    real_send = people.send

    async def send_then_minor(user_id, text):
        people.send = real_send
        msg_id = await real_send(user_id, text)
        await m.on_match("дайвинчик", ["https://t.me/masha_example"], [], "Маша, 16, Москва", 10, minor=True)
        return msg_id
    people.send = send_then_minor
    await m.answer_card(card, "ок")
    for _ in range(3):
        clock.t += 31
        await m.tick()
    assert "хочу тебя" not in people.sent_texts(555)


async def test_an_approved_series_survives_a_setting_change_in_the_middle(tmp_path):
    m, people, inbox, clock, card = await owner_ok_card(tmp_path, ["привет)", "как день прошёл?"])
    real_send = people.send

    async def send_then_setting(user_id, text):
        people.send = real_send
        msg_id = await real_send(user_id, text)
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

    def boom(user_id, remaining):
        raise RuntimeError("crash")
    store.trim_outbox = boom
    with pytest.raises(RuntimeError):
        store.commit_send(1, 77, 1.0, 5, ["y"])
    assert store.sent_ids(1) == set() and store.outbox_is(1, 5, ["x", "y"])


async def test_an_error_after_a_minor_flag_does_not_restore_an_approved_flirt(tmp_path):
    from doppelgram.transport import TransportError
    m, people, inbox, clock, card = await owner_ok_card(tmp_path, ["привет)", "хочу тебя"])
    real_send, real_history = people.send, people.history
    state = {"sent": 0}

    async def send_once(user_id, text):
        state["sent"] += 1
        msg_id = await real_send(user_id, text)
        people.send = real_send
        await m.on_match("дайвинчик", ["https://t.me/masha_example"], [], "Маша, 16, Москва", 10, minor=True)

        async def failing(uid, limit):
            people.history = real_history
            raise TransportError("network")
        people.history = failing
        return msg_id
    people.send = send_once
    await m.answer_card(card, "ок")
    for _ in range(4):
        clock.t += 130
        await m.tick()
    assert "хочу тебя" not in people.sent_texts(555)


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


async def test_a_failed_approved_series_is_not_offered_again_after_a_minor_flag(tmp_path):
    from doppelgram.transport import TransportError
    m, people, inbox, clock, card = await owner_ok_card(tmp_path, ["привет)", "хочу тебя"])
    real_send, real_history = people.send, people.history

    async def send_once(user_id, text):
        msg_id = await real_send(user_id, text)
        people.send = real_send
        await m.on_match("дайвинчик", ["https://t.me/masha_example"], [], "Маша, 16, Москва", 10, minor=True)

        async def failing(uid, limit):
            people.history = real_history
            raise TransportError("network")
        people.history = failing
        return msg_id
    people.send = send_once
    await m.answer_card(card, "ок")
    assert "остальное не отправляю" in inbox[-1]
    await m.answer_card(card, "ок")  # the old card is gone
    assert people.sent_texts(555) == ["привет)"]


async def test_a_restored_card_with_the_owners_text_still_sends_it_as_theirs(tmp_path):
    from doppelgram.transport import TransportError
    m, people, inbox, clock, card = await owner_ok_card(tmp_path, ["черновик"])
    real_history = people.history

    async def failing(uid, limit):
        people.history = real_history
        raise TransportError("network")
    people.history = failing
    await m.answer_card(card, "мой текст")  # fails before sending: the card comes back
    assert "Ответьте на карточку ещё раз" in inbox[-1]
    people.incoming(555, "ау?")  # the person writes meanwhile
    await m.answer_card(card, "ок")
    assert people.sent_texts(555) == ["мой текст"]


async def test_a_draft_written_before_a_minor_flag_is_withdrawn(tmp_path):
    m, people, inbox, clock, card = await owner_ok_card(tmp_path, ["хочу тебя"])
    await m.on_match("дайвинчик", ["https://t.me/masha_example"], [], "Маша, 16, Москва", 10, minor=True)
    assert "черновик агента отменён" in " ".join(inbox)
    await m.answer_card(card, "ок")
    assert people.sent_texts(555) == []


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


async def test_an_escalation_draft_written_before_a_minor_flag_is_not_sent(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, intimacy=True)
    await match(m)
    people.incoming(555, "давай встретимся вечером?")
    m.llm.reply = {"action": "reply", "messages": ["давай, хочу тебя"]}
    await answered(m, people, clock)
    card = 9000 + len(inbox)
    clock.t += 5
    await m.on_match("дайвинчик", ["https://t.me/masha_example"], [], "Маша, 16, Москва", 10, minor=True)
    await m.answer_card(card, "ок")
    assert people.sent_texts(555) == [] and "меньше 18" in inbox[-1]


async def test_a_draft_whose_card_appears_after_a_minor_flag_is_not_sent(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="approve", write_first=False, intimacy=True)
    await match(m)
    people.incoming(555, "привет)")
    m.llm.reply = {"action": "reply", "messages": ["хочу тебя"]}
    real_notify = m._notify

    async def notify_then_flag(text, image=None):
        m._notify = real_notify
        clock.t += 1
        await m.on_match("дайвинчик", ["https://t.me/masha_example"], [], "Маша, 16, Москва", 10, minor=True)
        return await real_notify(text, image)
    m._notify = notify_then_flag
    await answered(m, people, clock)
    drafts = [i for i, t in enumerate(inbox) if "Черновик" in t]
    await m.answer_card(9000 + drafts[-1] + 1, "ок")
    assert people.sent_texts(555) == []


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


async def test_a_minor_without_a_known_flag_time_gets_no_old_draft_sent(tmp_path):
    m, people, inbox, clock, card = await owner_ok_card(tmp_path, ["хочу тебя"])
    m.store.update(555, minor=1)  # e.g. flagged by an older version, no time recorded
    await m.answer_card(card, "ок")
    assert people.sent_texts(555) == [] and "меньше 18" in inbox[-1]


async def test_a_minor_flag_keeps_the_owners_queued_text(tmp_path):
    m, people, inbox, clock, card = await owner_ok_card(tmp_path, ["черновик"])
    real_typing = people.typing

    async def typing_then_setting(user_id, seconds):
        people.typing = real_typing
        await m.command("chat_mode", "first:off")
        await real_typing(user_id, seconds)
    people.typing = typing_then_setting
    await m.answer_card(card, "мой текст")
    await m.on_match("дайвинчик", ["https://t.me/masha_example"], [], "Маша, 16, Москва", 10, minor=True)
    for _ in range(2):
        clock.t += 31
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


async def test_the_owners_own_text_survives_a_minor_flag_during_typing(tmp_path):
    m, people, inbox, clock, card = await owner_ok_card(tmp_path, ["черновик"])
    real_typing = people.typing

    async def typing_then_minor(user_id, seconds):
        people.typing = real_typing
        await m.on_match("дайвинчик", ["https://t.me/masha_example"], [], "Маша, 16, Москва", 10, minor=True)
        await real_typing(user_id, seconds)
    people.typing = typing_then_minor
    await m.answer_card(card, "мой текст")
    for _ in range(2):
        clock.t += 31
        await m.tick()
    assert "мой текст" in people.sent_texts(555)


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


async def test_the_owners_queued_text_survives_a_minor_flag_and_a_network_error(tmp_path):
    from doppelgram.transport import TransportError
    m, people, inbox, clock, card = await owner_ok_card(tmp_path, ["черновик"])
    real_typing, real_history = people.typing, people.history

    async def typing_then_minor_and_error(user_id, seconds):
        people.typing = real_typing
        await m.on_match("дайвинчик", ["https://t.me/masha_example"], [], "Маша, 16, Москва", 10, minor=True)

        async def failing(uid, limit):
            people.history = real_history
            raise TransportError("network")
        people.history = failing
        await real_typing(user_id, seconds)
    people.typing = typing_then_minor_and_error
    await m.answer_card(card, "мой текст")
    assert "Ответьте на карточку ещё раз" in inbox[-1]  # not lost: the owner's text waits on its card
    await m.answer_card(card, "ок")
    assert people.sent_texts(555) == ["мой текст"]


async def test_an_approved_agent_rest_is_not_resumed_after_a_minor_flag(tmp_path):
    m, people, inbox, clock, card = await owner_ok_card(tmp_path, ["привет)", "хочу тебя"])
    real_send = people.send

    async def send_then_minor(user_id, text):
        people.send = real_send
        msg_id = await real_send(user_id, text)
        await m.on_match("дайвинчик", ["https://t.me/masha_example"], [], "Маша, 16, Москва", 10, minor=True)
        m.store.unset("redecide:555")  # e.g. a /lead already re-read the chat meanwhile
        return msg_id
    people.send = send_then_minor
    await m.answer_card(card, "ок")
    for _ in range(3):
        clock.t += 31
        await m.tick()
    assert "хочу тебя" not in people.sent_texts(555)


async def test_the_owners_queued_text_survives_a_minor_flag_while_it_is_typed(tmp_path):
    m, people, inbox, clock, card = await owner_ok_card(tmp_path, ["черновик"])
    real_typing = people.typing
    calls = {"n": 0}

    async def typing(user_id, seconds):
        calls["n"] += 1
        if calls["n"] == 1:  # the direct answer is interrupted by a setting: it goes to the queue
            await m.command("chat_mode", "first:off")
        elif calls["n"] == 2:  # while the queue types it, the person turns out to be a minor
            await m.on_match("дайвинчик", ["https://t.me/masha_example"], [], "Маша, 16, Москва", 10, minor=True)
        await real_typing(user_id, seconds)
    people.typing = typing
    await m.answer_card(card, "мой текст")
    for _ in range(3):
        clock.t += 31
        await m.tick()
    assert "мой текст" in people.sent_texts(555)


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
