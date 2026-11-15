"""What people send besides text (stickers, photos, voice) and people who write first without a match."""

from __future__ import annotations

import io

import pytest
from test_conversation import ChatMessage, ConversationManager, PersonRef, make, match

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
