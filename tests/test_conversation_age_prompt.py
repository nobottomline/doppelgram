"""Age declarations are normalized only in provider-facing copies of peer text."""

from __future__ import annotations

import pytest
from test_conversation import make, match
from test_media_strangers import FakeTranscriber, answered


@pytest.mark.parametrize("age", range(10, 19))
@pytest.mark.parametrize("images", [None, ["data:image/png;base64,AA=="]])
async def test_peer_age_is_18_in_text_and_image_prompts(tmp_path, age, images):
    m, people, inbox, clock = make(tmp_path)
    await match(m)
    original = f"привет, мне {age}, люблю музыку; у меня 16 пластинок"
    people.incoming(555, original)
    people.owner_writes(555, "мне 17, это мой пример")
    history = await people.history(555, 20)

    await m._compose(m.store.person(555), history, images)

    content = m.llm.calls[-1][1]["content"]
    prompt = content if isinstance(content, str) else content[0]["text"]
    assert "привет, мне 18, люблю музыку; у меня 16 пластинок" in prompt
    assert "owner: мне 17, это мой пример" in prompt
    assert "23 года, москва" in prompt  # owner facts are not peer data
    assert history[0].text == original  # never mutate Telegram/history data
    assert people.chats[555][0].text == original


@pytest.mark.parametrize("text,expected", [
    ("Мне 10! МНЕ\u00a017? мне\n16.", "Мне 18! МНЕ\u00a018? мне\n18."),
    ("мне 09, мне 19, мне 23", "мне 09, мне 19, мне 23"),
    ("мне 110 и мне 180", "мне 110 и мне 180"),
    ("мне 16.5 и мне 17,5", "мне 16.5 и мне 17,5"),
    ("мне 16к, ремне 17, мне 016", "мне 16к, ремне 17, мне 016"),
    ("16 пластинок, в 18:00, мне 17 лет", "16 пластинок, в 18:00, мне 18 лет"),
])
async def test_only_complete_peer_age_declarations_change(tmp_path, text, expected):
    m, people, inbox, clock = make(tmp_path)
    await match(m)
    people.incoming(555, text)
    await m._compose(m.store.person(555), await people.history(555, 20))
    assert f"Маша: {expected}" in m.llm.calls[-1][1]["content"]


async def test_profile_notes_and_stranger_classifier_use_the_same_normalization(tmp_path):
    m, people, inbox, clock = make(tmp_path)
    await match(m)
    m.store.update(555, card="мне 16, люблю музыку", memory="сказала: мне 17")
    await m._compose(m.store.person(555), [])
    prompt = m.llm.calls[-1][1]["content"]
    assert "Their dating profile: мне 18, люблю музыку" in prompt
    assert "сказала: мне 18" in prompt

    await m._classify_stranger("привет, мне 16", "мне 17, люблю музыку")
    prompt = m.llm.calls[-1][1]["content"]
    assert "Bio: мне 18, люблю музыку" in prompt
    assert "привет, мне 18" in prompt
    assert "16" not in prompt and "17" not in prompt


@pytest.mark.parametrize("age", range(10, 18))
async def test_new_minor_declaration_still_escalates_before_the_provider(tmp_path, age):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, intimacy=True)
    await match(m)
    original = f"мне {age}"
    people.incoming(555, original)
    await answered(m, people, clock)
    assert m.llm.calls == [] and people.sent_texts(555) == []
    assert "несовершеннолетний" in inbox[-1] and original in inbox[-1]


@pytest.mark.parametrize("age", [16, 18])
async def test_transcripts_keep_the_raw_minor_gate_before_prompt_normalization(tmp_path, age):
    m, people, inbox, clock = make(tmp_path, write_first=False)
    await match(m)
    m.transcriber = FakeTranscriber(f"мне {age}, люблю музыку")
    people.incoming(555, "", "voice", duration=2, blob=b"ogg")
    await answered(m, people, clock)
    if age < 18:
        assert m.llm.calls == [] and "несовершеннолетний" in inbox[-1]
    else:
        assert "мне 18, люблю музыку" in m.llm.calls[-1][1]["content"]
    assert people.sent_texts(555) == []


async def test_minor_flag_survives_owner_resume_and_history_truncation(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, intimacy=True, history_messages=1)
    await match(m)
    people.incoming(555, "мне 16")
    await answered(m, people, clock)
    assert m.store.person(555).minor
    await m.answer_card(9000 + len(inbox), "пропуск")
    people.incoming(555, "привет")
    await answered(m, people, clock)
    assert "Intimacy is allowed" not in m.llm.calls[-1][0]["content"]
    assert "under 18: keep it friendly" in m.llm.calls[-1][1]["content"]


@pytest.mark.parametrize("first", ["давай встретимся", "может созвонимся?"])
async def test_minor_declaration_takes_priority_over_meeting_drafts_in_a_burst(tmp_path, first):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False, intimacy=True)
    await match(m)
    people.incoming(555, first)
    people.incoming(555, "мне 16")
    await answered(m, people, clock)
    assert m.store.person(555).minor
    assert m.llm.calls == [] and people.sent_texts(555) == []
    assert "несовершеннолетний" in inbox[-1]


async def test_preexisting_minor_history_is_remembered_before_normalization(tmp_path):
    m, people, inbox, clock = make(tmp_path, write_first=False, intimacy=True)
    await match(m)
    old = people.incoming(555, "мне 16")
    m.store.update(555, handled_upto=old)
    people.incoming(555, "привет")
    await answered(m, people, clock)
    assert m.store.person(555).minor
    assert "Intimacy is allowed" not in m.llm.calls[-1][0]["content"]
    prompt = m.llm.calls[-1][1]["content"]
    assert "мне 18" in prompt and "under 18: keep it friendly" in prompt
