"""Age declarations are normalized only in provider-facing copies of peer text."""

from __future__ import annotations

import pytest
from test_conversation import make, match


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
