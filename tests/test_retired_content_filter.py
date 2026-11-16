"""Content labels no longer route a conversation to the owner by themselves."""

import pytest
from test_conversation import make, match
from test_media_strangers import FakeTranscriber, answered

from doppelgram.conversation import ESCALATIONS, system_prompt


@pytest.mark.parametrize("mode", ["auto", "approve"])
@pytest.mark.parametrize("direction", ["incoming", "outgoing"])
async def test_former_content_terms_use_the_normal_reply_flow(tmp_path, mode, direction):
    m, people, inbox, clock = make(tmp_path, mode=mode, write_first=False)
    await match(m)
    incoming = "что означает слово секстинг?" if direction == "incoming" else "привет"
    response = "об этом можно поговорить" if direction == "incoming" else "слово «секстинг» я слышал"
    people.incoming(555, incoming)
    m.llm.reply = {"action": "reply", "messages": [response]}
    await answered(m, people, clock)
    assert len(m.llm.calls) == 1 and m.store.person(555).state == "active"
    if mode == "approve":
        assert people.sent_texts(555) == []
        card = m.store.open_card(555)
        assert card["kind"] == "draft"
        await m.answer_card(card["control_msg_id"], "ок")
    else:
        assert m.store.open_card(555) is None
        clock.t += 31
        await m.tick()
    assert people.sent_texts(555) == [response]


async def test_a_transcript_uses_the_same_reply_flow(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False)
    await match(m)
    m.transcriber = FakeTranscriber("что означает слово секстинг?")
    people.incoming(555, "", "voice", duration=2, blob=b"ogg")
    m.llm.reply = {"action": "reply", "messages": ["об этом можно поговорить"]}
    await answered(m, people, clock)
    assert len(m.llm.calls) == 1 and m.store.person(555).state == "active"
    clock.t += 31
    await m.tick()
    assert people.sent_texts(555) == ["об этом можно поговорить"]


@pytest.mark.parametrize("lookup", [False, True])
def test_prompt_and_schema_do_not_offer_the_retired_content_category(lookup):
    prompt = system_prompt(lookup=lookup)
    assert "explicit" not in ESCALATIONS
    assert 'category "explicit"' not in prompt
    assert "|explicit|" not in prompt
    assert "No graphic sexual descriptions" not in prompt
