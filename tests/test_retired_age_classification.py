"""Retiring age classification preserves owner decisions and ordinary message processing."""

import pytest
from test_conversation import make, match
from test_media_strangers import FakeTranscriber, answered, stranger_writes

from doppelgram.config import load_config
from doppelgram.conversation import ConversationStore, PersonRef, classify_incoming


@pytest.mark.parametrize("text", ["мне 16", "я 17", "я в 11 классе", "учусь в девятом классе"])
def test_age_and_school_words_do_not_trigger_a_category(text):
    assert classify_incoming(text, None) is None


@pytest.mark.parametrize("voice", [False, True])
async def test_age_text_is_processed_as_an_ordinary_message(tmp_path, voice):
    m, people, inbox, clock = make(tmp_path, mode="auto", write_first=False)
    await match(m)
    text = "мне 16, люблю музыку"
    if voice:
        m.transcriber = FakeTranscriber(text)
        people.incoming(555, "", "voice", duration=2, blob=b"ogg")
    else:
        people.incoming(555, text)
    m.llm.reply = {"action": "reply", "messages": ["какую музыку слушаешь?"]}
    await answered(m, people, clock)
    assert len(m.llm.calls) == 1
    assert "мне 18, люблю музыку" in m.llm.calls[0][1]["content"]
    assert m.store.person(555).state == "active" and m.store.open_card(555) is None
    clock.t += 31
    await m.tick()
    assert people.sent_texts(555) == ["какую музыку слушаешь?"]


async def test_a_stranger_bio_does_not_bypass_the_normal_classifier(tmp_path):
    m, people, inbox, clock = make(tmp_path, write_first=False, strangers="lead")
    people.bio = "мне 16, люблю музыку"
    m.llm.reply = {"kind": "dating", "reason": "знакомится"}
    await stranger_writes(m, people, clock)
    assert len(m.llm.calls) == 1
    assert "Bio: мне 18, люблю музыку" in m.llm.calls[0][1]["content"]
    assert m.store.person(777).state == "active"


@pytest.mark.parametrize("state", ["active", "paused", "owner", "blocked"])
def test_legacy_flags_are_removed_without_changing_chat_state_or_work(tmp_path, state):
    path = tmp_path / "conversation.sqlite3"
    store = ConversationStore(path)
    store.add_person(PersonRef(555, "example", "Example"), "bot", "любит музыку", 100, score=9)
    store.update(555, state=state, due_at=200, handled_upto=5, owner_upto=6, memory="любит джаз")
    store.set_outbox(555, 5, ["owner message"], 300, approved=True, owner_text=True)
    store.add_card(999, 555, "escalation", 5, ["draft"], 100)
    store.set("mode", "approve")
    store.set("minor_at:555", "123")
    store.set("minorXat:unrelated", "keep")
    store.db.execute("ALTER TABLE people ADD COLUMN minor INTEGER NOT NULL DEFAULT 0")
    store.db.execute("UPDATE people SET minor=1 WHERE user_id=555")
    person = store.person(555)
    cards = store.db.execute("SELECT * FROM cards").fetchall()
    outbox = store.due_outbox(400)
    store.db.close()

    for _ in range(2):  # restart and idempotence
        store = ConversationStore(path)
        assert "minor" not in {r[1] for r in store.db.execute("PRAGMA table_info(people)")}
        assert store.get("minor_at:555") is None
        assert store.get("minorXat:unrelated") == "keep"
        assert store.get("mode") == "approve"
        assert store.person(555) == person
        assert store.db.execute("SELECT * FROM cards").fetchall() == cards
        assert store.due_outbox(400) == outbox
        store.db.close()


def test_old_age_gate_options_are_ignored_in_defaults_and_bot_overrides(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('''
[behavior]
min_age = 18
skip_buttons = ["👎"]
age_in_prompt = "18"
[[bots]]
name = "example"
username = "@example_bot"
min_age = 21
skip_buttons = ["skip"]
age_pattern = '(\\d{1,2}) лет'
''', encoding="utf-8")
    cfg = load_config(path, require_telegram=False, require_llm=False)
    assert cfg.behavior.age_in_prompt == cfg.targets[0].behavior.age_in_prompt == "18"
    assert cfg.targets[0].behavior.age_pattern == r"(\d{1,2}) лет"
    assert not hasattr(cfg.behavior, "min_age") and not hasattr(cfg.targets[0].behavior, "skip_buttons")
