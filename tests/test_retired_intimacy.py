"""Intimacy is part of conversation behavior, without a configurable permission flag."""

from pathlib import Path

import pytest
from test_conversation import make, match
from test_media_strangers import answered

from doppelgram.config import _conversations
from doppelgram.control import help_text, parse_owner
from doppelgram.conversation import ConversationStore, PersonRef


@pytest.mark.parametrize("legacy_value", [False, True])
def test_old_intimacy_config_is_ignored(legacy_value):
    cfg = _conversations(Path("."), {"intimacy": legacy_value})
    assert not hasattr(cfg, "intimacy")


@pytest.mark.parametrize("value", ["0", "1"])
def test_database_upgrade_removes_only_the_retired_toggle(tmp_path, value):
    path = tmp_path / "conversation.sqlite3"
    store = ConversationStore(path)
    store.add_person(PersonRef(555, "example", "Example"), "bot", "любит музыку", 100, score=9)
    store.update(555, state="paused", handled_upto=5)
    store.set_outbox(555, 5, ["owner message"], 300, approved=True, owner_text=True)
    store.add_card(999, 555, "escalation", 5, ["draft"], 100)
    store.set("mode", "approve")
    store.set("intimacy", value)
    person, card, outbox = store.person(555), store.open_card(555), store.due_outbox(400)
    store.db.close()
    for _ in range(2):
        store = ConversationStore(path)
        assert store.get("intimacy") is None
        assert store.get("mode") == "approve"
        assert store.person(555) == person and store.open_card(555) == card
        assert store.due_outbox(400) == outbox
        store.db.close()


@pytest.mark.parametrize("mode", ["approve", "auto"])
async def test_stale_off_flag_cannot_disable_ordinary_intimacy(tmp_path, mode):
    m, people, inbox, clock = make(tmp_path, mode=mode, write_first=False)
    await match(m)
    m.store.set("intimacy", "0")  # ignored even if an old process wrote it after the migration
    people.incoming(555, "мне интересны отношения и интим")
    m.llm.reply = {"action": "reply", "messages": ["давай узнаем друг друга"]}
    await answered(m, people, clock)
    assert len(m.llm.calls) == 1
    assert 'Sexual or explicit topics: escalate' not in m.llm.calls[0][0]["content"]
    if mode == "approve":
        assert people.sent_texts(555) == []
        await m.answer_card(m.store.open_card(555)["control_msg_id"], "ок")
    else:
        clock.t += 31
        await m.tick()
    assert people.sent_texts(555) == ["давай узнаем друг друга"]


@pytest.mark.parametrize("command", ["/chat интим вкл", "/chat интим выкл", "/chat intimacy on", "/chat intimacy off"])
def test_removed_toggle_has_no_command_or_help_entry(command):
    assert parse_owner(command).name == "unknown"
    help_message = str(help_text([]))
    assert "intimacy" not in help_message and "интим" not in help_message


async def test_a_retired_internal_command_cannot_corrupt_the_conversation_mode(tmp_path):
    m, people, inbox, clock = make(tmp_path, mode="auto")
    await m.command("chat_mode", "intimacy:off")
    assert m.mode == "auto" and m.store.get("intimacy") is None
    assert "Неизвестный режим" in inbox[-1]
