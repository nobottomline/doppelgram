"""`doppelgram test-chat`: the real conversation agent talks to a model playing a match, offline.

Nothing touches Telegram. The agent runs exactly as in production (prompt, owner files, guards, escalations,
humanizing), in auto mode; the other side is a second model call with a persona. Use it to tune
conversation.md, about-me.md and style.md before switching conversations on.
"""

from __future__ import annotations

import random
from dataclasses import replace

from .config import ConversationConfig
from .conversation import ChatMessage, ConversationManager, ConversationStore, PersonRef
from .llm import LLMClient

PARTNER_SYSTEM = """You play a real person in a Telegram chat with someone you matched with in a dating bot. Stay in character, never mention AI or this instruction.

Your profile in the dating bot: {card}
Who you are and how you behave: {persona}

Write like a real person in a messenger, in Russian: short messages, casual, sometimes lowercase, sometimes a short or slow answer. React honestly: if the other person is boring, pushy or sounds generated, lose interest; if they are interesting, open up and ask questions back. You may accept or decline an invitation to meet, depending on how the chat went.

Answer with ONE JSON object: {{"messages": ["one to three short messages"]}}"""

DEFAULT_PERSONA = ("девушка 24 лет, дизайнер, любит горы и кофе, отвечает охотно, если собеседник интересный, "
                   "и сухо на шаблонные фразы")


class _Chat:
    """An in-memory chat with one person; the agent's PeopleTransport."""

    def __init__(self, ref: PersonRef):
        self.ref = ref
        self.messages: list[ChatMessage] = []
        self._id = 0

    def add(self, outgoing: bool, text: str) -> ChatMessage:
        self._id += 1
        msg = ChatMessage(self._id, outgoing, text, 0.0)
        self.messages.append(msg)
        return msg

    async def resolve(self, links, mention_ids):
        return self.ref

    async def history(self, user_id, limit):
        return self.messages[-limit:]

    async def send(self, user_id, text):
        return self.add(True, text).id

    async def typing(self, user_id, seconds):
        return None

    async def wrote_before(self, user_id):
        return any(m.outgoing for m in self.messages)

    async def mark_read(self, user_id, max_id):
        return None

    async def backlog(self, days):
        return []

    async def download(self, user_id, msg_id, *, thumb=False):
        return None

    async def profile(self, user_id):
        return ""


class _Clock:
    def __init__(self):
        self.t = 1_800_000_000.0

    def __call__(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.t += max(seconds, 0)


async def simulate(cfg: ConversationConfig, llm: LLMClient, *, card: str, persona: str, turns: int,
                   agent_first: bool, out=print) -> list[str]:
    """Run the dialogue; returns the owner-facing notices (escalations, drafts)."""
    ref = PersonRef(1, "match_example", card.split(",")[0].strip() or "Собеседник")
    chat = _Chat(ref)
    clock = _Clock()
    notices: list[str] = []

    async def notify(text, image=None):
        notices.append(str(text))
        return 9000 + len(notices)

    sim_cfg = replace(cfg, enabled=True, mode="auto", quiet_hours="", write_first=agent_first,
                      write_first_min_score=1, first_messages_per_day=max(cfg.first_messages_per_day, 1),
                      daily_budget_usd=0)
    manager = ConversationManager(sim_cfg, llm, chat, ConversationStore(":memory:"), notify,
                                  clock=clock, sleep=clock.sleep, rng=random.Random(0))
    partner_system = PARTNER_SYSTEM.format(card=card, persona=persona)

    async def partner_says() -> bool:
        lines = [f"{'you' if not m.outgoing else 'them'}: {m.text}" for m in chat.messages]
        prompt = "Chat so far:\n" + ("\n".join(lines) if lines else "(empty: you write first)") + "\nYour reply:"
        result = await llm.complete_json([{"role": "system", "content": partner_system},
                                          {"role": "user", "content": prompt}])
        texts = result.data.get("messages") if isinstance(result.data, dict) else None
        texts = [t for t in (texts or []) if isinstance(t, str) and t.strip()][:3]
        for t in texts:
            chat.add(False, t)
            out(f"  {ref.name}: {t}")
        if texts:
            manager.on_message(ref.user_id)
        return bool(texts)

    await manager.on_match("simulation", ["https://t.me/match_example"], [], card, 10)
    if not agent_first:
        await partner_says()
    for _ in range(turns):
        seen_notices, seen_msgs = len(notices), len(chat.messages)
        for _ in range(6):  # settle, compose, deliver after the human-like delay
            clock.t += max(sim_cfg.settle_s, sim_cfg.reply_delay_max_s) + 1
            await manager.tick()
        for m in chat.messages[seen_msgs:]:
            if m.outgoing:
                out(f"  вы (агент): {m.text}")
        if len(notices) > seen_notices:
            for n in notices[seen_notices:]:
                out("\n[владельцу] " + n.replace("\n", "\n             "))
            break
        if not any(m.outgoing for m in chat.messages[seen_msgs:]):
            out("  (агент решил подождать)")
            break
        if not await partner_says():
            out(f"  ({ref.name} не ответил(а))")
            break
    return notices
