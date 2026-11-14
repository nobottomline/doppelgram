"""Simulated dating bots, modelled on the publicly documented Leomatch and Bibinto protocols."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace

from fakes import png_bytes

from doppelgram.models import BotMessage, Button, ClickOutcome


def reply_kb(rows: list[list[str]]) -> tuple[Button, ...]:
    return tuple(Button(id=f"r{r}c{c}", row=r, col=c, text=t, kind="reply", inline=False)
                 for r, row in enumerate(rows) for c, t in enumerate(row))


@dataclass
class _Base:
    messages: list[BotMessage] = field(default_factory=list)
    owner_inbox: list[tuple[str, bool]] = field(default_factory=list)
    sent: list[str] = field(default_factory=list)
    next_id: int = 100

    def _id(self) -> int:
        self.next_id += 1
        return self.next_id

    async def recent_bot_messages(self, limit: int) -> list[BotMessage]:
        await asyncio.sleep(0)
        return self.messages[-limit:]

    async def download_image(self, msg_id: int) -> bytes | None:
        m = next(m for m in self.messages if m.id == msg_id)
        return png_bytes() if m.has_image else None

    async def notify_owner(self, text: str, image: bytes | None = None) -> int | None:
        self.owner_inbox.append((text, image is not None))
        return 7000 + len(self.owner_inbox)

    async def click(self, msg_id: int, button: Button) -> ClickOutcome:
        assert button.kind == "reply", "these bots only use reply keyboards here"
        await self.send_to_bot(button.text)
        return ClickOutcome(ok=True)


LEO_KB = [["❤️", "💌 / 📹", "👎", "💤"]]


@dataclass
class FakeLeomatch(_Base):
    """Cards are separate messages; the reply keyboard comes with each card. Liking card #2 is a match."""

    cards: list[tuple[str, int]] = field(default_factory=lambda: [("Аня", 21), ("Маша", 23), ("Лиза", 17),
                                                                  ("Катя", 25)])
    shown: int = 0
    reactions: list[tuple[str, str]] = field(default_factory=list)  # (name, button)
    ad_after: int = 1  # an ad post arrives after this many reactions

    def _card(self) -> None:
        name, age = self.cards[self.shown]
        self.messages.append(BotMessage(id=self._id(), text=f"{name}, {age}, Москва – люблю горы",
                                        buttons=reply_kb(LEO_KB), media_kind="photo",
                                        media_key=f"leo-{name}", date=self.next_id))

    async def send_to_bot(self, text: str) -> None:
        self.sent.append(text)
        self.messages.append(BotMessage(id=self._id(), text=text, outgoing=True, date=self.next_id))
        if text == "/start":
            self._card()
            return
        if text not in ("❤️", "👎"):
            return
        name, _ = self.cards[self.shown]
        self.reactions.append((name, text))
        if text == "❤️" and name == "Маша":
            self.messages.append(BotMessage(
                id=self._id(), text="Есть взаимная симпатия! Начинай общаться 👉 Маша",
                links=("https://t.me/masha_example",), date=self.next_id))
        if len(self.reactions) == self.ad_after:
            self.messages.append(BotMessage(id=self._id(), text="Реклама: лучший канал про путешествия",
                                            date=self.next_id))
        self.shown += 1
        if self.shown < len(self.cards):
            self._card()
        else:
            self.messages.append(BotMessage(id=self._id(), text="Анкеты закончились, приходи позже",
                                            buttons=reply_kb([["1", "2", "3"]]), date=self.next_id))


BIB_KB = [["1", "2", "3", "4", "5"], ["6", "7", "8", "9", "10"],
          ["💘 Подарить валентинку", "Назад"], ["💌Сообщение", "⚠️Жалоба"]]


@dataclass
class FakeBibinto(_Base):
    """The 🐨 message carries the keyboard, the picture comes after it; rating edits the old picture."""

    people: list[str] = field(default_factory=lambda: ["Аня", "Оля", "Вика"])
    shown: int = 0
    ratings: list[tuple[str, str]] = field(default_factory=list)

    def _screen(self) -> None:
        self.messages.append(BotMessage(id=self._id(), text="🐨", buttons=reply_kb(BIB_KB), date=self.next_id))
        name = self.people[self.shown]
        self.messages.append(BotMessage(
            id=self._id(), text=f"👩‍🦰: {name}, 22 лет (✅)\n🌇Город: Москва\n✍️: привет", media_kind="photo",
            media_key=f"bib-{name}", date=self.next_id))

    async def send_to_bot(self, text: str) -> None:
        self.sent.append(text)
        self.messages.append(BotMessage(id=self._id(), text=text, outgoing=True, date=self.next_id))
        if text == "/start":
            self._screen()
            return
        if not text.isdigit():
            return
        if self.shown >= len(self.people):  # a rating with no card on screen: record it, it is a bug
            self.ratings.append(("?", text))
            return
        self.ratings.append((self.people[self.shown], text))
        # The rated picture loses its caption and gets an inline "back" button (seen via MessageEdited).
        idx = max(i for i, m in enumerate(self.messages) if m.has_image and not m.outgoing)
        self.messages[idx] = replace(self.messages[idx], text="", edit_date=1.0, buttons=(
            Button(id="r0c0", row=0, col=0, text="↩️", kind="callback", data=b"top_place:back_rate:1"),))
        self.shown += 1
        if self.shown < len(self.people):
            self._screen()
        else:
            self.messages.append(BotMessage(id=self._id(), text="🐨 Коала накушалась твоих оценок!",
                                            date=self.next_id))
