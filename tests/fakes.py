"""Test doubles: a simulated rating bot and a scripted LLM."""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from doppelgram.config import BehaviorConfig, Config, LLMConfig, TelegramConfig
from doppelgram.llm import LLMError, LLMResult
from doppelgram.models import BotMessage, Button, ClickOutcome


def png_bytes(color=(200, 50, 50)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(buf, format="PNG")
    return buf.getvalue()


def make_config(tmp: Path, **behavior) -> Config:
    (tmp / "task.md").write_text("Rate each drawing 1-5.", encoding="utf-8")
    (tmp / "taste.md").write_text("Likes clean lineart.", encoding="utf-8")
    b = dict(task_file=tmp / "task.md", taste_file=tmp / "taste.md", mode="auto", autostart=True,
             settle_s=0, action_delay_min_s=0, action_delay_max_s=0, report_every_h=0,
             active_hours_enabled=False)  # tests that need working hours turn them on
    b.update(behavior)
    return Config(
        telegram=TelegramConfig(api_id=1, api_hash="x", session=tmp / "s"),
        targets=[], owner_ids=[42], allow_saved_messages=True,
        llm=LLMConfig(base_url="http://llm.test/v1", api_key="k", model="m"),
        behavior=BehaviorConfig(**b), data_dir=tmp,
    )


class Clock:
    def __init__(self, t: float = 1_800_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    async def sleep(self, s: float) -> None:
        self.t += s


@dataclass
class FakeRatingBot:
    """Shows N cards in ONE message, editing it in place after every rating (like most such bots)."""

    cards: int = 3
    separate_keyboard: bool = False
    messages: list[BotMessage] = field(default_factory=list)
    ratings: list[str] = field(default_factory=list)
    owner_inbox: list[tuple[str, bool]] = field(default_factory=list)
    sent: list[str] = field(default_factory=list)
    shown: int = 0
    next_id: int = 100
    timeout_clicks: int = 0

    def _rating_buttons(self) -> tuple[Button, ...]:
        rates = [Button(id=f"r0c{i}", row=0, col=i, text=str(i + 1), kind="callback",
                        data=f"rate:{self.shown}:{i + 1}".encode()) for i in range(5)]
        return (*rates, Button(id="r1c0", row=1, col=0, text="Сайт", kind="url", url="https://x"))

    def start(self) -> None:
        self.shown = 1
        if self.separate_keyboard:
            self._add(BotMessage(id=self._id(), text="", media_kind="photo", media_key="p1", date=0))
            self._add(BotMessage(id=self._id(), text="Оцени", buttons=self._rating_buttons(), date=1))
        else:
            self._add(BotMessage(id=self._id(), text="Карточка 1", buttons=self._rating_buttons(),
                                 media_kind="photo", media_key="p1", date=0))

    def _id(self) -> int:
        self.next_id += 1
        return self.next_id

    def _add(self, m: BotMessage) -> None:
        self.messages.append(m)

    async def recent_bot_messages(self, limit: int) -> list[BotMessage]:
        await asyncio.sleep(0)  # a real network call yields to other tasks
        return self.messages[-limit:]

    async def download_image(self, msg_id: int) -> bytes | None:
        m = next(m for m in self.messages if m.id == msg_id)
        return png_bytes() if m.has_image else None

    async def click(self, msg_id: int, button: Button) -> ClickOutcome:
        if self.timeout_clicks:
            self.timeout_clicks -= 1
            return ClickOutcome(ok=True, timed_out=True)
        last = self.messages[-1]
        assert last.id == msg_id, "clicked a stale message"
        assert button.data and button.data.startswith(f"rate:{self.shown}:".encode())
        self.ratings.append(button.text)
        if self.shown < self.cards:
            self.shown += 1
            edited = BotMessage(id=last.id, text=f"Карточка {self.shown}", buttons=self._rating_buttons(),
                                media_kind="photo", media_key=f"p{self.shown}", date=last.date, edit_date=1)
            self.messages[-1] = edited
        else:
            self.messages[-1] = BotMessage(id=last.id, text="Анкеты закончились. Приходи через 3 часа.",
                                           date=last.date, edit_date=2)
        return ClickOutcome(ok=True, answer="Оценка принята")

    async def send_to_bot(self, text: str) -> None:
        self.sent.append(text)
        self._add(BotMessage(id=self._id(), text=text, outgoing=True))
        if text == "/start" and not self.shown:
            self.start()

    async def notify_owner(self, text: str, image: bytes | None = None) -> int | None:
        self.owner_inbox.append((text, image is not None))
        return 5000 + len(self.owner_inbox)


class ScriptedLLM:
    """Returns decisions from a function of the prompt text; records calls."""

    def __init__(self, decide):
        self.decide = decide
        self.calls: list[list[dict]] = []
        self.fail_next = 0

    async def complete_json(self, messages: list[dict], failures: list | None = None) -> LLMResult:
        self.calls.append(messages)
        if self.fail_next:
            self.fail_next -= 1
            raise LLMError("boom")
        text = messages[1]["content"][0]["text"]
        has_image = len(messages[1]["content"]) > 1
        return LLMResult(data=self.decide(text, has_image), model="m", prompt_tokens=1000,
                         completion_tokens=100, cost_usd=0.001)

    async def aclose(self) -> None:
        pass


def rate_or_wait(text: str, has_image: bool) -> dict:
    if "Приходи через" in text:
        return {"action": "wait", "wait_seconds": 3 * 3600, "reason": "лимит"}
    if has_image:
        return {"action": "click", "button_id": "r0c3", "confidence": 0.7, "reason": "аккуратные линии",
                "description": "девушка с мечом", "traits": {"фото": "портрет", "лицо": "видно"}}
    return {"action": "ask_owner", "reason": "непонятно"}
