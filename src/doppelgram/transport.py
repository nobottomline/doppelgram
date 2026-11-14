"""The boundary between the agent core and Telegram. The real implementation lives in telegram.py."""

from __future__ import annotations

from typing import Protocol

from .models import BotMessage, Button, ClickOutcome


class TransportError(RuntimeError):
    pass


class RateLimited(TransportError):
    """Telegram asked us to wait (FLOOD_WAIT)."""

    def __init__(self, seconds: float):
        super().__init__(f"rate limited for {seconds:.0f}s")
        self.seconds = seconds


class AccountRestricted(TransportError):
    """The account is frozen, banned, or the session was revoked: stop all writes."""


class Transport(Protocol):
    async def recent_bot_messages(self, limit: int) -> list[BotMessage]:
        """Latest messages in the bot chat, oldest first, both directions."""

    async def download_image(self, msg_id: int) -> bytes | None:
        """Image bytes (photo, image document, sticker, or a video thumbnail)."""

    async def click(self, msg_id: int, button: Button) -> ClickOutcome: ...

    async def send_to_bot(self, text: str) -> None: ...

    async def notify_owner(self, text: str, image: bytes | None = None) -> int | None:
        """Send a message to the owner control chat; returns its message id."""
