"""Transport-neutral data model shared by the Telegram adapter and the agent core."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

CLICKABLE_KINDS = frozenset({"callback", "reply"})


@dataclass(frozen=True)
class Button:
    """One keyboard button. ``data`` never leaves the executor; the model sees only ``id``."""

    id: str
    row: int
    col: int
    text: str
    kind: str  # callback | reply | url | webview | switch_inline | game | other
    data: bytes | None = None
    requires_password: bool = False
    url: str | None = None
    blocked: bool = False  # matched the bot's deny list (paid features, reports, message traps)
    inline: bool = True  # inline keyboard (attached to a message) vs. reply keyboard (replaces the input)

    @property
    def clickable(self) -> bool:
        return self.kind in CLICKABLE_KINDS and not self.requires_password and not self.blocked


@dataclass(frozen=True)
class BotMessage:
    id: int
    text: str
    buttons: tuple[Button, ...] = ()
    media_kind: str | None = None  # photo | image | sticker | video | other
    media_key: str = ""  # stable identifier of the attached media
    date: float = 0.0
    edit_date: float | None = None
    outgoing: bool = False
    keyboard_hidden: bool = False  # the message removes the reply keyboard
    links: tuple[str, ...] = ()  # URLs and @mentions found in the text
    mention_ids: tuple[int, ...] = ()  # users mentioned by id (no public username)

    @property
    def has_image(self) -> bool:
        return self.media_kind in {"photo", "image", "sticker", "video"}

    @property
    def clickable_buttons(self) -> tuple[Button, ...]:
        return tuple(b for b in self.buttons if b.clickable)

    @property
    def revision(self) -> str:
        """Content hash; bots usually edit a card in place, so the id alone is not enough."""
        h = hashlib.sha256()
        h.update(self.text.encode())
        h.update(b"\x00" + self.media_key.encode())
        for b in self.buttons:
            h.update(b"\x00%s|%s|" % (b.kind.encode(), b.text.encode()))
            h.update(b.data or b"")
        return h.hexdigest()[:16]

    def button(self, button_id: str) -> Button | None:
        return next((b for b in self.buttons if b.id == button_id), None)


@dataclass(frozen=True)
class ClickOutcome:
    ok: bool
    answer: str | None = None  # callback answer text (toast/alert), if any
    alert: bool = False
    timed_out: bool = False


@dataclass(frozen=True)
class OwnerMessage:
    id: int
    chat_id: int
    text: str
    reply_to: int | None = None
    image: bytes | None = None  # a photo the owner sent (taste reference)
    has_media: bool = False  # any attachment, even one that could not be used as a reference


@dataclass
class Decision:
    action: str  # click | send | wait | ask_owner
    button_id: str | None = None
    text: str | None = None
    wait_seconds: float | None = None
    reason: str = ""
    confidence: float | None = None
    description: str = ""          # free text for the owner's card only; never stored
    traits: dict = field(default_factory=dict)  # fixed-vocabulary visual traits; what examples keep
    raw: dict = field(default_factory=dict)
