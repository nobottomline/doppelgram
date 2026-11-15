"""Telethon (MTProto user account) implementation of the Transport boundary."""

from __future__ import annotations

import asyncio
import contextlib
import html
import io
import logging
import re
from typing import TYPE_CHECKING

from telethon import TelegramClient, events, functions, helpers
from telethon import errors as tg_errors

from .config import Config, Target
from .media import to_jpeg
from .models import BotMessage, Button, ClickOutcome, OwnerMessage
from .transport import AccountRestricted, RateLimited, TransportError
from .ui import as_html

if TYPE_CHECKING:
    from .supervisor import Supervisor

log = logging.getLogger(__name__)

MAX_MEDIA_BYTES = 20 * 1024 * 1024
CAPTION_LIMIT = 1024
MESSAGE_LIMIT = 4000


def _chunks(body: str, limit: int) -> list[str]:
    """Split long HTML at line breaks (tags never span lines in our templates). A single line longer than
    the limit is sent as plain text in pieces, so no piece ever ends inside a tag or an entity."""
    if len(body) <= limit:
        return [body]
    lines: list[str] = []
    for line in body.split("\n"):
        if len(line) <= limit:
            lines.append(line)
            continue
        plain = html.unescape(re.sub(r"<[^>]*>", "", line))
        step = limit // 6  # escaping can grow a character up to 5 times ("&" -> "&amp;")
        lines += [html.escape(plain[k:k + step], quote=False) for k in range(0, len(plain), step)]
    out, cur = [], ""
    for line in lines:
        if cur and len(cur) + 1 + len(line) > limit:
            out.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        out.append(cur)
    return out


OWNER_IMAGE_BYTES = 10 * 1024 * 1024

_RESTRICTED_ERRORS = tuple(
    getattr(tg_errors, name)
    for name in ("FrozenMethodInvalidError", "UserDeactivatedError", "UserDeactivatedBanError",
                 "AuthKeyUnregisteredError", "SessionRevokedError", "AuthKeyDuplicatedError",
                 "PeerFloodError")
    if hasattr(tg_errors, name)
)


def make_client(cfg: Config) -> TelegramClient:
    t = cfg.telegram
    t.session.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    return TelegramClient(
        str(t.session), t.api_id, t.api_hash,
        device_model=t.device_model, system_version=t.system_version, app_version=t.app_version,
        lang_code=t.lang_code, system_lang_code=t.lang_code,
        flood_sleep_threshold=20, connection_retries=10, retry_delay=5, auto_reconnect=True,
    )


# --------------------------------------------------------------- normalization
def _button_kind(tname: str, inline: bool) -> str:
    if "Callback" in tname:
        return "callback"
    if "Url" in tname and "Auth" not in tname:
        return "url"
    if "WebView" in tname:
        return "webview"
    if "SwitchInline" in tname:
        return "switch_inline"
    if "Game" in tname:
        return "game"
    if not inline and tname in ("ButtonTypeDefault", "KeyboardButton"):
        return "reply"  # plain reply-keyboard button: pressing it sends its text
    return "other"  # phone/geo/peer/poll requests, buy, copy, ... are never auto-clicked


def parse_markup(markup) -> tuple[Button, ...]:
    if markup is None:
        return ()
    inline = type(markup).__name__ == "ReplyInlineMarkup"
    buttons: list[Button] = []
    for r, row in enumerate(getattr(markup, "rows", None) or []):
        for c, btn in enumerate(getattr(row, "buttons", None) or []):
            # Current layers wrap the behaviour in `btn.type`; older layers used distinct button classes.
            spec = getattr(btn, "type", None) or btn
            kind = _button_kind(type(spec).__name__, inline)
            buttons.append(Button(
                id=f"r{r}c{c}", row=r, col=c, text=getattr(btn, "text", "") or "", kind=kind,
                data=getattr(spec, "data", None) if kind == "callback" else None,
                requires_password=bool(getattr(spec, "requires_password", False)),
                url=getattr(spec, "url", None),
                inline=inline,
            ))
    return tuple(buttons)


def classify_media(msg) -> tuple[str | None, str]:
    media = getattr(msg, "media", None)
    if media is None:
        return None, ""
    name = type(media).__name__
    if name == "MessageMediaPhoto" and getattr(media, "photo", None) is not None:
        return "photo", f"photo:{media.photo.id}"
    if name == "MessageMediaDocument" and getattr(media, "document", None) is not None:
        doc = media.document
        key = f"doc:{doc.id}"
        attrs = {type(a).__name__ for a in getattr(doc, "attributes", []) or []}
        mime = getattr(doc, "mime_type", "") or ""
        if "DocumentAttributeSticker" in attrs:
            return "sticker", key
        if mime.startswith("image/"):
            return "image", key
        if mime.startswith("video/") or "DocumentAttributeVideo" in attrs:
            return "video", key
        return "other", key
    if name == "MessageMediaWebPage":
        return None, ""
    return "other", name


def chat_media(msg) -> tuple[str | None, str, float]:
    """(kind, sticker emoji, duration) of a message in a private chat with a person.

    Kinds: photo, image, sticker, gif, video, video_note, voice, audio, other; None for plain text.
    """
    media = getattr(msg, "media", None)
    if media is None or type(media).__name__ == "MessageMediaWebPage":
        return None, "", 0.0
    if type(media).__name__ == "MessageMediaPhoto":
        return ("photo" if getattr(media, "photo", None) is not None else "other"), "", 0.0
    doc = getattr(media, "document", None)
    if doc is None:
        return "other", "", 0.0
    attrs = {type(a).__name__: a for a in getattr(doc, "attributes", []) or []}
    mime = getattr(doc, "mime_type", "") or ""
    if "DocumentAttributeSticker" in attrs:
        return "sticker", getattr(attrs["DocumentAttributeSticker"], "alt", "") or "", 0.0
    if "DocumentAttributeAudio" in attrs:
        audio = attrs["DocumentAttributeAudio"]
        return ("voice" if getattr(audio, "voice", False) else "audio"), "", float(audio.duration or 0)
    if "DocumentAttributeVideo" in attrs:
        video = attrs["DocumentAttributeVideo"]
        if getattr(video, "round_message", False):
            return "video_note", "", float(video.duration or 0)
        kind = "gif" if "DocumentAttributeAnimated" in attrs else "video"
        return kind, "", float(video.duration or 0)
    if "DocumentAttributeAnimated" in attrs:
        return "gif", "", 0.0
    if mime.startswith("image/"):
        return "image", "", 0.0
    return "other", "", 0.0


def extract_links(msg) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """URLs, t.me links and @mentions from message entities (offsets are UTF-16 based)."""
    text = msg.message or ""
    surrogated = helpers.add_surrogate(text)
    links: list[str] = []
    ids: list[int] = []
    for e in getattr(msg, "entities", None) or []:
        name = type(e).__name__
        if name == "MessageEntityTextUrl":
            links.append(e.url)
        elif name in ("MessageEntityUrl", "MessageEntityMention"):
            links.append(helpers.del_surrogate(surrogated[e.offset:e.offset + e.length]))
        elif name in ("MessageEntityMentionName", "InputMessageEntityMentionName"):
            uid = getattr(e, "user_id", None)
            if isinstance(uid, int):
                ids.append(uid)
    return tuple(dict.fromkeys(links)), tuple(dict.fromkeys(ids))


def normalize(msg) -> BotMessage:
    kind, key = classify_media(msg)
    markup = getattr(msg, "reply_markup", None)
    links, mention_ids = extract_links(msg)
    return BotMessage(
        id=msg.id,
        text=msg.message or "",
        buttons=parse_markup(getattr(msg, "reply_markup", None)),
        media_kind=kind,
        media_key=key,
        date=msg.date.timestamp() if getattr(msg, "date", None) else 0.0,
        edit_date=msg.edit_date.timestamp() if getattr(msg, "edit_date", None) else None,
        outgoing=bool(getattr(msg, "out", False)),
        keyboard_hidden=type(markup).__name__ == "ReplyKeyboardHide",
        links=links,
        mention_ids=mention_ids,
    )


# ------------------------------------------------------------------ transport
async def _call(coro):
    """Map Telethon errors onto the transport error model."""
    try:
        return await coro
    except tg_errors.FloodWaitError as exc:
        raise RateLimited(float(exc.seconds)) from exc
    except _RESTRICTED_ERRORS as exc:
        raise AccountRestricted(type(exc).__name__) from exc
    except tg_errors.RPCError as exc:
        raise TransportError(f"{type(exc).__name__}: {exc}") from exc
    except (TimeoutError, ConnectionError, OSError) as exc:
        raise TransportError(f"network: {exc}") from exc


class TelegramAccount:
    """One MTProto connection for the account; routes bot updates and owner commands."""

    def __init__(self, cfg: Config, client: TelegramClient | None = None):
        self.cfg = cfg
        self.client = client or make_client(cfg)
        self.me = None
        self.owner_peer = None
        self.router: Supervisor | None = None
        self.channels: dict[int, BotChannel] = {}  # bot user id -> channel
        self._tasks: set[asyncio.Task] = set()

    async def start(self) -> dict[str, BotChannel]:
        """Connect, verify every configured bot, register handlers. Returns channels by target name."""
        await self.client.connect()
        if not await self.client.is_user_authorized():
            raise TransportError("the session is not authorized; run `doppelgram login` first")
        self.me = await self.client.get_me()
        by_name: dict[str, BotChannel] = {}
        for target in self.cfg.targets:
            entity = await _call(self.client.get_entity(target.bot))
            if not getattr(entity, "bot", False):
                raise TransportError(f"{target.bot} is not a bot; refusing to automate a chat with a person")
            if entity.id in self.channels:
                other = self.channels[entity.id].target
                raise TransportError(f"bots {other.name!r} ({other.bot}) and {target.name!r} ({target.bot}) "
                                     f"are the same Telegram bot; configure it once")
            channel = BotChannel(self, target, await self.client.get_input_entity(entity), entity.id)
            self.channels[entity.id] = channel
            by_name[target.name] = channel
        await self._resolve_owner()

        c = self.client
        bot_ids = list(self.channels)
        c.add_event_handler(self._on_bot, events.NewMessage(chats=bot_ids))
        c.add_event_handler(self._on_bot, events.MessageEdited(chats=bot_ids))
        if self.cfg.owner_ids:
            c.add_event_handler(self._on_owner, events.NewMessage(chats=self.cfg.owner_ids, incoming=True))
        if self.cfg.allow_saved_messages:
            c.add_event_handler(self._on_saved, events.NewMessage(chats=[self.me.id], outgoing=True))
        c.add_event_handler(self._on_person, events.NewMessage(incoming=True, func=lambda e: e.is_private))
        log.info("logged in as %s (id %s); bots: %s; owner chat: %s",
                 self.me.username or self.me.first_name, self.me.id,
                 ", ".join(f"{t.name}={t.bot}" for t in self.cfg.targets),
                 "saved messages" if self.owner_peer is None else "DM")
        return by_name

    async def _resolve_owner(self) -> None:
        for attempt in range(2):
            for uid in self.cfg.owner_ids:
                try:
                    self.owner_peer = await self.client.get_input_entity(uid)
                    return
                except (ValueError, tg_errors.RPCError):
                    continue
            if attempt == 0:
                # Populate the entity cache from recent dialogs, then retry once.
                with contextlib.suppress(tg_errors.RPCError):
                    await self.client.get_dialogs(limit=100)
        if self.cfg.owner_ids:
            log.warning("owner %s not resolvable yet; write to this account from the owner account once. "
                        "Using Saved Messages meanwhile.", self.cfg.owner_ids)

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _on_bot(self, event) -> None:
        # chats=[id] also matches groups/channels with the same raw id; only the private bot chat counts.
        channel = self.channels.get(event.chat_id) if event.is_private else None
        if channel is not None and self.router:
            self.router.on_bot_event(channel.target.name)

    async def _on_owner(self, event) -> None:
        # Telethon expands chats=[id] to user, group and channel ids alike: accept only a private chat
        # whose sender is one of the owner accounts.
        if not event.is_private or event.sender_id not in self.cfg.owner_ids:
            return
        if self.owner_peer is None:
            self.owner_peer = await event.get_input_chat()
        if self.router:
            self._spawn(self._deliver(event, from_saved_messages=False))

    async def _on_person(self, event) -> None:
        """Private messages from people: matches go to the conversation manager; a first message from someone
        unknown goes there as a possible stranger, unless it is a contact, a bot or a service account."""
        conv = getattr(self.router, "conversations", None) if self.router else None
        sender = event.sender_id
        if conv is None or sender in self.cfg.owner_ids or sender in self.channels or sender == self.me.id:
            return
        if conv.knows(sender):
            conv.on_message(sender)
            return
        try:
            user = await event.get_sender()
        except Exception:  # the entity may be unavailable; it will be seen again on the next message
            return
        if not is_stranger_candidate(user):
            return
        from .conversation import PersonRef

        name = " ".join(filter(None, [user.first_name, getattr(user, "last_name", None)])) or "без имени"
        await conv.on_stranger(PersonRef(user.id, getattr(user, "username", None), name))

    async def _on_saved(self, event) -> None:
        if not event.is_private or event.chat_id != self.me.id:
            return
        if self.router:
            self._spawn(self._deliver(event, from_saved_messages=True))

    async def _deliver(self, event, *, from_saved_messages: bool) -> None:
        try:
            om = await self._owner_message(event)
        except AccountRestricted as exc:  # a TransportError subclass: trip the breaker, do not swallow
            await self.router.restrict_all(str(exc))
            return
        except TransportError as exc:
            log.warning("cannot read the owner's message: %s", exc)
            return
        await self.router.handle_owner(om, from_saved_messages=from_saved_messages)

    async def _owner_message(self, event) -> OwnerMessage:
        m = event.message
        image = None
        kind, _ = classify_media(m)
        doc = getattr(getattr(m, "media", None), "document", None)
        if kind == "photo" or (kind == "image" and (getattr(doc, "size", 0) or 0) <= OWNER_IMAGE_BYTES):
            image = await _call(self.client.download_media(m, file=bytes))
        return OwnerMessage(id=m.id, chat_id=event.chat_id, text=m.message or "",
                            reply_to=m.reply_to_msg_id, image=image, has_media=kind is not None)

    async def notify_owner(self, text, image: bytes | None = None) -> int | None:
        """Owner messages use Telegram HTML; plain text is escaped. Returns the id of the message the owner
        should reply to (the one carrying the text)."""
        target = self.owner_peer or "me"
        body = as_html(text)
        if image is not None:
            try:
                photo = io.BytesIO(to_jpeg(image, 1280))
                photo.name = "card.jpg"
            except Exception:
                photo = None
            if photo is not None:
                if len(body) <= CAPTION_LIMIT:
                    sent = await _call(self.client.send_file(target, photo, caption=body, parse_mode="html"))
                    return sent.id
                await _call(self.client.send_file(target, photo))
        sent = None
        for chunk in _chunks(body, MESSAGE_LIMIT):
            sent = await _call(self.client.send_message(target, chunk, parse_mode="html", link_preview=False))
        return sent.id if sent is not None else None


class BotChannel:
    """The Transport for one bot chat, sharing the account connection."""

    def __init__(self, account: TelegramAccount, target: Target, peer, bot_id: int):
        self.account = account
        self.target = target
        self.peer = peer
        self.bot_id = bot_id

    @property
    def client(self) -> TelegramClient:
        return self.account.client

    async def recent_bot_messages(self, limit: int) -> list[BotMessage]:
        msgs = await _call(self.client.get_messages(self.peer, limit=limit))
        return [normalize(m) for m in reversed(list(msgs))]

    async def download_image(self, msg_id: int) -> bytes | None:
        msg = await _call(self.client.get_messages(self.peer, ids=msg_id))
        if msg is None or msg.media is None:
            return None
        kind, _ = classify_media(msg)
        doc = getattr(msg.media, "document", None)
        size = getattr(doc, "size", 0) or 0
        mime = getattr(doc, "mime_type", "") or ""
        if kind == "photo" or (kind == "image" and size <= MAX_MEDIA_BYTES) or (
                kind == "sticker" and mime == "image/webp"):
            return await _call(self.client.download_media(msg, file=bytes))
        # Animated/video stickers and videos: the largest thumbnail is a still frame.
        if kind in ("sticker", "video", "image") and doc is not None and (
                getattr(doc, "thumbs", None) or getattr(doc, "video_thumbs", None)):
            return await _call(self.client.download_media(msg, file=bytes, thumb=-1))
        return None

    async def click(self, msg_id: int, button: Button) -> ClickOutcome:
        if button.kind == "reply":
            await _call(self.client.send_message(self.peer, button.text, parse_mode=None))
            return ClickOutcome(ok=True)
        if button.kind != "callback" or button.data is None:
            return ClickOutcome(ok=False, answer=f"unsupported button kind {button.kind}")
        req = functions.messages.GetBotCallbackAnswerRequest(peer=self.peer, msg_id=msg_id, data=button.data)
        try:
            answer = await _call(self.client(req))
        except TransportError as exc:
            cause = exc.__cause__
            if isinstance(cause, tg_errors.BotResponseTimeoutError):
                return ClickOutcome(ok=True, timed_out=True)
            if isinstance(cause, (tg_errors.MessageIdInvalidError, tg_errors.DataInvalidError)):
                return ClickOutcome(ok=False, answer=type(cause).__name__)
            raise
        return ClickOutcome(ok=True, answer=getattr(answer, "message", None),
                            alert=bool(getattr(answer, "alert", False)))

    async def send_to_bot(self, text: str) -> None:
        await _call(self.client.send_message(self.peer, text, parse_mode=None))

    async def notify_owner(self, text: str, image: bytes | None = None) -> int | None:
        return await self.account.notify_owner(text, image)


_USERNAME_RE = re.compile(r"(?:https?://)?(?:t\.me|telegram\.me)/([A-Za-z]\w{3,31})/?$|^@([A-Za-z]\w{3,31})$"
                          r"|tg://resolve\?domain=([A-Za-z]\w{3,31})")
_TG_USER_RE = re.compile(r"tg://user\?id=(\d+)")


def is_stranger_candidate(user) -> bool:
    """A real person the owner has no relation with: not a contact, not a bot, not a service, deleted, scam
    or fake account. Whether there was an earlier chat is checked by the conversation manager."""
    if user is None or type(user).__name__ != "User":
        return False
    return not any(getattr(user, flag, False) for flag in
                   ("bot", "contact", "mutual_contact", "deleted", "support", "scam", "fake", "is_self"))


class PeopleChannel:
    """Direct chats with matched people. Only users that are real people (not bots, not the owner)."""

    def __init__(self, account: TelegramAccount):
        self.account = account

    @property
    def client(self) -> TelegramClient:
        return self.account.client

    async def resolve(self, links: list[str], mention_ids: list[int]):
        from .conversation import PersonRef

        candidates: list = []
        for link in links:
            if m := _TG_USER_RE.search(link):
                candidates.append(int(m.group(1)))
            elif m := _USERNAME_RE.search(link.strip()):
                candidates.append(next(g for g in m.groups() if g))
        candidates += list(mention_ids)
        for ref in candidates:
            try:
                entity = await _call(self.client.get_entity(ref))
            except AccountRestricted:
                raise
            except (TransportError, ValueError):
                continue
            if type(entity).__name__ != "User" or getattr(entity, "bot", False) or getattr(entity, "deleted", False):
                continue
            if entity.id in self.account.cfg.owner_ids or (self.account.me and entity.id == self.account.me.id):
                continue
            name = " ".join(filter(None, [entity.first_name, getattr(entity, "last_name", None)])) or "без имени"
            return PersonRef(entity.id, getattr(entity, "username", None), name)
        return None

    async def history(self, user_id: int, limit: int):
        from .conversation import ChatMessage

        msgs = await _call(self.client.get_messages(user_id, limit=limit))
        out = []
        for m in reversed(list(msgs)):
            kind, emoji, duration = chat_media(m)
            out.append(ChatMessage(m.id, bool(m.out), m.message or "", m.date.timestamp() if m.date else 0.0,
                                   kind, emoji, duration))
        return out

    MAX_DOWNLOAD = 20 * 1024 * 1024
    WROTE_BEFORE_SCAN = 1000

    async def download(self, user_id: int, msg_id: int, *, thumb: bool = False) -> bytes | None:
        msg = await _call(self.client.get_messages(user_id, ids=msg_id))
        if msg is None or getattr(msg, "media", None) is None:
            return None
        doc = getattr(msg.media, "document", None)
        if thumb:
            if doc is not None and not (getattr(doc, "thumbs", None) or getattr(doc, "video_thumbs", None)):
                return None
            data = await _call(self.client.download_media(msg, file=bytes, thumb=-1))
        else:
            if doc is not None and (getattr(doc, "size", 0) or 0) > self.MAX_DOWNLOAD:
                return None
            data = await _call(self.client.download_media(msg, file=bytes))
        return data if isinstance(data, bytes) else None

    async def wrote_before(self, user_id: int) -> bool:
        """Whether this account ever sent anything in the chat: then it is not a stranger's first message."""
        # Telegram ignores a sender filter in private chats, so the history itself is scanned (once per
        # stranger or match, up to WROTE_BEFORE_SCAN messages back).
        msgs = await _call(self.client.get_messages(user_id, limit=self.WROTE_BEFORE_SCAN))
        return any(getattr(m, "out", False) for m in msgs or [])

    async def mark_read(self, user_id: int, max_id: int) -> None:
        await _call(self.client.send_read_acknowledge(user_id, max_id=max_id))

    BACKLOG_PAGE = 100

    async def _recent_dialogs(self, since: float) -> list:
        """All dialogs with activity since `since`, page by page (dialogs come newest first; pinned ones may
        be older, so only an unpinned old dialog ends the scan)."""
        found: list = []
        try:
            async for d in self.client.iter_dialogs():
                date = getattr(d, "date", None)
                if date is not None and date.timestamp() < since and not getattr(d, "pinned", False):
                    break
                found.append(d)
        except tg_errors.FloodWaitError as exc:
            raise RateLimited(float(exc.seconds)) from exc
        except _RESTRICTED_ERRORS as exc:
            raise AccountRestricted(type(exc).__name__) from exc
        except tg_errors.RPCError as exc:
            raise TransportError(f"{type(exc).__name__}: {exc}") from exc
        except (TimeoutError, ConnectionError, OSError) as exc:
            raise TransportError(f"network: {exc}") from exc
        return found

    async def backlog(self, days: int):
        """Private chats with unknown people whose last message is theirs and unread (recent first)."""
        import time as _time

        from .conversation import PersonRef

        since = _time.time() - days * 86400
        out = []
        for d in await self._recent_dialogs(since):
            user, msg = d.entity, d.message
            if not getattr(d, "is_user", False) or not is_stranger_candidate(user):
                continue
            if user.id in self.account.cfg.owner_ids or user.id in self.account.channels:
                continue
            if msg is None or msg.out or not d.unread_count or not msg.date or msg.date.timestamp() < since:
                continue
            name = " ".join(filter(None, [user.first_name, getattr(user, "last_name", None)])) or "без имени"
            out.append(PersonRef(user.id, getattr(user, "username", None), name))
        return out

    async def profile(self, user_id: int) -> str:
        full = await _call(self.client(functions.users.GetFullUserRequest(user_id)))
        return getattr(getattr(full, "full_user", None), "about", None) or ""

    async def send(self, user_id: int, text: str, reply_to: int | None = None) -> int:
        sent = await _call(self.client.send_message(user_id, text, parse_mode=None, reply_to=reply_to))
        return sent.id

    async def typing(self, user_id: int, seconds: float) -> None:
        try:
            async with self.client.action(user_id, "typing"):
                await asyncio.sleep(seconds)
        except tg_errors.RPCError:
            await asyncio.sleep(seconds)
