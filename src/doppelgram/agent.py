"""Agent core: observe the bot chat, decide with the LLM, act, and obey the owner.

Telegram updates only *wake* the loop; every step re-reads the chat from the server.
That makes edits-in-place, reconnects and duplicate updates all the same case.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import random
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

from . import brain
from . import schedule as sched
from . import settings as st
from .config import BehaviorConfig, Config
from .control import Command, help_text, parse_answer, parse_owner
from .llm import LLMClient, LLMError
from .media import image_data_url
from .models import BotMessage, Button, Decision, OwnerMessage
from .storage import Pending, Store, scrub_text
from .taste import TasteProfile
from .transport import AccountRestricted, RateLimited, Transport, TransportError
from .ui import Html, as_html, b, code, count, esc, i, join

log = logging.getLogger(__name__)

COMPANION_IMAGE_WINDOW_S = 120.0
MAX_LLM_FAILURE_STREAK = 5


def _ago(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 90:
        return f"{seconds} с назад"
    if seconds < 5400:
        return f"{seconds // 60} мин назад"
    return f"{seconds / 3600:.1f} ч назад"


def _loggable(text: str | None, msg: BotMessage) -> str:
    """What is kept of a typed message: commands, numbers and button labels; anything else is not stored."""
    t = (text or "").strip()
    if t.startswith("/"):
        return t.split()[0][:40]  # the command only, never its arguments
    if (t.isdigit() and len(t) <= 3) or any(btn.text.strip() == t for btn in msg.buttons):
        return t[:40]
    return "(сообщение)"


def _digest(text: str) -> str:
    return hashlib.sha256((text or "").encode()).hexdigest()[:16]


def _hours(h: float) -> str:
    if h < 1:
        return f"{round(h * 60)} мин"
    return f"{h:g} ч"


@dataclass(frozen=True)
class Observation:
    """What the bot currently shows: its latest incoming message plus the picture that belongs to it."""

    msgs: list[BotMessage]
    current: BotMessage
    image_msg: BotMessage | None  # `current` itself, or a separate picture message sent right before it
    rev: str  # identity of the card: keyboard message content + picture content
    keyboard_id: int | None = None  # the message that carries the active reply keyboard

    @property
    def image_from_previous(self) -> bool:
        return self.image_msg is not None and self.image_msg.id != self.current.id




def _menu_lines(text: str) -> dict[str, str]:
    return brain.menu_lines(text)


# Purchase intent in any button label: such a button is never a refusal, whatever else it says.
_BUY_RE = re.compile(r"купи|оформ|оплат|подключ|активир|хочу|приобр|buy|purchase|activate|subscribe|upgrade", re.I)
# Agreeing to whatever the message asks: under a charge question that means paying.
_AGREE_RE = re.compile(r"\W*(?:(?:да|ок|ok|yes|конечно)\b|подтвер|соглас|продолж|continue|confirm)", re.I)
# A price on a button: whatever else it says, pressing it may cost money.
_PRICE_RE = re.compile(r"₽|\$|€|руб|⭐|stars?\b|\bза\s*\d", re.I)


# Screens only a person can handle, whatever the config says: blocks, bans, identity checks.
# Phrased as a state of the account: an ad promising "how not to get banned" must not match.
_OWNER_ALWAYS = re.compile(r"(?:аккаунт|анкет\w*|профил\w*|страниц\w*|\bвы)\W+(?:\w+\W+){0,2}?"
                           r"(?:заблокирован|забанен|ограничен)|верификац|подтверд\w*\s+личност|селфи|капч"
                           r"|\b(?:banned|blocked|suspended|captcha)\b", re.I)
# A message that charges for something: no button on it is a safe way on, only a restart is.
# A bare ⭐ is decoration (Bibinto's "⭐️ Кто-то оценил вас"); a number next to a currency is a price.
_CHARGE_RE = re.compile(r"\d\s*(?:₽|\$|€|руб|⭐|звёзд|звезд|stars?\b)|[₽$€]\s*\d|\bза\s*\d|списа|покупк|купи|оплат",
                        re.I)


def _affirmative(text: str) -> str:
    """The text without negated words: "это не капча" does not say it is a captcha."""
    return re.sub(r"\b(?:не|нет|без|no|not|without)\s+\w+", " ", text or "", flags=re.I)


def _is_safe_continue(label: str, words: list[str]) -> bool:
    """A button that leads on without buying anything."""
    return (any(w and w.lower() in label.lower() for w in words)
            and not _BUY_RE.search(label) and not _PRICE_RE.search(label))


def _is_refusal(text: str, decline: tuple[str, ...]) -> bool:
    """A label that starts with a refusal word ("Нет", "Пока без …") and shows no purchase intent."""
    label = re.sub(r"^[^\w]+", "", text or "")  # leading emoji/punctuation
    affirmative = re.sub(r"\b(?:не|без|no|without)\s+\w+", " ", label, flags=re.I)  # "не хочу", "без подключения"
    if _BUY_RE.search(affirmative) or _PRICE_RE.search(label):  # "Пропустить за 10 ⭐" costs money
        return False
    return any(w and re.match(rf"{re.escape(w)}(?!\w)", label, re.I) for w in decline)


def _with_deny(b: Button, deny: tuple[str, ...], menu: dict[str, str], decline: tuple[str, ...] = ()) -> Button:
    """Blocked if the label contains a denied word (any case), or if the label is a menu number whose
    line in the message text does ("2" for "2. Купить Премиум"). A refusal ("Пока без Premium") is never
    blocked, even when it names the paid feature it refuses."""
    if _is_refusal(b.text, decline):
        return b
    texts = [b.text.lower()]
    number = re.match(r"\s*(\d{1,2})(?!\d)", b.text)
    if number and number.group(1) in menu:
        texts.append(menu[number.group(1)].lower())
    if any(d and d.lower() in t for d in deny for t in texts):
        return replace(b, blocked=True)
    return b


def observe(msgs: list[BotMessage], deny: tuple[str, ...] = (), decline: tuple[str, ...] = ()) -> Observation | None:
    """Build the current card view: latest incoming message + its picture + the active keyboards.

    A reply keyboard stays on screen until the bot replaces or hides it, so it is taken from the latest
    incoming message that carries one (Bibinto sends the keyboard before the picture; Leomatch keeps it).
    """
    incoming = [m for m in msgs if not m.outgoing]
    if not incoming:
        return None
    current = incoming[-1]
    image_msg = current if current.has_image else None
    if image_msg is None:
        # Some bots send the picture and the keyboard as two messages.
        idx = next(i for i, m in enumerate(msgs) if m.id == current.id)
        if idx > 0:
            prev = msgs[idx - 1]
            if (not prev.outgoing and prev.has_image
                    and abs(current.date - prev.date) <= COMPANION_IMAGE_WINDOW_S):
                image_msg = prev

    keyboard_msg = None
    for m in reversed(incoming):
        if m.keyboard_hidden:
            break
        if any(not b.inline for b in m.buttons):
            keyboard_msg = m
            break
    inline = tuple(b for b in current.buttons if b.inline)
    reply: tuple[Button, ...] = ()
    if keyboard_msg is not None:
        reply = tuple(replace(b, id="k" + b.id[1:]) for b in keyboard_msg.buttons if not b.inline)
    # Bot menus are text messages; a profile's bio (on a picture card) may contain a numbered list too.
    menu = {} if current.has_image else _menu_lines(current.text)
    view = replace(current, buttons=tuple(_with_deny(b, deny, menu, decline) for b in inline + reply))

    rev = view.revision  # covers text, media and every visible button (including the reply keyboard)
    if image_msg is not None and image_msg.id != current.id:
        rev = hashlib.sha256(f"{rev}|{image_msg.id}|{image_msg.revision}".encode()).hexdigest()[:16]
    return Observation(msgs, view, image_msg, rev, keyboard_msg.id if keyboard_msg is not None else None)


class Agent:
    def __init__(self, cfg: Config, transport: Transport, llm: LLMClient, store: Store, *,
                 clock: Callable[[], float] = time.time, sleep=asyncio.sleep,
                 rng: random.Random | None = None, behavior: BehaviorConfig | None = None,
                 label: str = "", taste: TasteProfile | None = None,
                 on_restricted: Callable[[str], Awaitable[None]] | None = None, name: str = ""):
        self.cfg = cfg
        self.store = store
        self._base = behavior or cfg.behavior  # config values
        self._effective = None                  # config + overrides set from Telegram (cached)
        self.label = label  # bot name shown in owner messages when several bots run
        self.taste = taste
        self.on_restricted = on_restricted  # account-wide circuit breaker (stops every bot)
        self._ref_cache: dict[int, str] = {}
        self.name = name or label or "bot"
        # (bot, links, mention ids, card text, age) -> conversation manager, set by the supervisor
        self.on_match: Callable[..., Awaitable[None]] | None = None
        # Recently decided picture cards (text, taste score), to attach to a match. In memory only.
        self._recent_cards: list[tuple[str, int | None]] = []
        self._breaker_reason: str | None = None
        self.tg = transport
        self.llm = llm
        self.clock = clock
        self.sleep = sleep
        self.rng = rng or random.Random()
        self.wake = asyncio.Event()
        self.lock = asyncio.Lock()
        self.last_bot_event = 0.0
        self.notes: list[str] = []
        self.llm_failures = 0
        # Bumped by stop/pause/mode commands: a model decision prepared under an older epoch is dropped.
        self.epoch = 0
        if store.get("running") is None:
            store.set("running", "1" if self._base.autostart else "0")
        legacy_numbers = store.get("setting:hide_numbers_10_20")  # renamed setting
        if legacy_numbers is not None:
            if store.get("setting:hide_numbers_10_18") is None:
                store.set("setting:hide_numbers_10_18", legacy_numbers)
            store.unset("setting:hide_numbers_10_20")
        legacy_mode = store.get("mode")  # older versions stored the mode directly
        if legacy_mode is not None:
            if legacy_mode != self._base.mode and store.get("setting:mode") is None:
                store.set("setting:mode", json.dumps(legacy_mode))
            store.unset("mode")

    # --------------------------------------------------------------- settings
    @property
    def b(self):
        """Effective behavior: config.toml values with overrides set from Telegram."""
        if self._effective is None:
            values: dict = {}
            for s in st.SETTINGS:
                if s.scope == "bot" and (raw := self.store.get(f"setting:{s.key}")) is not None:
                    try:
                        values.update(st.to_fields(s, tuple(json.loads(raw)) if s.key == "action_delay"
                                                   else json.loads(raw)))
                    except (ValueError, TypeError):
                        log.warning("ignoring a broken stored setting %s", s.key)
            self._effective = replace(self._base, **values) if values else self._base
        return self._effective

    @b.setter
    def b(self, behavior) -> None:
        self._base = behavior
        self._effective = None

    def set_setting(self, setting: st.Setting, value) -> None:
        self.store.set(f"setting:{setting.key}", json.dumps(value))
        self._effective = None
        self.epoch += 1  # decisions prepared under the old settings are dropped
        self.wake.set()

    def reset_setting(self, setting: st.Setting) -> None:
        self.store.unset(f"setting:{setting.key}")
        self._effective = None
        self.epoch += 1
        self.wake.set()

    def overridden(self, setting: st.Setting) -> bool:
        return self.store.get(f"setting:{setting.key}") is not None

    # ------------------------------------------------------------------ state
    @property
    def running(self) -> bool:
        return self.store.get("running") == "1"

    @property
    def mode(self) -> str:
        return self.b.mode

    @property
    def paused_until(self) -> float:
        return self.store.get_float("paused_until")

    @property
    def restricted(self) -> str | None:
        return self.store.get("restricted") or None

    def restrict(self, reason: str) -> None:
        self.epoch += 1
        self.store.set("restricted", reason)
        self.store.set("running", "0")

    def active(self) -> bool:
        return (self.running and not self.restricted and self.clock() >= self.paused_until
                and self.clock() >= (self.limit_resume_at() or 0) and self.schedule_resume_at() is None)

    # -------------------------------------------------- working hours, daily norm
    @property
    def _tz(self):
        return sched.parse_tz(self.b.timezone)

    def _time(self, ts: float, at: bool = True) -> str:
        return sched.format_local(ts, self.clock(), self._tz, at=at)

    def _norm_done(self, now: float) -> bool:
        return (self.b.daily_ratings_enabled
                and self.store.cards_since(sched.day_start(now, self._tz)) >= self.b.daily_ratings)

    def schedule_resume_at(self) -> float | None:
        """Outside working hours, or with today's norm done: when the bot may act again. None if it may now."""
        now = self.clock()
        at = sched.next_day_start(now, self._tz) if self._norm_done(now) else now
        if self.b.active_hours_enabled:
            at = sched.next_window_start(at, sched.parse_window(self.b.active_hours), self._tz)
        return at if at > now else None

    async def _tell_norm(self, now: float) -> None:
        """Once a day, when the norm is reached."""
        day = str(int(sched.day_start(now, self._tz)))
        if not self._norm_done(now) or self.store.get("norm_told") == day:
            return
        self.store.set("norm_told", day)
        at = self.schedule_resume_at() or now
        await self.say(Html(f"{b('Дневная норма')}: {count(self.b.daily_ratings, 'анкета', 'анкеты', 'анкет')} "
                            f"за сегодня. Продолжу {esc(self._time(at))}. "
                            f"Изменить: {code('/set норма 200')}, выключить: {code('/set дневная_норма выкл')}"))

    # -------------------------------------------------------- bot warnings
    def _is_warning(self, msg: BotMessage) -> bool:
        return bool(self.b.warning_pattern and not msg.has_image
                    and re.search(self.b.warning_pattern, msg.text or "", re.I)
                    and not self._owner_required(msg.text))

    async def _handle_warning(self, obs: Observation, epoch: int) -> bool:
        """"Слишком быстро", "подозрительная активность": wait a few seconds, then go on. Never stop."""
        cur = obs.current
        if not self._is_warning(cur):
            return False
        now = self.clock()
        try:
            w = json.loads(self.store.get("warning") or "null")
        except ValueError:
            w = None
        # An edited message with a new warning is a new warning: keyed by id and content.
        if not isinstance(w, dict) or w.get("msg") != cur.id or w.get("rev") != obs.rev:
            self.store.set("warning", json.dumps({"msg": cur.id, "rev": obs.rev, "at": now}))
            self.store.log_action(now, cur.id, obs.rev, "warning", decided_by="rule")
            self.pause_for(self.b.warning_pause_s)
            return True
        if w.get("done"):
            return False  # we went on and the bot left the warning on screen: decide normally
        if now < float(w.get("at", 0)) + self.b.warning_pause_s:
            return True
        if await self._limits_exceeded(now):  # the hourly cap and the budget apply here too
            return True
        if self._looping():
            return False
        button = None if _CHARGE_RE.search(cur.text or "") else next(
            (btn for btn in cur.buttons if btn.clickable
             and _is_safe_continue(btn.text, [*self.b.continue_buttons, *self.b.resume_buttons])), None)
        if button is not None:
            d = Decision(action="click", button_id=button.id, reason="предупреждение бота: продолжаю")
        elif self.b.start_command:
            d = Decision(action="send", text=self.b.start_command, reason="предупреждение бота: продолжаю")
        else:
            return False
        if await self._execute(obs, d, decided_by="rule", epoch=epoch):
            w["done"] = True  # only once it really went: a cancelled press is tried again
            self.store.set("warning", json.dumps(w))
        return True

    # ------------------------------------------------------------ bot limits
    LIMIT_REPLY_S = 60            # how long the bot may take to answer a retry
    LIMIT_NOTIFY_EVERY_S = 6 * 3600  # at most one "limit reached" message per this period

    def _limit(self) -> dict | None:
        try:
            q = json.loads(self.store.get("limit") or "null")
        except ValueError:
            return None
        return q if isinstance(q, dict) and "msg" in q and "at" in q else None

    def _save_limit(self, q: dict) -> None:
        self.store.set("limit", json.dumps(q))

    def limit_resume_at(self) -> float | None:
        """While the bot shows its limit notice: when the agent tries again. None otherwise."""
        q = self._limit()
        if q is None or q.get("tried") or not self.b.limit_resume:
            return None
        return float(q["at"]) + self.b.limit_retry_h * 3600

    def _is_limit_notice(self, msg: BotMessage) -> bool:
        return bool(self.b.limit_pattern and not msg.has_image
                    and re.search(self.b.limit_pattern, msg.text or "", re.I))

    async def _handle_limit(self, obs: Observation, epoch: int) -> bool:
        """The bot gives no more profiles for now: pause without the model, then try again (or stop)."""
        cur = obs.current
        q = self._limit()
        if not self._is_limit_notice(cur):
            if q is not None and not q.get("tried") and not q.get("now") and not self.b.limit_resume:
                # Turned off during the pause, and something else (a "someone rated you") came on top.
                await self._stop_on_limit()
                return True
            return False
        now = self.clock()
        if q is None or q["msg"] != cur.id:  # a new notice: the first one, or the bot is still limited
            first = q is None
            asked = bool(q and q.get("now"))  # the owner's /start came before we saw this notice
            # Counted from when the bot sent it: an old notice (the agent was stopped) is retried at once.
            sent = cur.date if 0 < cur.date <= now else now
            q = {"msg": cur.id, "at": sent, "since": q["since"] if q else sent,
                 "tries": q.get("tries", 0) if q else 0, "tried": False, "now": asked}
            self._save_limit(q)
            self.store.log_action(now, cur.id, obs.rev, "limit", decided_by="rule", result="new" if first else "again")
            if not asked:
                if not self.b.limit_resume:
                    await self._stop_on_limit()
                    return True
                if float(q["at"]) + self.b.limit_retry_h * 3600 > now:
                    await self._tell_limit(now)
                    self.wake.set()
                    return True
        if q.get("tried"):
            if now - float(q["at"]) >= self.LIMIT_REPLY_S:
                # The bot did not answer the retry: wait another interval on the same notice.
                q.update(tried=False, at=now)
                self._save_limit(q)
                self.wake.set()
            return True
        if not self.b.limit_resume and not q.get("now"):  # only the owner's /start tries again
            await self._stop_on_limit()
            return True
        if not q.get("now") and now < float(q["at"]) + self.b.limit_retry_h * 3600:
            return True
        if self.store.actions_since(now - 3600) >= self.b.max_actions_per_hour or self._looping():
            q.update(at=now, now=False)  # the hourly caps apply to retries too: try at the next interval
            self._save_limit(q)
            return True
        # A notice that also charges for something ("Списать 99 звёзд? Нажмите 🚀"): only a restart is safe.
        button = None if _CHARGE_RE.search(cur.text or "") else next(
            (btn for btn in cur.buttons if btn.clickable
             and _is_safe_continue(btn.text, [*self.b.resume_buttons, *self.b.continue_buttons])), None)
        if button is not None:
            d = Decision(action="click", button_id=button.id, reason="лимит: пробую снова")
        elif self.b.start_command:
            d = Decision(action="send", text=self.b.start_command, reason="лимит: пробую снова")
        else:
            await self._stop_on_limit()
            return True
        q.update(tried=True, at=now, tries=int(q.get("tries", 0)) + 1, now=False)
        self._save_limit(q)
        await self._execute(obs, d, decided_by="rule", epoch=epoch)
        return True

    async def _tell_limit(self, now: float) -> None:
        if now - self.store.get_float("limit_notified") < self.LIMIT_NOTIFY_EVERY_S:
            return
        self.store.set("limit_notified", str(now))
        at = self.limit_resume_at() or now
        await self.say(Html(
            f"{b('Лимит анкет')}: бот пока не даёт оценивать. Попробую снова через "
            f"{esc(_hours(self.b.limit_retry_h))} ({esc(self._time(at))}). "
            f"Раньше: {code('/start')}. Не продолжать самому: {code('/set автопродолжение выкл')}"))

    async def _stop_on_limit(self) -> None:
        self.stop_now()
        await self.say(Html(f"{b('Лимит анкет')}: бот пока не даёт оценивать, я остановился "
                            f"(автопродолжение выключено). Когда лимит обновится: {code('/start')}"))

    def pause_for(self, seconds: float) -> None:
        self.store.set("paused_until", str(max(self.paused_until, self.clock() + seconds)))
        self.wake.set()

    def on_bot_event(self) -> None:
        self.last_bot_event = self.clock()
        self.wake.set()

    async def say(self, text, image: bytes | None = None) -> int | None:
        """Plain text is escaped; Html is sent as is. With several bots, messages carry the bot's name."""
        try:
            body = as_html(text)
            if self.label:
                body = Html(f"{b(self.label)} · {body}")
            return await self.tg.notify_owner(body, image=image)
        except AccountRestricted as exc:
            # Never swallow a restriction: stop acting now, trip the account breaker after this step.
            log.critical("account restricted while notifying the owner: %s", exc)
            self.restrict(str(exc) or "restricted")
            self._breaker_reason = str(exc) or "restricted"
            return None
        except Exception:  # notifications must never crash the loop
            log.exception("failed to notify owner")
            return None

    # ------------------------------------------------------------------- loop
    async def run_forever(self, announce: bool = True) -> None:
        if announce:
            await self.say(join(b("Doppelgram запущен."), self.status_text()))
        while True:
            await self._wait_for_wake()
            await self.maybe_report()
            if not self.active():
                continue
            while (remaining := self.b.settle_s - (self.clock() - self.last_bot_event)) > 0:
                await self.sleep(remaining)
            await self.step_safely()

    async def _wait_for_wake(self) -> None:
        now = self.clock()
        timeout = self.b.poll_interval_s
        if self.paused_until > now:
            timeout = min(timeout, self.paused_until - now + 1)
        for at in (self.limit_resume_at(), self.schedule_resume_at()):
            if at is not None and at > now:
                timeout = min(timeout, at - now + 1)
        retry_at = self.store.next_retry_at(now)
        if retry_at is not None:
            timeout = min(timeout, retry_at - now + 1)
        timeout_at = self._next_timeout_at()  # smart mode: an unanswered doubt resolves itself
        if timeout_at is not None:
            timeout = min(timeout, max(timeout_at - now, 0) + 1)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self.wake.wait(), timeout=max(timeout, 1.0))
        self.wake.clear()

    async def maybe_report(self) -> None:
        if not self.running or self.b.report_every_h <= 0:
            return
        now = self.clock()
        last = self.store.get_float("last_report", now)
        if self.store.get("last_report") is None:
            self.store.set("last_report", str(now))
        elif now - last >= self.b.report_every_h * 3600:
            self.store.set("last_report", str(now))
            await self.say("Отчёт. " + self.status_text())

    async def step_safely(self) -> None:
        if await self._guarded(self.step()):
            self.llm_failures = 0

    async def _guarded(self, work: Awaitable[None]) -> bool:
        """Run one unit of work; Telegram limits and failures put the agent into a safe state."""
        try:
            return await self._guarded_inner(work)
        finally:
            reason, self._breaker_reason = self._breaker_reason, None
            if reason and self.on_restricted is not None:
                await self.on_restricted(reason)

    async def _guarded_inner(self, work: Awaitable[None]) -> bool:
        try:
            await work
            return True
        except RateLimited as exc:
            wait = exc.seconds + self.rng.uniform(1, 5)
            log.warning("flood wait %.0fs", exc.seconds)
            self.store.set("flood_until", str(self.clock() + wait))
            self.pause_for(wait)
            if exc.seconds >= 60:
                await self.say(f"Telegram попросил подождать {exc.seconds:.0f} с (FLOOD_WAIT). "
                               f"Продолжу {self._time(self.clock() + wait)}.")
        except AccountRestricted as exc:
            log.critical("account restricted: %s", exc)
            self.restrict(str(exc) or "restricted")
            if self.on_restricted is not None:
                await self.on_restricted(str(exc) or "restricted")
            else:
                await self.say(f"Аккаунт ограничен Telegram ({exc}). Все действия остановлены. "
                               f"Проверьте аккаунт; команда /start снимет блокировку агента.")
        except LLMError as exc:
            self.llm_failures += 1
            log.error("LLM failure %d: %s", self.llm_failures, exc)
            if self.llm_failures >= MAX_LLM_FAILURE_STREAK:
                self.store.set("running", "0")
                await self.say(f"Модель не отвечает {self.llm_failures} раз подряд, остановился. "
                               f"Последняя ошибка: {str(exc)[:300]}")
            else:
                self.pause_for(120 * self.llm_failures)
                if self.llm_failures == 1:
                    await self.say(f"Ошибка модели, повторю позже: {str(exc)[:300]}")
        except TransportError as exc:
            log.error("telegram error: %s", exc)
            self.pause_for(60)
        except Exception:
            log.exception("unexpected error")
            self.pause_for(60)
        return False

    # ------------------------------------------------------------------- step
    async def step(self) -> None:
        async with self.lock:
            await self._step()

    async def _observe(self) -> Observation | None:
        # A reply keyboard stays on screen after the message that set it scrolls away: look far enough back.
        return observe(await self.tg.recent_bot_messages(max(self.b.history_in_prompt + 2, self.KEYBOARD_LOOKBACK)),
                       tuple(self.b.deny_buttons), tuple(self.b.decline_buttons))

    async def _step(self) -> None:
        if not self.active():
            return
        epoch = self.epoch
        now = self.clock()
        obs = await self._observe()
        if obs is None:
            if self.store.get("start_sent") is None:
                await self.tg.send_to_bot(self.b.start_command)
                self.store.set("start_sent", str(now))
                self.store.log_action(now, None, None, "send", text=self.b.start_command.split()[0],
                                      reason="чат с ботом пуст")
            return

        current = obs.current
        self.store.expire_pending("spot", now - 86400)  # unanswered taste checks do not linger
        # A match notice is often followed by the next card (or a limit notice) within the same second.
        for m in obs.msgs:
            if not m.outgoing:
                await self._record_match(m, now)
        if await self._handle_limit(obs, epoch):
            return
        if await self._handle_warning(obs, epoch):
            return
        status = self.store.handled_status(current.id, obs.rev, now)
        if status == "pending":
            pending = self.store.pending_for(current.id, obs.rev)
            if pending is not None and not self._card_fits_mode(pending.kind):
                # The mode changed since the card was sent (command, config edit, restart): decide anew.
                self.store.delete_pending(pending.control_msg_id)
                self.store.unmark(current.id, obs.rev)
                self.wake.set()
                return
            await self._maybe_time_out(obs, now, epoch)
            return
        if status is not None:
            await self._check_stuck(obs, status, now)
            return
        # Only when the edited message IS the picture: a changed separate text/keyboard message is a new card.
        acted_text = (self.store.get(self._acted_key(obs))
                      if obs.image_msg is not None and obs.image_msg.id == current.id else None)
        if acted_text is not None and (obs.image_msg.text == "" or _digest(obs.image_msg.text) == acted_text):
            # Same picture message we already pressed a button for, merely edited by the bot
            # (Bibinto drops the caption after rating): wait for the next card instead of rating again.
            # A new caption on the same picture is a new card and is decided normally.
            self.store.mark(current.id, obs.rev, "done", now)
            return

        if await self._limits_exceeded(now):
            return
        if await self._age_gate(obs, epoch):
            return
        if self._await_keyboard(obs, now):
            return
        # A block or a check is for the owner, even if it also offers Premium or a "see who rated you".
        for_owner = not obs.current.has_image and self._owner_required(obs.current.text)
        if not for_owner and await self._open_incoming_rating(obs, epoch):
            return
        if not for_owner and await self._decline_paid_offer(obs, epoch):
            return
        reciprocal = self._reciprocal_score(obs)
        if reciprocal is not None:
            self.notes.append("This person has already rated the owner highly. Rate them as usual, by the "
                              "owner's taste; the agent adjusts a fitting rating itself.")

        image = await self.tg.download_image(obs.image_msg.id) if obs.image_msg else None
        image_url = None
        if image:
            try:
                image_url = image_data_url(image, self.cfg.llm.image_max_side)
            except Exception as exc:  # unreadable/animated formats
                log.warning("cannot decode image of #%s: %s", obs.image_msg.id, exc)
        if obs.image_msg is not None and image_url is None:
            self.notes.append("The message has a picture, but it could not be downloaded or decoded.")

        messages = brain.build_messages(
            task=self._read(self.b.task_file),
            taste=self.taste.text() if self.taste else self._read(self.b.taste_file),
            references=self._references(),
            examples=self.store.recent_examples(self.b.examples_in_prompt),
            history=[m for m in obs.msgs if m.id != current.id][-self.b.history_in_prompt:],
            recent_actions=self.store.recent_actions(5), current=current,
            sanitize=self._sanitize if (self.b.age_in_prompt != "keep" or self.b.hide_numbers_10_18) else None,
            image_url=image_url, image_from_previous=obs.image_from_previous, notes=self.notes,
        )
        decision = await self._decide(messages, current)
        self.notes = []
        if epoch != self.epoch or not self.active():
            log.info("owner changed state while deciding #%s; dropping the decision", current.id)
            self.wake.set()
            return

        decision = self._apply_reciprocal(obs, decision)
        # Only a rating of a profile needs the owner in manual mode (even if the picture failed to load);
        # other presses on a picture message (an ad's "Пропустить", a menu) are not ratings.
        rating = decision.action == "click" and self._is_rating(obs, decision.button_id)
        needs_owner = rating and (obs.image_msg is not None or not self.b.supervise_images_only)
        if decision.action != "ask_owner" and not current.has_image and self._owner_required(current.text):
            # A block, a ban, a captcha, a verification: a text screen, not a profile; only a person can deal with it.
            decision = Decision(action="ask_owner", reason="нужен человек: блокировка, проверка или капча")
        if self._may_pay(obs, decision):
            decision = Decision(action="ask_owner", reason="платное предложение")
        if self.mode in ("smart", "auto") and await self._keep_going(obs, decision, epoch):
            return
        if (decision.action in ("click", "send") and not self._rates_profile(obs, decision.button_id)
                and self._looping()):
            decision = Decision(action="ask_owner", reason="бот ходит по кругу: много действий и ни одной анкеты")
        if needs_owner:
            if self.mode == "supervised":
                await self._ask_owner_to_confirm(obs, decision, image, epoch, kind="confirm")
                return
            if self.mode == "smart" and self._borderline(decision):
                await self._ask_owner_to_confirm(obs, decision, image, epoch, kind="unsure")
                return
        done = await self._execute(obs, decision, decided_by="model", image=image, epoch=epoch)
        if done and rating and obs.image_msg is not None and self.mode in ("smart", "auto"):
            await self._maybe_spot_check(obs, decision, image)

    async def _decide(self, messages: list[dict], current: BotMessage) -> Decision:
        failures: list = []
        try:
            result = await self.llm.complete_json(messages, failures=failures)
        finally:
            # Failed attempts (including paid replies that were not valid JSON) count against the budget,
            # even when this task is cancelled while a fallback model is still answering.
            self._log_failures(failures)
        self.store.log_llm(self.clock(), result.model, result.prompt_tokens, result.completion_tokens,
                           result.cost_usd, ok=True)
        decision = brain.validate_decision(result.data, current, max_wait_s=self.b.max_wait_s,
                                           max_send_chars=self.b.max_send_chars, deny=tuple(self.b.deny_buttons),
                                           allowed_commands=self._allowed_commands())
        # No explanation in the system log: it may describe a person.
        log.info("decision for #%s: %s %s conf=%s", current.id, decision.action,
                 decision.button_id or decision.wait_seconds or "", decision.confidence)
        return decision

    @staticmethod
    def _acted_key(obs: Observation) -> str:
        return f"acted:{obs.image_msg.id}:{obs.image_msg.media_key}"

    def _references(self) -> list[tuple[bool, str, str]]:
        if self.taste is None:
            return []
        out = []
        for ref in self.taste.prompt_references(self.b.max_reference_images):
            url = self._ref_cache.get(ref.id)
            if url is None:
                try:
                    url = image_data_url(ref.path.read_bytes(), 768)
                except (OSError, ValueError):
                    continue
                self._ref_cache[ref.id] = url
            out.append((ref.liked, ref.note, url))
        return out

    async def _record_match(self, msg: BotMessage, now: float) -> None:
        # The bot's notice is a text message; a caption on a picture is a profile someone wrote, and its
        # "Начинай общаться 👉 @someone" must never make a stranger a match.
        if not self.b.match_pattern or msg.media_kind or not re.search(self.b.match_pattern, msg.text or ""):
            return
        if self.store.add_match(now, msg.id, msg.text, list(msg.links), list(msg.mention_ids)):
            contact = ", ".join(msg.links) or ("профиль по ссылке в сообщении бота" if msg.mention_ids else "")
            card, score = self._card_for_match(msg.text)
            await self.say(f"💞 Взаимная симпатия! {contact}\n{msg.text[:300]}"
                           + (f"\nАнкета: {card[:300]}" if card else "")
                           + (f"\nОценка агента: {score}/10" if score is not None else ""))
            if self.on_match is not None:
                # The card is stored as is; the conversation module sanitizes it when it builds a prompt.
                await self.on_match(self.name, list(msg.links), list(msg.mention_ids), card,
                                    self._card_age(card) if card else None, score)

    def _allowed_commands(self) -> tuple[str, ...]:
        start = self.b.start_command.split()[:1]
        return tuple(dict.fromkeys([*start, *self.b.allowed_commands]))

    def _sanitize(self, text: str) -> str:
        return brain.sanitize_text(text, self.b.age_in_prompt, self.b.age_pattern, self.b.hide_numbers_10_18)

    def _card_for_match(self, notice: str) -> tuple[str, int | None]:
        """The profile card of the matched person and its taste score, by the name in the notice ('… 👉 Маша')."""
        name = notice.rsplit("👉", 1)[-1].strip() if "👉" in notice else ""
        for card, score in reversed(self._recent_cards):
            if name and card.startswith(name):
                return card, score
        return "", None

    def _card_age(self, text: str) -> int | None:
        if not self.b.age_pattern:
            return None
        m = re.search(self.b.age_pattern, text or "", re.M)
        try:
            return int(m.group(1)) if m else None
        except (IndexError, ValueError):
            return None

    async def _age_gate(self, obs: Observation, epoch: int) -> bool:
        """Cards below the minimum age are skipped by rule, without the model and without the owner."""
        age = self._card_age(obs.current.text)
        if not self.b.min_age or age is None or age >= self.b.min_age:
            return False
        seen_key = f"agegate:{obs.current.media_key or _digest(obs.current.text)}"
        skip = next((b for b in obs.current.buttons if b.clickable
                     and any(s and s in b.text for s in self.b.skip_buttons)), None)
        if skip is None or self.store.get(seen_key) is not None:
            # No skip button, or the same card came back: never loop, hand it to the owner.
            await self._execute(obs, Decision(action="ask_owner",
                                              reason=f"анкета с возрастом {age} < {self.b.min_age}, "
                                                     f"пропустить её сам не могу"), decided_by="rule")
            return True
        self.store.set(seen_key, str(self.clock()))
        await self._execute(obs, Decision(action="click", button_id=skip.id,
                                          reason=f"возраст {age} < {self.b.min_age}"),
                            decided_by="rule", epoch=epoch)
        return True

    KEYBOARD_WAIT_S = 10.0
    KEYBOARD_WAITS = 3

    def _await_keyboard(self, obs: Observation, now: float) -> bool:
        """Half a card: a picture without buttons, or rating buttons without the picture (Bibinto sends the
        keyboard first). The rest usually arrives a moment later: look again shortly, a few times."""
        has_buttons = any(b.clickable for b in obs.current.buttons)
        # Ratings without a picture count only when they came with this very message (Bibinto's "🐨"
        # keyboard precedes the photo); a text after a card under an old keyboard is not half a card.
        own_ratings = any(b.clickable and self._is_rating(obs, b.id) and (b.inline or obs.keyboard_id == obs.current.id)
                          for b in obs.current.buttons)
        half_card = (obs.image_msg is not None and not has_buttons) or (obs.image_msg is None and own_ratings)
        if not half_card:
            return False
        # One bounded record: the card being waited for and how many times (a new card resets it).
        card = f"{obs.current.id}:{obs.rev}"
        try:
            state = json.loads(self.store.get("kbwait") or "{}")
        except ValueError:
            state = {}
        tries = int(state.get("tries", 0)) if state.get("card") == card else 0
        if tries >= self.KEYBOARD_WAITS:
            return False
        self.store.set("kbwait", json.dumps({"card": card, "tries": tries + 1}))
        self.store.mark(obs.current.id, obs.rev, "waiting", now, retry_after=now + self.KEYBOARD_WAIT_S)
        return True

    KEYBOARD_LOOKBACK = 40  # messages read to find the reply keyboard still shown
    CONTINUES_PER_HOUR = 12
    SHORT_WAIT_S = 120

    async def _keep_going(self, obs: Observation, d: Decision, epoch: int) -> bool:
        """Instead of stopping for the owner, or waiting again and again on the same screen, move on to the
        profiles. Returns True if the decision was replaced by a 'continue' action."""
        cur = obs.current
        repeated_wait = False
        if d.action == "wait" and (d.wait_seconds or 0) <= self.SHORT_WAIT_S:
            try:
                last = json.loads(self.store.get("lastwait") or "{}")
            except ValueError:
                last = {}
            # The same message (id and content) waited on before: it is not going away by itself.
            repeated_wait = last.get("id") == cur.id and last.get("rev") == obs.rev
            if not repeated_wait:
                self.store.set("lastwait", json.dumps({"id": cur.id, "rev": obs.rev}))
                return False
        else:
            self.store.unset("lastwait")
        if d.action != "ask_owner" and not repeated_wait:
            return False
        if self._owner_required(cur.text) or (d.action == "ask_owner" and _OWNER_ALWAYS.search(_affirmative(d.reason))):
            return False  # e.g. a video verification or a captcha the model recognised: only a person can do it
        if self._looping():
            return False
        button = None if _CHARGE_RE.search(cur.text or "") else next(
            (b for b in cur.buttons if b.clickable and _is_safe_continue(b.text, self.b.continue_buttons)), None)
        if button is not None:
            done = await self._execute(obs, Decision(action="click", button_id=button.id, reason="продолжить"),
                                       decided_by="rule", epoch=epoch)
        elif self.b.start_command:
            done = await self._execute(obs, Decision(action="send", text=self.b.start_command, reason="продолжить"),
                                       decided_by="rule", epoch=epoch)
        else:
            return False
        if done:
            self.store.log_action(self.clock(), cur.id, obs.rev, "continue",
                                  button_text=button.text if button else self.b.start_command.split()[0],
                                  decided_by="rule", result="repeat-wait" if repeated_wait else "instead-of-stop")
        return True

    def _rates_profile(self, obs: Observation, button_id: str | None) -> bool:
        """A rating of a profile card: a rating button on a picture, not a numbered menu item next to an ad image."""
        button = obs.current.button(button_id or "")
        companion = obs.image_msg is not None and obs.image_msg.id != obs.current.id
        return bool(button and obs.image_msg is not None and self._is_rating(obs, button.id)
                    and button.text.strip() not in _menu_lines(obs.current.text)
                    # a purchase prompt right after the last profile photo is not that profile
                    and not (companion and _CHARGE_RE.search(obs.current.text or "")))

    def _may_pay(self, obs: Observation, d: Decision) -> bool:
        """A press or a reply that may confirm a purchase: a priced button, or anything but a refusal, a profile
        rating or a restart on a message that charges for something."""
        cur = obs.current
        button = cur.button(d.button_id or "") if d.action == "click" else None
        if button is not None and _is_refusal(button.text, tuple(self.b.decline_buttons)):
            return False
        if button is not None and (_PRICE_RE.search(button.text) or _BUY_RE.search(button.text)):
            return True
        if not _CHARGE_RE.search(cur.text or ""):
            return False
        # "1. Смотреть анкеты / 2. Купить Премиум": choosing the free item is fine, its own line decides.
        number = (button.text if button is not None else d.text or "").strip()
        item = brain.menu_lines(cur.text).get(number)
        if item is not None:  # "Да" / "Продолжить" under a charge question is agreeing to pay
            return bool(_CHARGE_RE.search(item) or _BUY_RE.search(item) or _PRICE_RE.search(item)
                        or _AGREE_RE.match(item))
        if d.action == "click":
            return button is not None and not self._rates_profile(obs, button.id)
        start = self.b.start_command.split()[0] if self.b.start_command else None
        return d.action == "send" and (d.text or "").split()[:1] != [start]

    def _owner_required(self, text: str | None) -> bool:
        return bool(_OWNER_ALWAYS.search(text or "") or (
            self.b.owner_required_pattern and re.search(self.b.owner_required_pattern, text or "", re.I)))

    def _looping(self) -> bool:
        """A dozen presses and commands within the last hour, whoever chose them, and not one of them rated a
        profile: the bot is going round in circles (ads, menus, restarts)."""
        return self.store.presses_since_card(self.clock() - 3600) >= self.CONTINUES_PER_HOUR

    def _is_rating(self, obs: Observation, button_id: str | None) -> bool:
        button = obs.current.button(button_id or "")
        return bool(button and self.b.rating_buttons and re.search(self.b.rating_buttons, button.text))

    # ------------------------------------------------------ reciprocal ratings
    RECIPROCAL_TTL_S = 120  # the rater's profile follows the "see who" press almost immediately

    async def _open_incoming_rating(self, obs: Observation, epoch: int) -> bool:
        """'Someone rated you N — see who?': remember N and open the profile, without the model."""
        cur = obs.current
        # The notification is text; a photo just before it (the previous card) does not make it a card.
        if not self.b.incoming_rating_pattern or cur.has_image:
            return False
        m = re.search(self.b.incoming_rating_pattern, cur.text or "", re.I)
        if not m:
            return False
        opener = next((btn for btn in cur.buttons if btn.clickable and any(
            w and w.lower() in btn.text.lower() for w in self.b.incoming_open_buttons)), None)
        if opener is None:
            return False
        try:
            score = int(m.group(1))
        except (IndexError, ValueError):
            return False
        if not 1 <= score <= 10 or self._looping() or _CHARGE_RE.search(cur.text or ""):
            return False  # "see who rated you for 10 stars": not a free look
        last_id = max((m.id for m in obs.msgs), default=cur.id)
        opened = await self._execute(obs, Decision(action="click", button_id=opener.id,
                                                   reason="посмотреть, кто оценил"), decided_by="rule", epoch=epoch)
        if opened:
            # Bound to the first profile that arrives after this press: the person who rated the owner.
            self.store.set("reciprocal", json.dumps({"score": score, "ts": self.clock(), "after": last_id}))
        return True

    def _reciprocal_score(self, obs: Observation | None = None) -> int | None:
        """The score of the person who rated the owner, if `obs` is that person's profile (the first
        picture card after the "see who" press, within the time limit)."""
        raw = self.store.get("reciprocal")
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except ValueError:
            return None
        if self.clock() - float(data.get("ts", 0)) > self.RECIPROCAL_TTL_S:
            self.store.unset("reciprocal")
            return None
        if obs is not None:
            card = obs.image_msg
            if card is None or card.id <= int(data.get("after", 0)):
                return None  # an older card: not the person who rated the owner
        return int(data.get("score", 0)) or None

    def _apply_reciprocal(self, obs: Observation, d: Decision) -> Decision:
        """A profile that fits the owner's taste gets at least the rating that person gave the owner."""
        score = self._reciprocal_score(obs)
        if score is None or d.action != "click" or not self._is_rating(obs, d.button_id):
            return d
        chosen = obs.current.button(d.button_id or "")
        if chosen is None or not chosen.text.strip().isdigit():
            return d
        own = int(chosen.text.strip())
        if own < self.b.reciprocal_min or own >= score:
            return d
        target = next((btn for btn in obs.current.buttons if btn.clickable and btn.text.strip() == str(score)
                       and self._is_rating(obs, btn.id)), None)
        if target is None:
            return d
        return replace(d, button_id=target.id, reason=f"{d.reason}; ответная оценка {score} (сам поставил бы {own})")

    def _borderline(self, d: Decision) -> bool:
        threshold = self.b.confidence_threshold
        return threshold > 0 and (d.confidence is None or d.confidence < threshold)

    async def _decline_paid_offer(self, obs: Observation, epoch: int) -> bool:
        """VIP / stars / boosts offers: a fast path that presses an explicit refusal and goes on. Without a
        refusal button the message is not necessarily an offer (menus mention premium too): the model decides,
        and its task says to decline paid features and keep browsing. Purchase buttons are deny-listed."""
        cur = obs.current
        if (not self.b.premium_pattern or cur.has_image
                or not re.search(self.b.premium_pattern, cur.text or "", re.I)):
            return False
        refusals = tuple(self.b.decline_buttons)
        decline = next((btn for btn in cur.buttons if btn.clickable and _is_refusal(btn.text, refusals)), None)
        if decline is None:
            return False
        if self.store.actions_since(self.clock() - 3600, ("decline",)) >= 5:
            await self._execute(obs, Decision(action="ask_owner", reason="слишком много платных предложений подряд"),
                                decided_by="rule")
            return True
        ok = await self._execute(obs, Decision(action="click", button_id=decline.id,
                                               reason="отказ от платной функции"), decided_by="rule", epoch=epoch)
        if ok:
            self.store.log_action(self.clock(), cur.id, obs.rev, "decline", button_text=decline.text,
                                  decided_by="rule")
        return True

    def _log_failures(self, failures) -> None:
        for f in failures:
            self.store.log_llm(self.clock(), f.model, f.prompt_tokens, f.completion_tokens, f.cost_usd, ok=False)

    async def _limits_exceeded(self, now: float) -> bool:
        spent = self.store.spend_today(now)
        if self.b.daily_budget_usd > 0 and spent >= self.b.daily_budget_usd:
            tomorrow = (datetime.fromtimestamp(now, UTC) + timedelta(days=1)).replace(
                hour=0, minute=0, second=5, microsecond=0)
            self.store.set("paused_until", str(tomorrow.timestamp()))
            await self.say(f"Дневной бюджет исчерпан (${spent:.4f} из ${self.b.daily_budget_usd}). "
                           f"Продолжу {self._time(tomorrow.timestamp())}.")
            return True
        if self.store.actions_since(now - 3600) >= self.b.max_actions_per_hour:
            self.pause_for(600)
            log.info("hourly action cap reached, pausing 10 min")
            return True
        return False

    async def _check_stuck(self, obs: Observation, status: str, now: float) -> None:
        if status != "done":
            return
        last = self.store.last_action_ts()
        if last is None or now - last < self.b.stuck_timeout_s:
            return
        cur = obs.current
        key = f"retry:{cur.id}:{obs.rev}"
        if self.store.get(key) is None:
            # The bot did not react to our last action at all: re-decide once.
            self.store.set(key, str(now))
            self.store.unmark(cur.id, obs.rev)
            if obs.image_msg is not None:
                self.store.unset(self._acted_key(obs))  # allow this one deliberate retry
            self.notes.append("Your previous action on this same message produced no visible reaction "
                              "from the bot. Decide again; ask the owner if unsure.")
            self.wake.set()
            return
        self.store.mark(cur.id, obs.rev, "stuck", now)
        self.store.set("running", "0")
        await self.say(f"Бот не реагирует на сообщение #{cur.id} больше "
                       f"{self.b.stuck_timeout_s / 60:.0f} мин даже после повтора. Остановился. "
                       f"Посмотрите чат и отправьте команду /start.")

    # ---------------------------------------------------------------- actions
    async def _still_current(self, msg_id: int, rev: str) -> Observation | None:
        """The card is still the bot's latest message and unchanged (text, buttons and picture)."""
        obs = await self._observe()
        if obs is None or obs.current.id != msg_id or obs.rev != rev:
            return None
        return obs

    async def _execute(self, obs: Observation, d: Decision, *, decided_by: str,
                       image: bytes | None = None, epoch: int | None = None) -> bool:
        msg, rev = obs.current, obs.rev
        now = self.clock()
        if d.action == "wait":
            self.store.mark(msg.id, rev, "waiting", now, retry_after=now + (d.wait_seconds or 600))
            self.store.log_action(now, msg.id, rev, "wait", text=f"{d.wait_seconds:.0f}s",
                                  reason=d.reason, confidence=d.confidence, decided_by=decided_by)
            return True
        if d.action == "ask_owner":
            self.store.mark(msg.id, rev, "stuck", now)
            self.store.set("running", "0")
            # Visible button labels (the bot's interface, not personal data) make the stop diagnosable.
            labels = " | ".join(btn.text.strip() if re.fullmatch(r"\s*(?:10|[1-9])\s*", btn.text)
                                else scrub_text(btn.text, 24) for btn in msg.buttons if btn.kind != "url")[:300] \
                or "(нет кнопок)"
            self.store.log_action(now, msg.id, rev, "ask_owner", button_text=labels,
                                  text="картинка" if obs.image_msg is not None else "текст",
                                  confidence=d.confidence, decided_by=decided_by)
            await self.say(f"Нужна ваша помощь с сообщением бота #{msg.id}: {d.reason}\n"
                           f"Я остановился. Сделайте нужное в чате с ботом сами и отправьте команду /start.",
                           image=image)
            return True

        if self.restricted:
            return False
        await self.sleep(self.rng.uniform(self.b.action_delay_min_s, self.b.action_delay_max_s))
        if self.restricted:
            log.info("account restricted during the pre-action delay; not acting on #%s", msg.id)
            return False
        # The owner's own answer is not bound to working hours or the norm, but /stop or /pause cancels it.
        allowed = (self.running and not self.restricted) if decided_by == "owner" else self.active()
        if epoch is not None and (epoch != self.epoch or not allowed):
            log.info("owner changed state during the pre-action delay; not acting on #%s", msg.id)
            self.wake.set()
            return False
        fresh = await self._still_current(msg.id, rev)
        if fresh is None:
            log.info("card #%s changed or was replaced before acting; re-evaluating", msg.id)
            self.wake.set()
            return False
        if self.restricted:  # the breaker may have tripped while the chat was re-read
            return False
        if epoch is not None and (epoch != self.epoch or not allowed):
            # The owner may have stopped us while the chat was being re-read.
            self.wake.set()
            return False

        if d.action == "click":
            button = fresh.current.button(d.button_id or "")
            if button is None or not button.clickable:
                self.wake.set()
                return False
            outcome = await self.tg.click(msg.id, button)
            if not outcome.ok:
                # Rejected by Telegram (stale message/data): log it and re-read the chat.
                self.store.log_action(self.clock(), msg.id, rev, "click", button_text=button.text,
                                      reason=d.reason, confidence=d.confidence, decided_by=decided_by,
                                      result=f"error: {outcome.answer or 'rejected'}")
                self.store.mark(msg.id, rev, "done", self.clock())
                self.wake.set()
                return False
            if outcome.timed_out:
                result = "timeout"
            elif outcome.answer:
                result = "answer"  # the text itself goes to the next prompt only, never to the database
                self.notes.append(f'After the last click the bot showed: "{outcome.answer[:300]}"')
            else:
                result = "ok"
            self.store.mark(msg.id, rev, "done", self.clock())
            rated = self._rates_profile(obs, button.id)
            self.store.log_action(self.clock(), msg.id, rev, "click", button_text=button.text,
                                  reason=d.reason, confidence=d.confidence, decided_by=decided_by,
                                  result=result, card=rated)
            if rated:
                self.store.unset("limit")  # profiles again: the limit is over
                await self._tell_norm(self.clock())
            if rated and self._reciprocal_score(obs) is not None:
                self.store.unset("reciprocal")  # that was the person who rated the owner
            if obs.image_msg is not None:
                # A hash is enough to recognise the same caption later; the profile text is not kept.
                self.store.set(self._acted_key(obs), _digest(obs.image_msg.text))
                if msg.text:
                    # A 1-10 rating button is the score itself (Bibinto); otherwise the model's estimate.
                    score = int(button.text) if rated and brain.is_rating_label(button.text) else d.score
                    self._recent_cards = [*self._recent_cards, (msg.text, score if rated else None)][-100:]
            return True

        if d.action == "send":
            await self.tg.send_to_bot(d.text or "")
            self.store.mark(msg.id, rev, "done", self.clock())
            self.store.log_action(self.clock(), msg.id, rev, "send", text=_loggable(d.text, fresh.current),
                                  reason=d.reason, confidence=d.confidence, decided_by=decided_by)
            return True
        return False

    # --------------------------------------------------------- owner cards
    def _card_lines(self, msg: BotMessage, d: Decision, image: bytes | None) -> list:
        lines: list = []
        if d.reason:
            lines.append(i(d.reason))
        if d.description:
            lines.append(Html(f"Вижу: {esc(d.description)}"))
        if image is None:
            lines.append(i("картинку загрузить не удалось — посмотрите её в чате с ботом"))
        return lines

    def _options(self, msg: BotMessage) -> tuple[list[str], list[str], Html]:
        choices = [btn.id for btn in msg.clickable_buttons]
        labels = [msg.button(c).text for c in choices]
        listing = Html(" · ".join(f"{n} {esc(t)}" for n, t in enumerate(labels, 1)))
        return choices, labels, listing

    async def _ask_owner_to_confirm(self, obs: Observation, d: Decision, image: bytes | None,
                                    epoch: int, kind: str = "confirm") -> None:
        msg = obs.current
        choices, labels, listing = self._options(msg)
        chosen = msg.button(d.button_id or "")
        conf = f" · уверенность {d.confidence:.2f}" if d.confidence is not None else ""
        head = b("Не уверен — решите вы") if kind == "unsure" else b("Новая анкета")
        lines = [head, Html(f"Я бы выбрал: {b(chosen.text if chosen else '?')}{esc(conf)}")]
        lines += self._card_lines(msg, d, image)
        lines.append(Html(f"Варианты: {listing}"))
        if kind == "unsure" and self.b.ask_timeout_min > 0:
            lines.append(i(f"Если не ответите за {self.b.ask_timeout_min:g} мин, выберу сам."))
        lines.append(i("Ответьте на это сообщение: кнопка или номер (можно с комментарием), «ок» или «пропуск»."))
        ctrl_id = await self.say(join(*lines), image=image)
        if ctrl_id is None:
            self.store.set("running", "0")
            log.error("cannot reach the owner for confirmation; stopping")
            return
        if epoch != self.epoch:
            # The owner changed something while this card was being sent: do not leave it pending.
            self.wake.set()
            return
        self.store.add_pending(Pending(ctrl_id, msg.id, obs.rev, {
            "button_id": d.button_id, "description": brain.traits_text(d.traits), "reason": "",
            "confidence": d.confidence, "score": d.score}, choices, self.clock(), kind=kind, labels=labels))
        self.store.mark(msg.id, obs.rev, "pending", self.clock())

    def _card_fits_mode(self, kind: str) -> bool:
        return {"confirm": self.mode == "supervised", "unsure": self.mode == "smart"}.get(kind, True)

    async def _maybe_time_out(self, obs: Observation, now: float, epoch: int) -> None:
        """Smart mode with ask_timeout_min > 0: an unanswered doubt is resolved with the agent's own choice."""
        pending = self.store.pending_for(obs.current.id, obs.rev)
        timeout = self.b.ask_timeout_min * 60
        if (pending is None or pending.kind != "unsure" or self.mode != "smart" or timeout <= 0
                or now - pending.ts < timeout):
            return
        self.store.delete_pending(pending.control_msg_id)
        self.store.unmark(obs.current.id, obs.rev)
        d = Decision(action="click", button_id=pending.decision.get("button_id"),
                     reason="владелец не ответил вовремя", confidence=pending.decision.get("confidence"),
                     score=pending.decision.get("score"))
        if not await self._execute(obs, d, decided_by="model", epoch=epoch):
            self.wake.set()

    def _next_timeout_at(self) -> float | None:
        p = self.store.latest_pending(("unsure",))
        if p is None or self.mode != "smart" or self.b.ask_timeout_min <= 0:
            return None
        return p.ts + self.b.ask_timeout_min * 60

    # --------------------------------------------------------- spot checks
    @staticmethod
    def _bucket(label: str) -> str:
        t = label.strip()
        if t.isdigit():
            n = int(t)
            return "low" if n <= 3 else ("high" if n >= 8 else "mid")
        if any(x in t for x in ("❤", "👍", "лайк", "like")):
            return "high"
        if any(x in t for x in ("👎", "дизлайк", "dislike", "нет")):
            return "low"
        return "mid"

    async def _maybe_spot_check(self, obs: Observation, d: Decision, image: bytes | None) -> None:
        """Sometimes ask the owner to rate a card the agent already rated (non-blocking, never clicks).
        The owner rates first; the agent's choice is revealed afterwards, so it cannot anchor the answer."""
        rate, cap = self.b.spot_check_rate, self.b.spot_checks_per_hour
        now = self.clock()
        if (rate <= 0 or cap <= 0 or image is None or self.store.pending_count("spot") >= 3
                or self.store.spots_since(now - 3600) >= cap):
            return
        chosen = obs.current.button(d.button_id or "")
        if chosen is None:
            return
        bucket = self._bucket(chosen.text)
        counts = self.store.spot_bucket_counts(now - 86400)
        fewest = min((counts.get(k, 0) for k in ("low", "mid", "high")), default=0)
        weight = 2.0 if counts.get(bucket, 0) <= fewest else 0.5  # favour under-sampled decision levels
        probability = min(1.0, rate * weight)
        if self.rng.random() >= probability:
            return
        choices, labels, listing = self._options(obs.current)
        ctrl_id = await self.say(join(
            b("Проверка вкуса"),
            "Я уже оценил эту анкету сам. А как бы оценили вы?",
            Html(f"Варианты: {listing}"),
            i("Ответьте на это сообщение кнопкой или номером. Мой выбор покажу после вашего ответа."),
        ), image=image)
        if ctrl_id is None:
            return
        self.store.add_spot(ctrl_id, now, bucket, probability, self.cfg.llm.model, d.confidence, chosen.text)
        self.store.add_pending(Pending(ctrl_id, obs.current.id, obs.rev, {
            "button_id": d.button_id, "description": brain.traits_text(d.traits), "reason": "",
            "confidence": d.confidence}, choices, now, kind="spot", labels=labels))

    # ------------------------------------------------------------------ owner
    async def handle_owner(self, om: OwnerMessage, *, from_saved_messages: bool = False) -> None:
        """Single-bot entry point (the supervisor routes for several bots): /commands and card replies."""
        cmd = parse_owner(om.text)
        if cmd is None:
            if om.reply_to and self.store.get_pending(om.reply_to) is not None:
                cmd = parse_answer(om.text)
            else:
                return  # not a command and not an answer: owner's notes are ignored
        if cmd.name == "unknown":
            await self.say(Html(f"Не понял команду. Список команд: {code('/help')}"))
            return
        await self.command(cmd, om)

    async def command(self, cmd: Command, om: OwnerMessage) -> None:
        handler = getattr(self, f"_cmd_{cmd.name}", None)
        if handler is None:
            await self.say("Эта команда обрабатывается общим процессом.")
            return
        if not await self._guarded(handler(cmd, om)):
            await self.say(f"Не удалось выполнить «{om.text[:40]}», подробности в логе.")

    async def _cmd_start(self, cmd: Command, om: OwnerMessage) -> None:
        if self.active() and self._limit() is None:  # within a limit, /start means "try the bot now"
            await self.say(join(b("Уже работает."), self.status_text()))
            return
        await self.start_quietly()
        await self.say(join(b("Запущен."), self.status_text()))

    async def start_quietly(self) -> None:
        self.epoch += 1
        self.store.set("running", "1")
        self.store.set("paused_until", "0")
        self.store.set("restricted", "")
        q = self._limit()
        if q is not None:  # the owner says the limit is over: try the bot again right away
            q.update(at=0, tried=False, now=True)
            self._save_limit(q)
        self.llm_failures = 0
        # Re-evaluate whatever we stopped on.
        obs = await self._observe()
        if obs and self.store.handled_status(obs.current.id, obs.rev, self.clock()) == "stuck":
            self.store.unmark(obs.current.id, obs.rev)
        self.wake.set()

    def stop_now(self) -> bool:
        """Synchronous: takes effect before any notification is awaited. False if already stopped."""
        was = self.running
        self.epoch += 1
        self.store.set("running", "0")
        return was

    def pause_now(self, seconds: float) -> None:
        self.epoch += 1
        self.store.set("paused_until", str(self.clock() + seconds))
        self.wake.set()

    def paused_until_text(self) -> str:
        return self._time(self.paused_until, at=False)

    async def _cmd_stop(self, cmd: Command, om: OwnerMessage) -> None:
        if not self.stop_now():
            await self.say("Уже остановлен.")
            return
        await self.say(Html(f"{b('Остановлен.')} Продолжить: {code('/start')}"))

    async def _cmd_pause(self, cmd: Command, om: OwnerMessage) -> None:
        self.pause_now(float(cmd.arg or 3600))
        await self.say(Html(f"{b('Пауза')} до {esc(self.paused_until_text())}. Продолжить раньше: {code('/start')}"))

    async def _cmd_status(self, cmd: Command, om: OwnerMessage) -> None:
        await self.say(self.status_text())

    async def _cmd_help(self, cmd: Command, om: OwnerMessage) -> None:
        await self.tg.notify_owner(help_text([self.name]))

    async def _cmd_mode(self, cmd: Command, om: OwnerMessage) -> None:
        if self.mode == cmd.arg:
            await self.say(Html(f"Режим уже {b(st.MODE_NAMES[cmd.arg])}."))
            return
        await self.apply_mode(str(cmd.arg))

    async def apply_mode(self, mode: str, announce: bool = True) -> None:
        if mode != self.mode or self.overridden(st.find("mode")) or mode != self._base.mode:
            self.set_setting(st.find("mode"), mode)
        # Cards waiting for the owner are re-decided under the new mode (spot checks stay: they only teach).
        if mode != "supervised":
            while (p := self.store.latest_pending()) is not None:
                self.store.unmark(p.bot_msg_id, p.revision)
                self.store.delete_pending(p.control_msg_id)
        self.wake.set()
        if announce:
            await self.say(Html(f"Режим: {b(st.MODE_NAMES[mode])}."))

    async def _cmd_log(self, cmd: Command, om: OwnerMessage) -> None:
        rows = self.store.recent_actions(min(int(cmd.arg or 10), 30))
        if not rows:
            await self.say("Журнал пуст.")
            return
        now = self.clock()
        names = {"click": "нажал", "send": "отправил", "wait": "ждёт", "ask_owner": "спросил вас",
                 "decline": "отказался от платного"}
        lines = [b("Последние действия")]
        for r in rows:
            what = r["button_text"] or r["text"] or ""
            by = "вы" if r["decided_by"] == "owner" else ("правило" if r["decided_by"] == "rule" else "агент")
            lines.append(Html(f"{esc(_ago(now - r['ts']))} · {esc(names.get(r['action'], r['action']))} "
                              f"{b(what) if what else ''} ({esc(by)})"))
        await self.say(join(*lines))

    async def _cmd_accept(self, cmd: Command, om: OwnerMessage) -> None:
        await self._answer_pending(cmd, om)

    async def _cmd_skip(self, cmd: Command, om: OwnerMessage) -> None:
        await self._answer_pending(cmd, om)

    async def _cmd_choose(self, cmd: Command, om: OwnerMessage) -> None:
        await self._answer_pending(cmd, om)

    async def _cmd_choose_label(self, cmd: Command, om: OwnerMessage) -> None:
        await self._answer_pending(cmd, om)

    @staticmethod
    def _owner_choice(cmd: Command, pending: Pending) -> tuple[int | None, str | None]:
        """Index into pending.choices for the owner's answer, or (None, error)."""
        labels = pending.labels
        typed = str(cmd.arg or "").strip().lower()
        if cmd.name in ("accept", "skip", "choose_label") and typed:
            for n, label in enumerate(labels):  # a button with exactly this label always wins
                if label.strip().lower() == typed:
                    return n, None
        if cmd.name in ("accept", "skip"):
            bid = pending.decision.get("button_id")
            return (pending.choices.index(bid), None) if bid in pending.choices else (None, "нет варианта агента")
        if cmd.name == "choose_label":
            want = str(cmd.arg).strip().lower()
            for n, label in enumerate(labels):
                if label.strip().lower() == want:
                    return n, None
            return None, "такой кнопки нет"
        # A number means the rating itself when the buttons are ratings ("7" in Bibinto), else a position.
        n = int(cmd.arg or 0)
        for k, label in enumerate(labels):
            if label.strip() == str(n):
                return k, None
        if 1 <= n <= len(pending.choices):
            return n - 1, None
        return None, f"номер от 1 до {len(pending.choices)}"

    async def _answer_pending(self, cmd: Command, om: OwnerMessage) -> None:
        if self.restricted:
            await self.say(f"Аккаунт ограничен ({self.restricted}), ничего не нажимаю. /start снимет блокировку.")
            return
        flood_until = self.store.get_float("flood_until")
        if self.clock() < flood_until:
            await self.say(f"Telegram просит подождать до {self._time(flood_until, at=False)}, ответьте после.")
            return
        if om.reply_to:
            pending = self.store.get_pending(om.reply_to)
            if pending is None:
                await self.say("Эта карточка уже неактуальна.")
                return
        else:
            pending = self.store.latest_pending()
            if pending is None:
                await self.say("Сейчас нет анкет, ожидающих вашего ответа.")
                return
        index, error = self._owner_choice(cmd, pending)
        if index is None:
            await self.say(f"Не понял ответ: {error}.")
            return
        button_id = pending.choices[index]
        owner_label = pending.labels[index] if index < len(pending.labels) else button_id
        model_id = pending.decision.get("button_id")
        model_label = (pending.labels[pending.choices.index(model_id)]
                       if model_id in pending.choices and pending.labels else "")

        if pending.kind == "spot":
            # The card was already rated by the agent: only learn from the answer, never click.
            self.store.delete_pending(pending.control_msg_id)
            self.store.answer_spot(pending.control_msg_id, owner_label, self.clock())
            if cmd.name != "skip":
                self.store.add_example(self.clock(), pending.bot_msg_id, pending.decision.get("description", ""),
                                       model_label, owner_label, cmd.note)
            same = owner_label == model_label
            await self.say(Html(f"Спасибо. Я поставил {b(model_label)}, вы — {b(owner_label)}"
                                + (" — совпали." if same else ". Учту.")))
            return

        async with self.lock:
            # Another answer or a mode switch may have consumed this card while we waited for the lock.
            if self.store.get_pending(pending.control_msg_id) is None:
                await self.say("На эту анкету уже ответили.")
                return
            fresh = await self._still_current(pending.bot_msg_id, pending.revision)
            if fresh is None:
                self.store.delete_pending(pending.control_msg_id)
                self.store.unmark(pending.bot_msg_id, pending.revision)
                await self.say("Эта анкета уже сменилась, пропускаю её.")
                self.wake.set()
                return
            d = Decision(action="click", button_id=button_id, reason=cmd.note or "выбор владельца",
                         confidence=1.0, score=pending.decision.get("score"))
            # Consume the card before acting, so a failure can never lead to a second press.
            self.store.delete_pending(pending.control_msg_id)
            ok = await self._execute(fresh, d, decided_by="owner", epoch=self.epoch)
            if not ok:
                self.store.unmark(pending.bot_msg_id, pending.revision)
                await self.say("Не получилось нажать: анкета сменилась во время ожидания.")
            elif cmd.name != "skip":
                self.store.add_example(self.clock(), fresh.current.id, pending.decision.get("description", ""),
                                       model_label, owner_label, cmd.note)
        self.wake.set()

    # ---------------------------------------------------------------- helpers
    def status_text(self) -> Html:
        now = self.clock()
        if self.restricted:
            state = f"аккаунт ограничен ({self.restricted})"
        elif not self.running:
            state = "остановлен"
        elif self.paused_until > now:
            state = f"пауза до {self._time(self.paused_until, at=False)}"
        elif (limit_at := self.limit_resume_at()) is not None and limit_at > now:
            state = f"лимит бота, попробую снова {self._time(limit_at)}"
        elif (resume := self.schedule_resume_at()) is not None:
            state = (f"дневная норма выполнена, продолжу {self._time(resume)}" if self._norm_done(now)
                     else f"нерабочее время ({self.b.active_hours}), начну {self._time(resume)}")
        else:
            state = "работает"
        counts = self.store.counts_today(now)
        spent = f"${counts['spend_usd']:.3f}"
        total, agree = self.store.example_stats()
        spot = self.store.spot_stats()
        last = self.store.last_action_ts()
        lines = [
            Html(f"Состояние: {b(state)} · режим {b(st.MODE_NAMES.get(self.mode, self.mode))}"),
            Html(f"Сегодня: {count(counts['rated'], 'оценка', 'оценки', 'оценок')}, "
                 f"{count(counts['clicks'] - counts['rated'], 'нажатие', 'нажатия', 'нажатий')} в меню, "
                 f"{count(counts['llm_calls'], 'запрос', 'запроса', 'запросов')} к модели, "
                 f"{b(spent)} из ${self.b.daily_budget_usd:g}"),
        ]
        if total:
            share = f"{100 * agree / total:.0f}%"
            lines.append(Html(f"Ваших ответов: {b(total)}, агент совпал в {b(share)} случаев"))
        if spot["answered"]:
            share = f"{100 * spot['agree'] / spot['answered']:.0f}%"
            mae = f", средняя разница {spot['mae']:.1f}" if spot["mae"] is not None else ""
            lines.append(Html(f"Проверки вкуса: {b(spot['answered'])} из {spot['sent']}, "
                              f"совпадение {b(share)}{esc(mae)}"))
        if self.store.latest_pending() is not None:
            lines.append(i("Ждёт вашего ответа: анкета"))
        if last:
            lines.append(Html(f"Последнее действие: {esc(_ago(now - last))}"))
        return join(*lines)

    @staticmethod
    def _read(path) -> str:
        from .storage import read_user_text
        return read_user_text(path)
