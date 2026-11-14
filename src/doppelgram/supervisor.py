"""Runs one Agent per configured bot over a single account and routes owner commands between them.

Owner input rules: a message starting with "/" is a command; a Telegram reply to one of the agent's cards is
an answer to that card; anything else is the owner's own text (notes) and is ignored.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from . import settings as st
from .agent import Agent
from .control import Command, help_text, parse_answer, parse_duration, parse_owner, parse_reference_caption
from .models import OwnerMessage
from .taste import TasteProfile
from .transport import AccountRestricted
from .ui import Html, b, code, esc, i, join

if TYPE_CHECKING:
    from .conversation import ConversationManager
    from .llm import LLMClient
    from .storage import Store

log = logging.getLogger(__name__)

ANSWERS = ("choose", "choose_label", "accept", "skip")
TASTE_COMMANDS = ("rule", "taste", "refs", "delete", "like", "dislike")
CHAT_COMMANDS = ("dialogs", "lead", "self", "chat_mode")
HINT_EVERY_S = 6 * 3600


class Supervisor:
    def __init__(self, agents: dict[str, Agent], notify, profiles: dict[str, TasteProfile] | None = None,
                 conversations: ConversationManager | None = None, *, control: Store | None = None,
                 llm: LLMClient | None = None, allowed_models: list[str] | None = None):
        """`notify(text, image=None)` sends a message to the owner chat. `control` stores account-wide
        settings set from Telegram (the LLM model)."""
        if not agents:
            raise ValueError("at least one agent is required")
        self.agents = agents
        self.notify = notify
        self.profiles = profiles or {}
        self.conversations = conversations
        self.control = control
        self.llm = llm
        self.allowed_models = list(dict.fromkeys(allowed_models or []))
        self._last_hint = 0.0
        for agent in agents.values():
            agent.on_restricted = self.restrict_all
            if conversations is not None:
                agent.on_match = conversations.on_match
        if conversations is not None:
            conversations.on_restricted = self.restrict_all
            conversations.sanitize = self._sanitize_for_bot
        self._apply_model()

    # ------------------------------------------------------------- basics
    def _sanitize_for_bot(self, bot: str, text: str) -> str:
        agent = self.agents.get(bot)
        if agent is not None:
            return agent._sanitize(text)
        # The bot was removed or renamed: unless every bot sends ages as is, hide all digits.
        if all(a.b.age_in_prompt == "keep" for a in self.agents.values()):
            return text
        from .brain import sanitize_age
        return sanitize_age(text, "hide")

    async def restrict_all(self, reason: str) -> None:
        """Account-wide circuit breaker: a Telegram restriction stops every bot at once."""
        for agent in self.agents.values():
            agent.restrict(reason)
        if self.conversations is not None:
            self.conversations.restrict(reason)
        await self.say(join(b("Аккаунт ограничен Telegram."), f"Причина: {reason}. Все боты остановлены.",
                            Html(f"Проверьте аккаунт (например, через @SpamBot); {code('/start')} снимет "
                                 f"блокировку агента.")))

    @property
    def multi(self) -> bool:
        return len(self.agents) > 1

    async def say(self, text) -> None:
        try:
            await self.notify(text)
        except AccountRestricted as exc:
            for agent in self.agents.values():
                agent.restrict(str(exc))
            if self.conversations is not None:
                self.conversations.restrict(str(exc))
        except Exception:
            log.exception("failed to notify owner")

    def on_bot_event(self, name: str) -> None:
        agent = self.agents.get(name)
        if agent:
            agent.on_bot_event()

    async def run_forever(self) -> None:
        await self.say(join(b("Doppelgram запущен."), "", self.status_text()))
        loops = [a.run_forever(announce=False) for a in self.agents.values()]
        if self.conversations is not None:
            loops.append(self.conversations.run_forever())
        await asyncio.gather(*loops)

    def status_text(self) -> Html:
        parts: list = []
        for name, agent in self.agents.items():
            if self.multi:
                parts += [b(name.capitalize()), agent.status_text(), ""]
            else:
                parts += [agent.status_text(), ""]
        if self.llm is not None:
            parts.append(Html(f"Модель: {code(self.llm.cfg.model)}"))
        c = self.conversations
        if c is not None:
            parts.append(Html(f"Переписка с мэтчами: {b('включена' if c.enabled else 'выключена')}"
                              + (f", режим {esc('авто' if c.mode == 'auto' else 'подтверждение')}" if c.enabled
                                 else "")))
        return join(*parts)

    # ------------------------------------------------------------- routing
    def _split_target(self, text: str) -> tuple[str, str | None]:
        """'pause 2ч бибинто' -> ('pause 2ч', 'бибинто') when the last word names a bot."""
        words = text.split()
        if len(words) >= 2 and words[-1].lower() in self.agents:
            return " ".join(words[:-1]), words[-1].lower()
        return text, None

    @staticmethod
    def _unexpected_tail(cmd: Command, body: str) -> str | None:
        """Arguments that are not valid for the command: most likely a mistyped bot name."""
        parts = body.split(maxsplit=1)
        rest = parts[1].strip() if len(parts) > 1 else ""
        if not rest:
            return None
        if cmd.name in ("start", "stop", "status", "help", "matches"):
            return rest
        if cmd.name == "pause" and parse_duration(rest) is None:
            return rest
        if cmd.name == "log" and not rest.isdigit():
            return rest
        return None

    async def handle_owner(self, om: OwnerMessage, *, from_saved_messages: bool = False) -> None:
        text = (om.text or "").strip()
        if om.image is not None or om.has_media:
            await self._attachment(om, from_saved_messages)
            return
        cmd = parse_owner(text)
        if cmd is None:
            await self._plain_text(om, text, from_saved_messages)
            return
        body, target = self._split_target(text.lstrip()[1:])
        cmd = parse_owner("/" + body)
        if cmd.name == "unknown":
            await self.say(Html(f"Не знаю команду {code(text.split()[0])}. Список: {code('/help')}"))
            return
        if cmd.name in ANSWERS:
            await self._route_answer(cmd, om, target)
            return
        if cmd.name in CHAT_COMMANDS:
            if self.conversations is None:
                await self.say("Переписка не настроена.")
            else:
                await self.conversations.command(cmd.name, cmd.arg)
            return
        if cmd.name in TASTE_COMMANDS:
            await self._taste_command(cmd, target)
            return
        if cmd.name in ("settings", "set", "reset"):
            await self._settings_command(cmd, target)
            return
        if (tail := self._unexpected_tail(cmd, body)) is not None:
            await self.say(Html(f"Не понял «{esc(tail)}»: это не имя бота ({esc(', '.join(self.agents))}) "
                                f"и не аргумент команды. Справка: {code('/help')}"))
            return
        if cmd.name == "help":
            await self.say(help_text(list(self.agents)))
            return
        if cmd.name == "matches":
            await self.say(self.matches_text())
            return
        if target is None and cmd.name == "status":
            await self.say(self.status_text())
            return
        agents = [self.agents[target]] if target is not None else list(self.agents.values())
        if cmd.name == "start" and target is None and self.conversations is not None:
            self.conversations.clear_restriction()
        if cmd.name in ("start", "stop", "pause") and len(agents) > 1:
            await self._broadcast_state(cmd, agents)
            return
        for agent in agents:
            await agent.command(cmd, om)

    async def _broadcast_state(self, cmd: Command, agents: list[Agent]) -> None:
        """Change every bot's state first and confirm once, so no bot keeps acting while we notify."""
        if cmd.name == "stop":
            changed = [a.name for a in agents if a.stop_now()]
            await self.say(Html(f"{b('Остановлены')}: {esc(', '.join(changed))}. Продолжить: {code('/start')}")
                           if changed else "Все боты уже остановлены.")
        elif cmd.name == "pause":
            for a in agents:
                a.pause_now(float(cmd.arg or 3600))
            await self.say(Html(f"{b('Пауза')} до {esc(agents[0].paused_until_text())} для всех ботов. "
                                f"Продолжить раньше: {code('/start')}"))
        else:  # start
            to_start = [a for a in agents if not a.active()]
            if not to_start:
                await self.say(join(b("Уже работает."), "", self.status_text()))
                return
            for a in to_start:
                await a.start_quietly()
            await self.say(join(b("Запущено."), "", self.status_text()))

    async def _plain_text(self, om: OwnerMessage, text: str, from_saved_messages: bool) -> None:
        """Not a command: an answer if it replies to one of our cards, otherwise the owner's own note."""
        if om.reply_to:
            if self.conversations is not None and self.conversations.owns(om.reply_to):
                await self.conversations.answer_card(om.reply_to, text)
                return
            agent = self._agent_with_pending(om.reply_to)
            if agent is not None:
                await agent.command(parse_answer(text), om)
                return
        if from_saved_messages:
            return
        now = datetime.now(UTC).timestamp()
        if now - self._last_hint >= HINT_EVERY_S:
            self._last_hint = now
            await self.say(Html(f"Команды начинаются с «/», список: {code('/help')}. "
                                f"Обычный текст я не читаю."))

    async def _route_answer(self, cmd: Command, om: OwnerMessage, target: str | None) -> None:
        if target is not None:
            agent = self.agents[target]
            if om.reply_to:
                owner_of_card = self._agent_with_pending(om.reply_to)
                if owner_of_card is not None and owner_of_card is not agent:
                    await self.say("Эта анкета от другого бота — уберите имя бота или ответьте на нужную.")
                    return
            await agent.command(cmd, om)
            return
        if om.reply_to:
            agent = self._agent_with_pending(om.reply_to)
            # Not pending anywhere: the first agent explains that the card is no longer current.
            await (agent or next(iter(self.agents.values()))).command(cmd, om)
            return
        waiting = [a for a in self.agents.values() if a.store.latest_pending() is not None]
        if len(waiting) > 1:
            await self.say(Html(f"Анкеты ждут ответа у нескольких ботов — ответьте реплаем на нужную или "
                                f"добавьте имя бота: {code('/ok ' + waiting[0].name)}."))
            return
        await (waiting[0] if waiting else next(iter(self.agents.values()))).command(cmd, om)

    def _agent_with_pending(self, control_msg_id: int) -> Agent | None:
        for agent in self.agents.values():
            if agent.store.get_pending(control_msg_id) is not None:
                return agent
        return None

    # ------------------------------------------------------------ settings
    def _model_setting(self) -> st.Setting:
        return st.find("model")

    def _apply_model(self) -> None:
        if self.control is None or self.llm is None:
            return
        raw = self.control.get("setting:model")
        if raw:
            model = json.loads(raw)
            if not self.allowed_models or model in self.allowed_models:
                self.llm.cfg = replace(self.llm.cfg, model=model)

    async def _settings_command(self, cmd: Command, target: str | None) -> None:
        if cmd.name == "settings":
            await self.say(self.settings_text(target))
            return
        words = str(cmd.arg or "").split()
        if words and words[0].lower() in self.agents:  # "/set бибинто режим авто"
            target, words = words[0].lower(), words[1:]
        if not words:
            await self.say(Html(f"Укажите настройку: {code('/set порог 0.8')}. Список: {code('/settings')}"))
            return
        if cmd.name == "reset" and words[0].lower() in ("all", "всё", "все"):
            for agent in self._targets(target):
                for s in st.SETTINGS:
                    if s.scope == "bot" and agent.overridden(s):
                        agent.reset_setting(s)
                await agent.apply_mode(agent.mode, announce=False)  # release cards held by the old mode
            await self.say("Настройки ботов возвращены к значениям из конфига.")
            return
        setting = st.find(words[0])
        if setting is None:
            await self.say(Html(f"Нет настройки «{esc(words[0])}». Список: {code('/settings')}"))
            return
        if cmd.name == "reset":
            await self._reset(setting, target)
            return
        raw = " ".join(words[1:])
        if not raw:
            await self.say(Html(f"{b(setting.names[0])}: {esc(setting.help)}. Пример: "
                                f"{code('/set ' + setting.names[0] + ' ...')}"))
            return
        try:
            value = setting.parse(raw)
        except st.SettingError as exc:
            await self.say(f"Не подходит значение для «{setting.names[0]}»: {exc}.")
            return
        if setting.scope == "global":
            await self._set_global(setting, value)
            return
        agents = self._targets(target)
        for agent in agents:
            if setting.key == "mode":
                await agent.apply_mode(value)
            else:
                agent.set_setting(setting, value)
        who = esc(", ".join(a.name for a in agents)) if self.multi else ""
        await self.say(Html(f"{b(setting.names[0].capitalize())}: {b(setting.show(value))}"
                            + (f" для {who}" if who else "") + "."))

    def _targets(self, target: str | None) -> list[Agent]:
        return [self.agents[target]] if target else list(self.agents.values())

    async def _set_global(self, setting: st.Setting, value) -> None:
        if setting.key == "model":
            if self.allowed_models and value not in self.allowed_models:
                await self.say(Html(f"Эта модель не разрешена в конфиге. Можно: "
                                    f"{esc(', '.join(self.allowed_models))}"))
                return
            if self.control is None or self.llm is None:
                await self.say("Смена модели недоступна в этом режиме запуска.")
                return
            self.control.set("setting:model", json.dumps(value))
            self._apply_model()
            for agent in self.agents.values():
                agent.epoch += 1
            await self.say(Html(f"Модель: {code(value)}."))

    async def _reset(self, setting: st.Setting, target: str | None) -> None:
        if setting.scope == "global":
            if self.control is not None:
                self.control.unset(f"setting:{setting.key}")
            if self.llm is not None and self.allowed_models:
                self.llm.cfg = replace(self.llm.cfg, model=self.allowed_models[0])
            await self.say("Модель возвращена к значению из конфига.")
            return
        for agent in self._targets(target):
            if setting.key == "mode":
                agent.reset_setting(setting)
                await agent.apply_mode(agent.mode, announce=False)
            else:
                agent.reset_setting(setting)
        await self.say(f"«{setting.names[0]}» возвращено к значению из конфига.")

    def settings_text(self, target: str | None) -> Html:
        lines: list = [b("Настройки")]
        for agent in self._targets(target):
            if self.multi:
                lines += ["", b(agent.name.capitalize())]
            for s in st.SETTINGS:
                if s.scope != "bot":
                    continue
                value = s.show(st.from_behavior(s, agent.b))
                mark = i(" · из лички") if agent.overridden(s) else ""
                lines.append(Html(f"{code(s.names[0])} {b(value)}{mark}"))
        if self.llm is not None:
            overridden = self.control is not None and self.control.get("setting:model") is not None
            lines += ["", Html(f"{code('модель')} {b(self.llm.cfg.model)}"
                               + (i(" · из лички") if overridden else ""))]
        lines += ["", Html(f"Изменить: {code('/set порог 0.8')} или {code('/set бибинто режим авто')}. "
                           f"Вернуть из конфига: {code('/reset порог')}, {code('/reset all')}.")]
        return join(*lines)

    # ------------------------------------------------------------------ taste
    def _profile(self, target: str | None) -> TasteProfile | None:
        if target is not None:
            return self.profiles.get(self.agents[target].b.taste_profile)
        names = {a.b.taste_profile for a in self.agents.values()}
        return self.profiles.get(names.pop()) if len(names) == 1 else None

    async def _no_profile(self) -> None:
        await self.say(f"У ботов разные профили вкуса — добавьте имя бота в конец: {', '.join(self.agents)}.")

    async def _attachment(self, om: OwnerMessage, from_saved_messages: bool) -> None:
        """A photo is a taste example only with /like or /dislike in its caption; other attachments
        (and photos without that caption) are the owner's own and ignored."""
        caption, target = self._split_target((om.text or "").strip())
        liked, note = parse_reference_caption(caption)
        if liked is None:
            return
        if om.image is None:
            await self.say("Как пример вкуса принимаю только фото до 10 МБ.")
            return
        profile = self._profile(target)
        if profile is None:
            await self._no_profile()
            return
        try:
            ref_id = profile.add_reference(om.image, liked, note)
        except ValueError as exc:
            await self.say(f"Не сохранил фото: {exc}.")
            return
        rules, likes, dislikes = profile.counts()
        await self.say(Html(f"Фото №{ref_id} сохранено как {b('нравится' if liked else 'не нравится')}. "
                            f"Примеров: {likes} нравится, {dislikes} не нравится; правил: {rules}."))

    async def _taste_command(self, cmd: Command, target: str | None) -> None:
        profile = self._profile(target)
        if profile is None:
            await self._no_profile()
            return
        if cmd.name in ("like", "dislike"):
            await self.say(Html(f"Пришлите фото с подписью {code('/' + cmd.name)} — команда работает в "
                                f"подписи к фото."))
        elif cmd.name == "rule":
            rule_id = profile.add_rule(str(cmd.arg))
            await self.say(Html(f"Правило №{rule_id} добавлено: {esc(cmd.arg)}"))
        elif cmd.name == "delete":
            kind, item = cmd.arg
            ok = profile.delete_rule(item) if kind == "rule" else profile.delete_reference(item)
            what = "Правило" if kind == "rule" else "Фото"
            await self.say(f"{what} №{item} удалено." if ok else f"{what} №{item} не найдено.")
        elif cmd.name == "refs":
            refs = profile.references()
            if not refs:
                await self.say(Html(f"Фото-примеров пока нет. Пришлите фото с подписью {code('/like')} или "
                                    f"{code('/dislike')}."))
                return
            lines = [b("Фото-примеры")]
            lines += [Html(f"№{r.id} {b('нравится' if r.liked else 'не нравится')}"
                           + (f" — {esc(r.note)}" if r.note else "")) for r in refs[-40:]]
            await self.say(join(*lines))
        else:  # taste
            rules = profile.rules()
            _, likes, dislikes = profile.counts()
            lines = [b("Вкус"), f"Фото-примеров: {likes} нравится, {dislikes} не нравится."]
            if rules:
                lines += ["", b("Правила")] + [f"№{n} {text}" for n, text in rules]
            else:
                lines.append(Html(f"Правил пока нет: {code('/rule не нравятся полные')}"))
            await self.say(join(*lines))

    def matches_text(self) -> Html:
        rows = []
        for name, agent in self.agents.items():
            for m in agent.store.matches(10):
                when = datetime.fromtimestamp(m["ts"], UTC).strftime("%d.%m %H:%M")
                contact = ", ".join(m["links"]) or "ссылка в сообщении бота"
                rows.append((m["ts"], Html(f"{esc(when)} · {b(name)} · {esc(contact)}")))
        if not rows:
            return Html("Мэтчей пока нет.")
        return join(b("Последние мэтчи"), *[r for _, r in sorted(rows, key=lambda x: -x[0])[:20]])
