"""Owner command parsing (pure functions, no I/O).

Commands always start with "/" (Russian aliases too: /старт, /стоп ...). Text without a slash is never a
command: the owner keeps notes in the same chats. Answers to a card are written as a Telegram reply to it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .settings import MODES
from .ui import Html, b, code, esc, join

# Kept for compatibility; the agent no longer prefixes its messages.
AGENT_PREFIX = ""

_ALIASES = {
    "start": {"start", "старт", "запуск"},
    "stop": {"stop", "стоп"},
    "pause": {"pause", "пауза"},
    "status": {"status", "статус"},
    "settings": {"settings", "настройки"},
    "set": {"set", "установить", "задать"},
    "reset": {"reset", "сбросить"},
    "mode": {"mode", "режим"},
    "log": {"log", "журнал"},
    "help": {"help", "помощь", "команды"},
    "accept": {"ok", "ок", "да", "yes", "+"},
    "skip": {"skip", "пропуск"},
    "rule": {"rule", "правило"},
    "taste": {"taste", "вкус", "правила"},
    "refs": {"refs", "фото", "примеры"},
    "like": {"like", "нравится"},
    "dislike": {"dislike", "не_нравится", "ненравится"},
    "delete": {"delete", "удалить"},
    "matches": {"matches", "мэтчи", "матчи"},
    "dialogs": {"dialogs", "диалоги"},
    "lead": {"lead", "веди"},
    "self": {"self", "сам"},
    "chat_mode": {"chat", "переписка"},
}
_LOOKUP = {alias: name for name, aliases in _ALIASES.items() for alias in aliases}

_UNITS = r"часов|часа|час|ч|h|минут|мин|м|m|сек|с|s|дн|д|d"  # longer alternatives first
# One "<number><unit>" token. Tokens are consumed left to right without backtracking across them,
# so parsing is linear in the input length (owner input runs on the shared event loop).
_TOKEN_RE = re.compile(rf"\s*(\d+(?:[.,]\d+)?)\s*({_UNITS})?\s*", re.I)
MAX_DURATION_CHARS = 64
_UNIT_SECONDS = {"ч": 3600, "час": 3600, "часа": 3600, "часов": 3600, "h": 3600,
                 "м": 60, "мин": 60, "минут": 60, "m": 60,
                 "с": 1, "сек": 1, "s": 1, "д": 86400, "дн": 86400, "d": 86400}


@dataclass(frozen=True)
class Command:
    name: str
    arg: object = None
    note: str = ""


def parse_duration(text: str) -> float | None:
    """'2ч', '30 м', '1h30m', '90' (minutes) -> seconds. The whole text must be a duration."""
    text = text.strip().lower()
    if not text or len(text) > MAX_DURATION_CHARS:
        return None
    total, pos = 0.0, 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if m is None:
            return None
        num, unit = m.groups()
        total += float(num.replace(",", ".")) * _UNIT_SECONDS.get(unit or "", 60)
        pos = m.end()
    return total if total > 0 else None


def is_command(text: str) -> bool:
    return (text or "").lstrip().startswith("/")


def parse_owner(text: str) -> Command | None:
    """A slash command, or None when the text is not a command (notes, chatter)."""
    if not is_command(text):
        return None
    body = text.lstrip()[1:]
    head, _, rest = body.partition(" ")
    head = head.split("@", 1)[0]  # /status@SomeBot
    return parse_command(f"{head} {rest}".strip())


def parse_command(text: str) -> Command:
    """The command body, without the leading slash."""
    text = (text or "").strip()
    if not text:
        return Command("unknown")
    head, _, rest = text.partition(" ")
    rest = rest.strip()

    if head.isdigit():  # "/7" answers the latest card
        return Command("choose", int(head), note=rest)

    name = _LOOKUP.get(head.lower())
    if name is None:
        return Command("unknown", text)
    if name == "pause":
        return Command("pause", parse_duration(rest) or 3600.0)
    if name == "mode":
        mode = MODES.get(rest.lower())
        return Command("mode", mode) if mode else Command("unknown", text)
    if name == "log":
        return Command("log", int(rest) if rest.isdigit() else 10)
    if name in ("accept", "skip"):
        return Command(name, None, note=rest)
    if name in ("lead", "self"):
        return Command(name, int(rest)) if rest.isdigit() else Command("unknown", text)
    if name == "chat_mode":
        arg = {"вкл": "on", "on": "on", "выкл": "off", "off": "off", "авто": "auto", "auto": "auto",
               "подтверждение": "approve", "approve": "approve", "черновики": "approve"}.get(rest.lower())
        return Command("chat_mode", arg) if arg else Command("unknown", text)
    if name == "rule":
        return Command("rule", rest) if rest else Command("unknown", text)
    if name in ("like", "dislike"):
        return Command(name, None, note=rest)
    if name == "delete":
        what, _, num = rest.partition(" ")
        kind = {"правило": "rule", "rule": "rule", "фото": "ref", "ref": "ref", "пример": "ref"}.get(what.lower())
        if kind and num.strip().isdigit():
            return Command("delete", (kind, int(num.strip())))
        return Command("unknown", text)
    if name in ("set", "reset"):
        return Command(name, rest) if rest else Command("unknown", text)
    return Command(name, rest or None)


_ACCEPT = {"ок", "ok", "окей", "+", "да", "yes", "согласен"}
_SKIP = {"пропуск", "skip"}


def parse_answer(text: str) -> Command:
    """A reply to a card: "7", "7 руки кривые", "❤️", "ок", "пропуск"."""
    text = (text or "").strip()
    head, _, rest = text.partition(" ")
    low = head.lower()
    if head.isdigit():
        return Command("choose", int(head), note=rest.strip())
    if low in _ACCEPT:  # arg keeps the word: if a button has exactly this label, the button wins
        return Command("accept", head, note=rest.strip())
    if low in _SKIP:
        return Command("skip", head, note=rest.strip())
    return Command("choose_label", head, note=rest.strip())


def parse_reference_caption(caption: str) -> tuple[bool | None, str]:
    """A photo becomes a taste example only with an explicit command in its caption:
    '/like улыбка' -> (True, 'улыбка'); '/dislike' -> (False, ''); anything else -> (None, '')."""
    cmd = parse_owner(caption or "")
    if cmd is None or cmd.name not in ("like", "dislike"):
        return None, ""
    return cmd.name == "like", cmd.note


def help_text(bots: list[str]) -> Html:
    example = bots[0] if bots else "бот"
    lines = [
        b("Управление"),
        Html(f"{code('/start')} · {code('/stop')} · {code('/pause 2ч')} — запуск, остановка, пауза"),
        Html(f"{code('/status')} — состояние, траты, точность"),
        Html(f"{code('/mode ручной | умный | авто')} — как оценивать анкеты"),
        Html(f"{code('/settings')} · {code('/set порог 0.8')} · {code('/reset порог')} — настройки"),
        Html(f"{code('/log')} · {code('/matches')} — действия и мэтчи"),
        "",
        b("Режимы"),
        Html(f"{b('ручной')} — каждая анкета приходит вам"),
        Html(f"{b('умный')} — оценивает сам, при сомнении спрашивает вас"),
        Html(f"{b('авто')} — полностью сам; иногда присылает анкету на проверку вкуса"),
        "",
        b("Ответы на анкеты"),
        "Ответьте реплаем на сообщение агента: кнопка или номер (можно с комментарием), «ок» — согласиться, "
        "«пропуск» — оставить выбор агента без примера. Без реплая: /ok, /skip, /7.",
        "",
        b("Вкус"),
        Html(f"Фото с подписью {code('/like')} или {code('/dislike')} (можно с комментарием) — пример вкуса"),
        Html(f"{code('/rule не нравятся полные')} · {code('/taste')} · {code('/refs')} · "
             f"{code('/delete правило 3')} · {code('/delete фото 5')}"),
        "",
        b("Переписка с мэтчами"),
        Html(f"{code('/chat вкл | выкл | подтверждение | авто')} · {code('/dialogs')} · "
             f"{code('/lead 123')} · {code('/self 123')}"),
        "",
        Html(f"Обычный текст без «/» агент не читает — можно писать заметки. Команда для одного бота: "
             f"имя бота в конце, например {code(f'/stop {example}')}."),
    ]
    if len(bots) > 1:
        lines.append(Html(f"Боты: {esc(', '.join(bots))}."))
    return join(*lines)
