"""Settings the owner may change from Telegram. config.toml holds the defaults; a value set from Telegram is
stored in SQLite and overrides it until reset. Safety-relevant options (button lists, regexes, task files,
credentials) stay config-only on purpose."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

MODES = {
    "supervised": "supervised", "manual": "supervised", "ручной": "supervised", "ручная": "supervised",
    "учёба": "supervised", "учеба": "supervised",
    "smart": "smart", "умный": "smart", "умная": "smart",
    "auto": "auto", "авто": "auto", "автомат": "auto",
}
MODE_NAMES = {"supervised": "ручной", "smart": "умный", "auto": "авто"}


class SettingError(ValueError):
    pass


def _float_range(lo: float, hi: float | None = None) -> Callable[[str], float]:
    def parse(raw: str) -> float:
        try:
            v = float(raw.replace(",", "."))
        except ValueError as exc:
            raise SettingError("нужно число") from exc
        if not math.isfinite(v):
            raise SettingError("нужно обычное число")
        if v < lo or (hi is not None and v > hi):
            raise SettingError(f"нужно число от {lo:g}" + (f" до {hi:g}" if hi is not None else ""))
        return v
    return parse


def _int_range(lo: int, hi: int | None = None) -> Callable[[str], int]:
    def parse(raw: str) -> int:
        if not raw.strip().isdigit():
            raise SettingError("нужно целое число")
        v = int(raw)
        if v < lo or (hi is not None and v > hi):
            raise SettingError(f"нужно число от {lo}" + (f" до {hi}" if hi is not None else ""))
        return v
    return parse


def _mode(raw: str) -> str:
    v = MODES.get(raw.strip().lower())
    if v is None:
        raise SettingError("ручной, умный или авто")
    return v


def _age(raw: str) -> str:
    v = {"keep": "keep", "как_есть": "keep", "как есть": "keep", "18": "18", "hide": "hide",
         "скрыть": "hide", "без_цифр": "hide"}.get(raw.strip().lower())
    if v is None:
        raise SettingError("keep, 18 или hide")
    return v


def _bool(raw: str) -> bool:
    v = {"вкл": True, "on": True, "да": True, "yes": True, "true": True, "1": True,
         "выкл": False, "off": False, "нет": False, "no": False, "false": False, "0": False}.get(raw.strip().lower())
    if v is None:
        raise SettingError("вкл или выкл")
    return v


def _delay(raw: str) -> tuple[float, float]:
    m = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*[-–]\s*(\d+(?:[.,]\d+)?)\s*", raw)
    if not m:
        raise SettingError("диапазон секунд, например 3-8")
    lo, hi = (float(x.replace(",", ".")) for x in m.groups())  # the regex admits only finite numbers
    if hi < lo or hi > 600:
        raise SettingError("от меньшего к большему, не больше 600 секунд")
    return lo, hi


def _window(raw: str) -> str:
    from .schedule import format_window, parse_window
    try:
        return format_window(parse_window(raw))
    except ValueError as exc:
        raise SettingError("часы, например 10-23 или 09:30-01:00") from exc


def _tz(raw: str) -> str:
    from .schedule import parse_tz
    v = raw.strip()
    try:
        parse_tz(v)
    except ValueError as exc:
        raise SettingError("часовой пояс, например Europe/Moscow или +3") from exc
    return v


def _model(raw: str) -> str:
    v = raw.strip()
    if not re.fullmatch(r"[\w.\-/:@~]{2,120}", v):
        raise SettingError("идентификатор модели, например openai/gpt-6-luna")
    return v


@dataclass(frozen=True)
class Setting:
    key: str                     # BehaviorConfig field (bot scope) or a global key
    names: tuple[str, ...]       # what the owner can type
    scope: str                   # bot | global
    parse: Callable[[str], Any]
    help: str
    fields: tuple[str, ...] = ()  # BehaviorConfig fields it sets (defaults to (key,))

    def show(self, value) -> str:
        if self.key == "mode":
            return MODE_NAMES.get(value, str(value))
        if self.key == "action_delay":
            return f"{value[0]:g}-{value[1]:g} с"
        if isinstance(value, bool):
            return "вкл" if value else "выкл"
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)


SETTINGS: tuple[Setting, ...] = (
    Setting("mode", ("режим", "mode"), "bot", _mode,
            "ручной — каждая анкета вам; умный — сам, спрашивает при сомнении; авто — полностью сам"),
    Setting("confidence_threshold", ("порог", "threshold"), "bot", _float_range(0, 1),
            "умный режим спрашивает вас, если уверенность ниже (0–1)"),
    Setting("ask_timeout_min", ("ожидание", "timeout"), "bot", _float_range(0, 1440),
            "умный режим: через сколько минут без ответа поставить свою оценку (0 — ждать вас)"),
    Setting("spot_check_rate", ("проверки", "spot"), "bot", _float_range(0, 1),
            "доля автоматических оценок, которые агент присылает вам на проверку (0–1)"),
    Setting("spot_checks_per_hour", ("проверок_в_час", "spot_per_hour"), "bot", _int_range(0, 60),
            "не больше стольких проверок в час"),
    Setting("age_in_prompt", ("возраст", "age"), "bot", _age,
            "что модель видит из возраста: keep — как есть, 18 — заменить на 18, hide — убрать цифры"),
    Setting("hide_numbers_10_18", ("числа", "скрыть_10_18", "скрыть_10_20", "numbers"), "bot", _bool,
            "убирать из текста анкеты числа от 10 до 18 перед отправкой модели (вкл/выкл)"),
    Setting("limit_resume", ("автопродолжение", "продолжать", "resume"), "bot", _bool,
            "когда бот перестал показывать анкеты (лимит), пробовать снова самому (вкл/выкл)"),
    Setting("limit_retry_h", ("после_лимита", "limit_retry"), "bot", _float_range(0.25, 48),
            "через сколько часов после лимита пробовать снова"),
    Setting("active_hours_enabled", ("часы_работы", "active_hours_on"), "bot", _bool,
            "работать только в часы работы, ночью отдыхать (вкл/выкл)"),
    Setting("active_hours", ("часы", "hours"), "bot", _window, "часы работы, например 10-23"),
    Setting("timezone", ("пояс", "tz", "timezone"), "bot", _tz, "часовой пояс: Europe/Moscow, +3 и т. п."),
    Setting("daily_ratings_enabled", ("дневная_норма", "daily_on"), "bot", _bool,
            "не больше нормы оценок в день (вкл/выкл)"),
    Setting("daily_ratings", ("норма", "daily"), "bot", _int_range(1, 5000), "сколько анкет оценивать в день"),
    Setting("daily_budget_usd", ("бюджет", "budget"), "bot", _float_range(0, 1000),
            "дневной бюджет на модель, $ (0 — без лимита)"),
    Setting("max_actions_per_hour", ("лимит", "limit"), "bot", _int_range(1, 1000),
            "не больше стольких действий в час"),
    Setting("action_delay", ("задержка", "delay"), "bot", _delay,
            "пауза перед каждым нажатием, секунд (например 3-8)",
            fields=("action_delay_min_s", "action_delay_max_s")),
    Setting("report_every_h", ("отчёт", "отчет", "report"), "bot", _float_range(0, 168),
            "как часто присылать сводку, часов (0 — не присылать)"),
    Setting("model", ("модель", "model"), "global", _model, "модель LLM (на том же endpoint)"),
)

_BY_NAME = {name: s for s in SETTINGS for name in (s.key, *s.names)}


def find(name: str) -> Setting | None:
    return _BY_NAME.get(name.strip().lower())


def to_fields(setting: Setting, value) -> dict[str, Any]:
    """The BehaviorConfig fields a parsed value sets."""
    if setting.key == "action_delay":
        return {"action_delay_min_s": value[0], "action_delay_max_s": value[1]}
    return {setting.key: value}


def from_behavior(setting: Setting, behavior) -> Any:
    if setting.key == "action_delay":
        return (behavior.action_delay_min_s, behavior.action_delay_max_s)
    return getattr(behavior, setting.key)
