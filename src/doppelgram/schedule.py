"""Working hours and local days in the owner's time zone (pure functions, no I/O)."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_OFFSET_RE = re.compile(r"(?:utc|gmt)?\s*([+-])(\d{1,2})(?::?(\d{2}))?", re.I)
_WINDOW_RE = re.compile(r"\s*(\d{1,2})(?::(\d{2}))?\s*[-–]\s*(\d{1,2})(?::(\d{2}))?\s*")


def parse_tz(name: str) -> tzinfo:
    """An IANA zone ("Europe/Moscow") or a fixed offset ("+3", "UTC+05:30")."""
    raw = (name or "").strip()
    if raw.upper() in ("UTC", "GMT", "Z"):
        return UTC
    m = _OFFSET_RE.fullmatch(raw)
    if m:
        hours, minutes = int(m.group(2)), int(m.group(3) or 0)
        if hours > 14 or minutes >= 60:
            raise ValueError(f"bad UTC offset {name!r}")
        sign = 1 if m.group(1) == "+" else -1
        return timezone(sign * timedelta(hours=hours, minutes=minutes))
    try:
        return ZoneInfo(raw)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"unknown time zone {name!r}") from exc


def parse_window(text: str) -> tuple[int, int]:
    """"10:00-23:00" or "22-2" -> (start, end) in minutes after midnight; may cross midnight."""
    m = _WINDOW_RE.fullmatch(text or "")
    if not m:
        raise ValueError(f"bad hours {text!r}, expected like 10:00-23:00")
    h1, m1, h2, m2 = (int(x or 0) for x in m.groups())
    if h1 > 24 or h2 > 24 or m1 >= 60 or m2 >= 60 or (h1 == 24 and m1) or (h2 == 24 and m2):
        raise ValueError(f"bad hours {text!r}")
    start, end = (h1 * 60 + m1) % 1440, h2 * 60 + m2  # "0-24" is the whole day
    end = end if end == 1440 and start == 0 else end % 1440
    if start == end:
        raise ValueError("start and end must differ")
    return start, end


def format_window(window: tuple[int, int]) -> str:
    return "-".join(f"{m // 60:02d}:{m % 60:02d}" for m in window)


def _minute(dt: datetime) -> int:
    return dt.hour * 60 + dt.minute


def in_window(ts: float, window: tuple[int, int], tz: tzinfo) -> bool:
    start, end = window
    now = _minute(datetime.fromtimestamp(ts, tz))
    if start < end:
        return start <= now < end
    return now >= start or now < end  # crosses midnight


def next_window_start(ts: float, window: tuple[int, int], tz: tzinfo) -> float:
    """The next moment at or after `ts` that is inside the window."""
    if in_window(ts, window, tz):
        return ts
    local = datetime.fromtimestamp(ts, tz)
    for day in range(3):
        midnight = datetime(local.year, local.month, local.day, tzinfo=tz) + timedelta(days=day)
        candidate = (midnight + timedelta(minutes=window[0])).timestamp()
        if candidate > ts:
            return candidate
    return ts + 86400  # unreachable for a valid window


def day_start(ts: float, tz: tzinfo) -> float:
    local = datetime.fromtimestamp(ts, tz)
    return datetime(local.year, local.month, local.day, tzinfo=tz).timestamp()


def next_day_start(ts: float, tz: tzinfo) -> float:
    local = datetime.fromtimestamp(ts, tz)
    return (datetime(local.year, local.month, local.day, tzinfo=tz) + timedelta(days=1)).timestamp()


def format_local(ts: float, now: float, tz: tzinfo, at: bool = True) -> str:
    """"в 10:00 MSK", "завтра в 10:00 MSK", "14.11 в 10:00 MSK"; without "в" when at=False (after "до")."""
    when, today = datetime.fromtimestamp(ts, tz), datetime.fromtimestamp(now, tz)
    clock = f"{'в ' if at else ''}{when:%H:%M} {when.tzname() or ''}".strip()
    if when.date() == today.date():
        return clock
    if when.date() == today.date() + timedelta(days=1):
        return f"завтра {clock}"
    return f"{when:%d.%m} {clock}"
