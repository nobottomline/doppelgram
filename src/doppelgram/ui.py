"""Owner-facing message formatting (Telegram HTML). Every dynamic value must go through `esc`."""

from __future__ import annotations

import html


class Html(str):
    """A string that is already safe Telegram HTML."""


def esc(value) -> Html:
    return Html(html.escape(str(value), quote=False))


def b(value) -> Html:
    return Html(f"<b>{esc(value)}</b>")


def i(value) -> Html:
    return Html(f"<i>{esc(value)}</i>")


def code(value) -> Html:
    return Html(f"<code>{esc(value)}</code>")


def quote(value) -> Html:
    return Html(f"<blockquote>{esc(value)}</blockquote>")


def join(*parts, sep: str = "\n") -> Html:
    """Join already-formatted parts (Html) and plain values (escaped)."""
    return Html(sep.join(p if isinstance(p, Html) else esc(p) for p in parts if p is not None))


def field(label: str, value) -> Html:
    return Html(f"{esc(label)}: {value if isinstance(value, Html) else esc(value)}")


_TOKEN_RE = __import__("re").compile(r"\S+")  # linear: each token is visited once
# A command is a standalone word: never inside a URL ("…?next=/start") or a path.
_COMMAND_RE = __import__("re").compile(r"(?<![\w/:.?=&%#~+-])(/[a-zA-Zа-яА-ЯёЁ_]+(?: \d+)?)(?![\w/])")


def as_html(text) -> Html:
    """Plain text becomes escaped HTML (slash commands shown as code, so they never wrap at the slash);
    Html passes through unchanged."""
    if isinstance(text, Html):
        return text
    body = esc(text)

    def code_commands(segment: str) -> str:
        return _COMMAND_RE.sub(lambda m: f"<code>{m.group(1)}</code>", segment)

    # One pass: URL tokens are copied as is, the text between them gets command formatting.
    out, pos = [], 0
    for m in _TOKEN_RE.finditer(body):
        token = m.group(0)
        if "://" in token or "www." in token or "t.me/" in token:
            out += [code_commands(body[pos:m.start()]), token]
            pos = m.end()
    out.append(code_commands(body[pos:]))
    return Html("".join(out))


def plural(n: int, one: str, few: str, many: str) -> str:
    """Russian plural: 1 оценка, 2 оценки, 5 оценок."""
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def count(n: int, one: str, few: str, many: str) -> Html:
    return Html(f"<b>{int(n)}</b> {esc(plural(n, one, few, many))}")
