"""SQLite state: run flags, handled revisions, audit log, owner examples, spend."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS handled (
    msg_id INTEGER NOT NULL,
    revision TEXT NOT NULL,
    ts REAL NOT NULL,
    status TEXT NOT NULL,          -- done | pending | waiting | stuck
    retry_after REAL,              -- re-decide this revision after this time
    PRIMARY KEY (msg_id, revision)
);
CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    msg_id INTEGER,
    revision TEXT,
    action TEXT NOT NULL,
    button_text TEXT,
    text TEXT,
    reason TEXT,
    confidence REAL,
    decided_by TEXT,               -- model | owner | rule
    result TEXT,
    card INTEGER NOT NULL DEFAULT 0  -- 1: a decision on a profile card (a rating), 0: navigation
);
CREATE TABLE IF NOT EXISTS examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    msg_id INTEGER,
    description TEXT,
    model_choice TEXT,
    owner_choice TEXT NOT NULL,
    note TEXT
);
CREATE TABLE IF NOT EXISTS pending (
    control_msg_id INTEGER PRIMARY KEY,
    bot_msg_id INTEGER NOT NULL,
    revision TEXT NOT NULL,
    decision TEXT NOT NULL,        -- JSON of the model decision
    choices TEXT NOT NULL,         -- JSON list of offered button ids, 1-based for the owner
    ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    bot_msg_id INTEGER NOT NULL UNIQUE,
    text TEXT NOT NULL,
    links TEXT NOT NULL,           -- JSON list of URLs / @usernames
    mention_ids TEXT NOT NULL,     -- JSON list of user ids
    status TEXT NOT NULL DEFAULT 'new'
);
CREATE TABLE IF NOT EXISTS spot_checks (
    control_msg_id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    bucket TEXT NOT NULL,          -- low | mid | high (of the agent's decision)
    probability REAL NOT NULL,     -- selection probability (for unbiased aggregates)
    model TEXT,
    confidence REAL,
    model_label TEXT NOT NULL,
    owner_label TEXT,              -- set when the owner answers
    answered_ts REAL
);
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    day TEXT NOT NULL,
    model TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cost_usd REAL NOT NULL,
    ok INTEGER NOT NULL
);
"""


def scrub_text(text: str, limit: int = 500) -> str:
    """Remove digits (ages, heights, phone fragments) from derived text before storing it."""
    out = re.sub(r"\d+", "", text or "")
    out = re.sub(r"\s*,(\s*,)+", ",", out)
    out = re.sub(r"[ \t]+([,.!?;:])", r"\1", out)
    return re.sub(r"[ \t]{2,}", " ", out).strip()[:limit]


def _only_traits(text: str) -> str:
    from .brain import parse_traits_text, traits_text
    return traits_text(parse_traits_text(text or ""))


def read_user_text(path) -> str:
    """An owner-edited text file (taste, task, about-me, style). The leading block of "#" lines is a
    comment the model never sees (templates explain themselves there); everything after it is kept as is,
    including Markdown headings and hashtags."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    start = 0
    while start < len(lines) and (not lines[start].strip() or lines[start].lstrip().startswith("#")):
        start += 1
    return "\n".join(lines[start:]).strip()


def utc_day(ts: float | None = None) -> str:
    return datetime.fromtimestamp(ts if ts is not None else time.time(), UTC).strftime("%Y-%m-%d")


@dataclass
class Pending:
    control_msg_id: int
    bot_msg_id: int
    revision: str
    decision: dict
    choices: list[str]
    ts: float
    kind: str = "confirm"          # confirm (manual mode) | unsure (smart mode) | spot (after-the-fact check)
    labels: list[str] = field(default_factory=list)  # button texts aligned with `choices`


class Store:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns introduced after a database was created."""
        for table, column, ddl in (("pending", "kind", "TEXT NOT NULL DEFAULT 'confirm'"),
                                   ("pending", "labels", "TEXT NOT NULL DEFAULT '[]'"),
                                   ("actions", "card", "INTEGER NOT NULL DEFAULT 0")):
            cols = {r[1] for r in self.db.execute(f"PRAGMA table_info({table})")}
            if column not in cols:
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
        self._scrub_v1()

    def _scrub_v1(self) -> None:
        """One-time cleanup of data older versions kept but never needed: profile captions inside
        bookkeeping keys, model explanations, match notice texts, digits in example descriptions."""
        if self.get("privacy_scrub") == "6":
            return
        from .brain import parse_traits_text, traits_text
        self.db.execute("DELETE FROM kv WHERE key LIKE 'acted:%' OR key LIKE 'agegate:%'")
        self.db.execute("UPDATE actions SET reason = '' WHERE reason IS NOT NULL AND reason != ''")
        self.db.execute("UPDATE matches SET text = ''")
        self.db.execute("UPDATE actions SET result = 'answer' WHERE result LIKE 'answer:%'")
        for row_id, raw in self.db.execute("SELECT control_msg_id, decision FROM pending").fetchall():
            try:
                decision = json.loads(raw)
            except ValueError:
                continue
            decision.update(description=scrub_text(decision.get("description") or ""), reason="")
            self.db.execute("UPDATE pending SET decision = ? WHERE control_msg_id = ?",
                            (json.dumps(decision), row_id))
        for row_id, desc in self.db.execute("SELECT id, description FROM examples").fetchall():
            self.db.execute("UPDATE examples SET description = ? WHERE id = ?", (scrub_text(desc or ""), row_id))
        # Free-text descriptions of earlier versions may hold names or retold profiles: examples keep only
        # the owner's choices (new examples store fixed-vocabulary traits).
        for row_id, desc, note in self.db.execute("SELECT id, description, note FROM examples").fetchall():
            kept = traits_text(parse_traits_text(desc or ""))  # rebuilt from valid traits, never copied
            self.db.execute("UPDATE examples SET description = ?, note = ? WHERE id = ?",
                            (kept, scrub_text(note or ""), row_id))
        for row_id, raw in self.db.execute("SELECT control_msg_id, decision FROM pending").fetchall():
            try:
                decision = json.loads(raw)
            except ValueError:
                continue
            rebuilt = traits_text(parse_traits_text(decision.get("description") or ""))
            if rebuilt != decision.get("description"):
                decision["description"] = rebuilt
                self.db.execute("UPDATE pending SET decision = ? WHERE control_msg_id = ?",
                                (json.dumps(decision), row_id))
        for row_id, text in self.db.execute("SELECT id, text FROM actions WHERE action = 'send' "
                                            "AND text IS NOT NULL").fetchall():
            t = text.strip()
            kept = t.split()[0] if t.startswith("/") else (t if t.isdigit() and len(t) <= 3 else "(сообщение)")
            self.db.execute("UPDATE actions SET text = ? WHERE id = ?", (kept, row_id))
        self.set("privacy_scrub", "6")

    # --- key/value -------------------------------------------------------
    def get(self, key: str, default: str | None = None) -> str | None:
        row = self.db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set(self, key: str, value: str) -> None:
        self.db.execute("INSERT INTO kv(key, value) VALUES(?, ?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    def unset(self, key: str) -> None:
        self.db.execute("DELETE FROM kv WHERE key=?", (key,))

    def get_float(self, key: str, default: float = 0.0) -> float:
        v = self.get(key)
        return float(v) if v is not None else default

    # --- handled revisions ----------------------------------------------
    def handled_status(self, msg_id: int, revision: str, now: float) -> str | None:
        row = self.db.execute("SELECT status, retry_after FROM handled WHERE msg_id=? AND revision=?",
                              (msg_id, revision)).fetchone()
        if not row:
            return None
        status, retry_after = row
        if retry_after is not None and now >= retry_after:
            return None
        return status

    def mark(self, msg_id: int, revision: str, status: str, now: float, retry_after: float | None = None) -> None:
        self.db.execute(
            "INSERT INTO handled(msg_id, revision, ts, status, retry_after) VALUES(?,?,?,?,?) "
            "ON CONFLICT(msg_id, revision) DO UPDATE SET ts=excluded.ts, status=excluded.status, "
            "retry_after=excluded.retry_after",
            (msg_id, revision, now, status, retry_after),
        )

    def unmark(self, msg_id: int, revision: str) -> None:
        self.db.execute("DELETE FROM handled WHERE msg_id=? AND revision=?", (msg_id, revision))

    def next_retry_at(self, now: float) -> float | None:
        row = self.db.execute("SELECT MIN(retry_after) FROM handled WHERE retry_after > ?", (now,)).fetchone()
        return row[0] if row and row[0] is not None else None

    # --- audit -------------------------------------------------------------
    def log_action(self, now: float, msg_id: int | None, revision: str | None, action: str, *,
                   button_text: str | None = None, text: str | None = None, reason: str = "",
                   confidence: float | None = None, decided_by: str = "model", result: str = "",
                   card: bool = False) -> None:
        """The audit log keeps what was done, not why: the model's explanation may describe a person, so it
        is not stored (`reason` is accepted for call compatibility and dropped)."""
        self.db.execute(
            "INSERT INTO actions(ts, msg_id, revision, action, button_text, text, reason, confidence, "
            "decided_by, result, card) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (now, msg_id, revision, action, button_text, text, "", confidence, decided_by, result, int(card)),
        )

    def recent_actions(self, limit: int) -> list[dict]:
        cur = self.db.execute(
            "SELECT ts, msg_id, action, button_text, text, reason, confidence, decided_by, result "
            "FROM actions ORDER BY id DESC LIMIT ?", (limit,))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]

    def actions_since(self, ts: float, kinds: tuple[str, ...] = ("click", "send")) -> int:
        q = "SELECT COUNT(*) FROM actions WHERE ts >= ? AND action IN ({})".format(",".join("?" * len(kinds)))
        return self.db.execute(q, (ts, *kinds)).fetchone()[0]

    def cards_since(self, since: float) -> int:
        """Profiles rated since `since`."""
        return self.db.execute("SELECT COUNT(*) FROM actions WHERE ts >= ? AND action = 'click' AND card = 1",
                               (since,)).fetchone()[0]

    def presses_since_card(self, since: float) -> int:
        """Presses and sends since `since` after the last rating of a profile card."""
        last = self.db.execute("SELECT MAX(id) FROM actions WHERE ts >= ? AND card = 1", (since,)).fetchone()[0]
        return self.db.execute("SELECT COUNT(*) FROM actions WHERE ts >= ? AND id > ? AND action IN ('click','send')",
                               (since, last or 0)).fetchone()[0]

    def count_actions(self, since: float, action: str, text: str) -> int:
        return self.db.execute("SELECT COUNT(*) FROM actions WHERE ts >= ? AND action = ? AND text = ?",
                               (since, action, text)).fetchone()[0]

    def last_action_ts(self) -> float | None:
        row = self.db.execute("SELECT MAX(ts) FROM actions WHERE action IN ('click','send')").fetchone()
        return row[0]

    # --- owner examples ----------------------------------------------------
    def add_example(self, now: float, msg_id: int, description: str, model_choice: str,
                    owner_choice: str, note: str = "") -> None:
        description = _only_traits(description)  # validated here too: the store never keeps free text
        note = scrub_text(note)  # the owner's own words are kept to teach the agent, but never an age
        self.db.execute(
            "INSERT INTO examples(ts, msg_id, description, model_choice, owner_choice, note) VALUES(?,?,?,?,?,?)",
            (now, msg_id, description, model_choice, owner_choice, note),
        )

    def recent_examples(self, limit: int) -> list[dict]:
        cur = self.db.execute("SELECT description, model_choice, owner_choice, note FROM examples "
                              "ORDER BY id DESC LIMIT ?", (limit,))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r, strict=True)) for r in reversed(cur.fetchall())]

    def example_stats(self) -> tuple[int, int]:
        total, agree = self.db.execute(
            "SELECT COUNT(*), COALESCE(SUM(model_choice = owner_choice), 0) FROM examples").fetchone()
        return total, agree

    # --- pending owner confirmations ---------------------------------------
    def add_pending(self, p: Pending) -> None:
        p.decision = {**p.decision, "description": _only_traits(p.decision.get("description") or ""), "reason": ""}
        self.db.execute(
            "INSERT OR REPLACE INTO pending(control_msg_id, bot_msg_id, revision, decision, choices, ts, kind, "
            "labels) VALUES(?,?,?,?,?,?,?,?)",
            (p.control_msg_id, p.bot_msg_id, p.revision, json.dumps(p.decision), json.dumps(p.choices), p.ts,
             p.kind, json.dumps(p.labels, ensure_ascii=False)))

    _PENDING_COLS = "control_msg_id, bot_msg_id, revision, decision, choices, ts, kind, labels"

    def _pending(self, row) -> Pending | None:
        if not row:
            return None
        return Pending(row[0], row[1], row[2], json.loads(row[3]), json.loads(row[4]), row[5], row[6],
                       json.loads(row[7]))

    def get_pending(self, control_msg_id: int) -> Pending | None:
        return self._pending(self.db.execute(f"SELECT {self._PENDING_COLS} FROM pending WHERE control_msg_id=?",
                                             (control_msg_id,)).fetchone())

    def latest_pending(self, kinds: tuple[str, ...] = ("confirm", "unsure")) -> Pending | None:
        q = (f"SELECT {self._PENDING_COLS} FROM pending WHERE kind IN ({','.join('?' * len(kinds))}) "
             "ORDER BY ts DESC LIMIT 1")
        return self._pending(self.db.execute(q, kinds).fetchone())

    def pending_count(self, kind: str) -> int:
        return self.db.execute("SELECT COUNT(*) FROM pending WHERE kind=?", (kind,)).fetchone()[0]

    def pending_for(self, bot_msg_id: int, revision: str) -> Pending | None:
        return self._pending(self.db.execute(
            f"SELECT {self._PENDING_COLS} FROM pending WHERE bot_msg_id=? AND revision=? AND kind != 'spot' "
            "ORDER BY ts DESC LIMIT 1", (bot_msg_id, revision)).fetchone())

    def expire_pending(self, kind: str, before: float) -> None:
        self.db.execute("DELETE FROM pending WHERE kind=? AND ts < ?", (kind, before))

    # --- spot checks -----------------------------------------------------------
    def add_spot(self, control_msg_id: int, now: float, bucket: str, probability: float, model: str | None,
                 confidence: float | None, model_label: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO spot_checks(control_msg_id, ts, bucket, probability, model, "
                        "confidence, model_label) VALUES(?,?,?,?,?,?,?)",
                        (control_msg_id, now, bucket, probability, model, confidence, model_label))

    def answer_spot(self, control_msg_id: int, owner_label: str, now: float) -> None:
        self.db.execute("UPDATE spot_checks SET owner_label=?, answered_ts=? WHERE control_msg_id=?",
                        (owner_label, now, control_msg_id))

    def spot_row(self, control_msg_id: int) -> dict | None:
        r = self.db.execute("SELECT bucket, model_label, owner_label, confidence FROM spot_checks "
                            "WHERE control_msg_id=?", (control_msg_id,)).fetchone()
        return None if r is None else {"bucket": r[0], "model_label": r[1], "owner_label": r[2], "confidence": r[3]}

    def spots_since(self, ts: float) -> int:
        return self.db.execute("SELECT COUNT(*) FROM spot_checks WHERE ts >= ?", (ts,)).fetchone()[0]

    def spot_bucket_counts(self, since: float) -> dict[str, int]:
        return {b: n for b, n in self.db.execute(
            "SELECT bucket, COUNT(*) FROM spot_checks WHERE ts >= ? GROUP BY bucket", (since,))}

    def spot_stats(self) -> dict:
        rows = self.db.execute("SELECT model_label, owner_label FROM spot_checks WHERE owner_label IS NOT NULL"
                               ).fetchall()
        sent = self.db.execute("SELECT COUNT(*) FROM spot_checks").fetchone()[0]
        agree = sum(1 for m, o in rows if m == o)
        diffs = [abs(int(m) - int(o)) for m, o in rows if m.isdigit() and o.isdigit()]
        return {"sent": sent, "answered": len(rows), "agree": agree,
                "mae": (sum(diffs) / len(diffs)) if diffs else None}

    def delete_pending(self, control_msg_id: int) -> None:
        self.db.execute("DELETE FROM pending WHERE control_msg_id=?", (control_msg_id,))

    def clear_pending(self) -> None:
        self.db.execute("DELETE FROM pending")

    # --- matches ---------------------------------------------------------------
    def add_match(self, now: float, bot_msg_id: int, text: str, links: list[str], mention_ids: list[int]) -> bool:
        # Only the contact is kept; the notice text (with the person's name) stays in Telegram.
        cur = self.db.execute(
            "INSERT OR IGNORE INTO matches(ts, bot_msg_id, text, links, mention_ids) VALUES(?,?,?,?,?)",
            (now, bot_msg_id, "", json.dumps(links, ensure_ascii=False), json.dumps(mention_ids)))
        return cur.rowcount > 0

    def matches(self, limit: int = 20) -> list[dict]:
        cur = self.db.execute("SELECT id, ts, text, links, mention_ids, status FROM matches "
                              "ORDER BY id DESC LIMIT ?", (limit,))
        return [{"id": r[0], "ts": r[1], "text": r[2], "links": json.loads(r[3]),
                 "mention_ids": json.loads(r[4]), "status": r[5]} for r in cur.fetchall()]

    # --- LLM spend -----------------------------------------------------------
    def log_llm(self, now: float, model: str, prompt_tokens: int, completion_tokens: int,
                cost: float, ok: bool) -> None:
        self.db.execute(
            "INSERT INTO llm_calls(ts, day, model, prompt_tokens, completion_tokens, cost_usd, ok) "
            "VALUES(?,?,?,?,?,?,?)",
            (now, utc_day(now), model, prompt_tokens, completion_tokens, cost, int(ok)),
        )

    def spend_today(self, now: float) -> float:
        return self.db.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM llm_calls WHERE day=?",
                               (utc_day(now),)).fetchone()[0]

    def counts_today(self, now: float) -> dict:
        day_start = datetime.strptime(utc_day(now), "%Y-%m-%d").replace(tzinfo=UTC).timestamp()
        clicks = self.actions_since(day_start, ("click",))
        rated = self.db.execute("SELECT COUNT(*) FROM actions WHERE ts >= ? AND action = 'click' AND card = 1",
                                (day_start,)).fetchone()[0]
        calls = self.db.execute("SELECT COUNT(*) FROM llm_calls WHERE day=?", (utc_day(now),)).fetchone()[0]
        return {"clicks": clicks, "rated": rated, "llm_calls": calls, "spend_usd": self.spend_today(now)}
