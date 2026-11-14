"""Owner taste profiles shared between bots: free-text file, rules added from Telegram, reference photos.

Reference photos are provided by the owner on purpose and kept locally (data/refs/<profile>/).
Photos of people seen in the bots are never stored here.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .media import to_jpeg

SCHEMA = """
CREATE TABLE IF NOT EXISTS rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS refs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    liked INTEGER NOT NULL,        -- 1 = owner likes this, 0 = owner dislikes this
    note TEXT NOT NULL DEFAULT '',
    file TEXT NOT NULL
);
"""

MAX_REFERENCE_SIDE = 768  # references are context, not the main image: keep them cheap
MAX_REFS = 200


@dataclass(frozen=True)
class Reference:
    id: int
    liked: bool
    note: str
    path: Path


class TasteProfile:
    def __init__(self, name: str, data_dir: Path, taste_file: Path | None):
        self.name = name
        self.taste_file = taste_file
        self.refs_dir = data_dir / "refs" / name
        if str(data_dir) == ":memory:":
            self.db = sqlite3.connect(":memory:", isolation_level=None)
        else:
            data_dir.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(str(data_dir / f"taste-{name}.sqlite3"), isolation_level=None)
        self.db.executescript(SCHEMA)

    # --- text --------------------------------------------------------------
    def text(self) -> str:
        parts = []
        if self.taste_file is not None:
            from .storage import read_user_text
            parts.append(read_user_text(self.taste_file))
        rules = self.rules()
        if rules:
            parts.append("Rules the owner added:\n" + "\n".join(f"- {r}" for _, r in rules))
        return "\n\n".join(p for p in parts if p)

    def rules(self) -> list[tuple[int, str]]:
        return [(r[0], r[1]) for r in self.db.execute("SELECT id, text FROM rules ORDER BY id")]

    def add_rule(self, text: str) -> int:
        return self.db.execute("INSERT INTO rules(ts, text) VALUES(?, ?)", (time.time(), text.strip())).lastrowid

    def delete_rule(self, rule_id: int) -> bool:
        return self.db.execute("DELETE FROM rules WHERE id=?", (rule_id,)).rowcount > 0

    # --- reference photos ---------------------------------------------------
    def add_reference(self, image: bytes, liked: bool, note: str = "") -> int:
        if self.db.execute("SELECT COUNT(*) FROM refs").fetchone()[0] >= MAX_REFS:
            raise ValueError(f"reference limit reached ({MAX_REFS}); delete some first")
        jpeg = to_jpeg(image, MAX_REFERENCE_SIDE, quality=85)
        self.refs_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        cur = self.db.execute("INSERT INTO refs(ts, liked, note, file) VALUES(?, ?, ?, '')",
                              (time.time(), int(liked), note.strip()))
        ref_id = cur.lastrowid
        path = self.refs_dir / f"{ref_id}.jpg"
        path.write_bytes(jpeg)
        path.chmod(0o600)
        self.db.execute("UPDATE refs SET file=? WHERE id=?", (path.name, ref_id))
        return ref_id

    def delete_reference(self, ref_id: int) -> bool:
        row = self.db.execute("SELECT file FROM refs WHERE id=?", (ref_id,)).fetchone()
        if not row:
            return False
        (self.refs_dir / row[0]).unlink(missing_ok=True)
        self.db.execute("DELETE FROM refs WHERE id=?", (ref_id,))
        return True

    def references(self) -> list[Reference]:
        return [Reference(r[0], bool(r[1]), r[2], self.refs_dir / r[3])
                for r in self.db.execute("SELECT id, liked, note, file FROM refs ORDER BY id")]

    def prompt_references(self, limit: int) -> list[Reference]:
        """The newest references, balanced between liked and disliked, that still exist on disk."""
        if limit <= 0:
            return []
        refs = [r for r in self.references() if r.path.is_file()]
        liked = [r for r in refs if r.liked][::-1]
        disliked = [r for r in refs if not r.liked][::-1]
        picked: list[Reference] = []
        while len(picked) < limit and (liked or disliked):
            for pool in (liked, disliked):
                if pool and len(picked) < limit:
                    picked.append(pool.pop(0))
        return sorted(picked, key=lambda r: r.id)

    def counts(self) -> tuple[int, int, int]:
        liked = self.db.execute("SELECT COUNT(*) FROM refs WHERE liked=1").fetchone()[0]
        disliked = self.db.execute("SELECT COUNT(*) FROM refs WHERE liked=0").fetchone()[0]
        return len(self.rules()), liked, disliked
