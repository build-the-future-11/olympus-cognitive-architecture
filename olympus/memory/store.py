from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import Field

from olympus.core.schemas import StrictModel


class MemoryRecord(StrictModel):
    kind: str
    key: str
    value: str
    salience: float = 0.5
    tags: list[str] = Field(default_factory=list)


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                kind TEXT NOT NULL,
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                salience REAL NOT NULL,
                tags TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def put(self, record: MemoryRecord) -> None:
        self.connection.execute(
            "REPLACE INTO memory (kind, key, value, salience, tags) VALUES (?, ?, ?, ?, ?)",
            (record.kind, record.key, record.value, record.salience, ",".join(record.tags)),
        )
        self.connection.commit()

    def get(self, key: str) -> MemoryRecord | None:
        row = self.connection.execute(
            "SELECT kind, key, value, salience, tags FROM memory WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return MemoryRecord(
            kind=row[0],
            key=row[1],
            value=row[2],
            salience=row[3],
            tags=[item for item in row[4].split(",") if item],
        )

    def query(self, kind: str) -> list[MemoryRecord]:
        rows = self.connection.execute(
            (
                "SELECT kind, key, value, salience, tags FROM memory "
                "WHERE kind = ? ORDER BY salience DESC"
            ),
            (kind,),
        ).fetchall()
        return [
            MemoryRecord(
                kind=row[0],
                key=row[1],
                value=row[2],
                salience=row[3],
                tags=[item for item in row[4].split(",") if item],
            )
            for row in rows
        ]
