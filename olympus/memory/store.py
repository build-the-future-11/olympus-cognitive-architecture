from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Annotated

from pydantic import Field

from olympus.core.schemas import StrictModel


class MemoryRecord(StrictModel):
    kind: str = Field(min_length=1, max_length=100)
    key: str = Field(min_length=1, max_length=512)
    value: str = Field(min_length=1, max_length=1_000_000)
    salience: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(
        default_factory=list,
        max_length=64,
    )


class MemoryStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = FULL")
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

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def put(self, record: MemoryRecord) -> None:
        with self._lock:
            try:
                self.connection.execute(
                    """INSERT INTO memory (kind, key, value, salience, tags)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        kind = excluded.kind,
                        value = excluded.value,
                        salience = excluded.salience,
                        tags = excluded.tags""",
                    (
                        record.kind,
                        record.key,
                        record.value,
                        record.salience,
                        json.dumps(record.tags, ensure_ascii=False, allow_nan=False),
                    ),
                )
                self.connection.commit()
            except BaseException:
                self.connection.rollback()
                raise

    def get(self, key: str) -> MemoryRecord | None:
        with self._lock:
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
            tags=self._decode_tags(row[4]),
        )

    def query(self, kind: str) -> list[MemoryRecord]:
        with self._lock:
            rows = self.connection.execute(
                (
                    "SELECT kind, key, value, salience, tags FROM memory "
                    "WHERE kind = ? ORDER BY salience DESC, key ASC"
                ),
                (kind,),
            ).fetchall()
        return [
            MemoryRecord(
                kind=row[0],
                key=row[1],
                value=row[2],
                salience=row[3],
                tags=self._decode_tags(row[4]),
            )
            for row in rows
        ]

    @staticmethod
    def _decode_tags(value: str) -> list[str]:
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError, TypeError:
            decoded = None
        if isinstance(decoded, list) and all(isinstance(item, str) for item in decoded):
            return decoded
        return [item for item in value.split(",") if item]
