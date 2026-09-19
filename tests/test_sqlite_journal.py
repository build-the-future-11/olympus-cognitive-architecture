from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

from olympus.core.sqlite_journal import apply_durable_journal_mode
from olympus.foundry.store import FoundryStore


class _WalFailingConnection:
    """Duck-typed connection that rejects WAL enablement."""

    def __init__(self, real: sqlite3.Connection) -> None:
        self._real = real

    def execute(self, sql: str, parameters: object = ()) -> sqlite3.Cursor:
        if "journal_mode = WAL" in sql or "journal_mode=WAL" in sql:
            raise sqlite3.OperationalError("disk I/O error")
        return self._real.execute(sql, parameters)  # type: ignore[arg-type]


def test_apply_durable_journal_mode_prefers_wal(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "wal.sqlite")
    assert apply_durable_journal_mode(connection) == "wal"
    row = connection.execute("PRAGMA journal_mode").fetchone()
    assert str(row[0]).lower() == "wal"
    connection.close()


def test_apply_durable_journal_mode_falls_back_when_wal_errors(tmp_path: Path) -> None:
    real = sqlite3.connect(tmp_path / "delete.sqlite")
    assert apply_durable_journal_mode(_WalFailingConnection(real)) == "delete"  # type: ignore[arg-type]
    row = real.execute("PRAGMA journal_mode").fetchone()
    assert str(row[0]).lower() == "delete"
    real.close()


def test_foundry_store_opens_when_wal_unavailable(tmp_path: Path) -> None:
    def _force_delete(connection: sqlite3.Connection) -> str:
        connection.execute("PRAGMA journal_mode = DELETE")
        return "delete"

    with patch("olympus.foundry.store.apply_durable_journal_mode", side_effect=_force_delete):
        store = FoundryStore(tmp_path / "foundry.sqlite")
        mode = store._connection.execute("PRAGMA journal_mode").fetchone()
        assert str(mode[0]).lower() == "delete"
        store.close()
