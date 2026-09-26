from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from olympus.memory.store import MemoryRecord, MemoryStore


@pytest.mark.parametrize("phase", ["journal", "synchronous", "schema", "commit"])
@pytest.mark.parametrize("error_type", [sqlite3.OperationalError, RuntimeError, KeyboardInterrupt])
def test_initialization_failure_closes_connection(
    tmp_path: Path, phase: str, error_type: type[BaseException]
) -> None:
    real = sqlite3.connect(tmp_path / "failure.sqlite")
    connection = Mock(wraps=real)
    error = error_type("injected initialization failure")

    def execute(sql: str, parameters: tuple[object, ...] = ()) -> sqlite3.Cursor:
        if phase == "synchronous" and "synchronous" in sql:
            raise error
        if phase == "schema" and "CREATE TABLE" in sql:
            raise error
        return real.execute(sql, parameters)

    connection.execute.side_effect = execute
    if phase == "commit":
        connection.commit.side_effect = error

    def journal(handle: sqlite3.Connection) -> str:
        if phase == "journal":
            raise error
        handle.execute("PRAGMA journal_mode = DELETE")
        return "delete"

    try:
        with (
            patch("olympus.memory.store.sqlite3.connect", return_value=connection),
            patch("olympus.memory.store.apply_durable_journal_mode", side_effect=journal),
            pytest.raises(error_type) as caught,
        ):
            MemoryStore(tmp_path / "failure.sqlite")
        assert caught.value is error
        connection.close.assert_called_once_with()
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            real.execute("SELECT 1")
    finally:
        real.close()


def test_successful_initialization_keeps_connection_usable(tmp_path: Path) -> None:
    path = tmp_path / "success.sqlite"
    record = MemoryRecord(kind="fact", key="checkpoint", value="candidate", tags=["audit"])
    with MemoryStore(path) as store:
        assert store.connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        store.put(record)
        assert store.get(record.key) == record
        assert store.query("fact") == [record]
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        store.connection.execute("SELECT 1")
    with MemoryStore(path) as reopened:
        assert reopened.get(record.key) == record


def test_initialization_still_supports_delete_fallback(tmp_path: Path) -> None:
    def force_delete(connection: sqlite3.Connection) -> str:
        connection.execute("PRAGMA journal_mode = DELETE")
        return "delete"

    with (
        patch("olympus.memory.store.apply_durable_journal_mode", side_effect=force_delete),
        MemoryStore(tmp_path / "fallback.sqlite") as store,
    ):
        assert store.connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        store.put(MemoryRecord(kind="fact", key="fallback", value="works"))
        assert store.get("fallback") is not None
