"""A failed put must not leak memory or trigger effects into a later commit."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from olympus.memory.store import MemoryRecord, MemoryStore


def _open_store(path: Path, journal: str) -> MemoryStore:
    store = MemoryStore(path)
    assert store.connection.execute(f"PRAGMA journal_mode = {journal}").fetchone() == (
        journal.lower(),
    )
    store.connection.execute("PRAGMA busy_timeout = 0")
    assert store.connection.execute("PRAGMA synchronous").fetchone() == (2,)
    return store


def _seed(store: MemoryStore, replace: bool) -> MemoryRecord | None:
    store.put(MemoryRecord(kind="episodic", key="protected", value="keep"))
    previous = None
    if replace:
        previous = MemoryRecord(kind="episodic", key="target", value="original", tags=["old"])
        store.put(previous)
    return previous


def _assert_durable_recovery(store: MemoryStore, previous: MemoryRecord | None) -> None:
    later = MemoryRecord(kind="episodic", key="later", value="accepted")
    store.put(later)
    assert not store.connection.in_transaction
    assert store.get("target") == previous
    assert store.get("later") == later
    with MemoryStore(store.path) as observer:
        assert observer.get("target") == previous
        assert observer.get("later") == later
        protected = observer.get("protected")
        assert protected is not None
        assert protected.value == "keep"


@pytest.mark.parametrize("replace", [False, True], ids=["insert", "replace"])
@pytest.mark.parametrize("check_immediate", [False, True], ids=["later-commit", "immediate"])
def test_busy_commit_does_not_publish_failed_put(
    tmp_path: Path, replace: bool, check_immediate: bool
) -> None:
    # DELETE mode gives a real reader/writer COMMIT conflict, not a mocked exception.
    with _open_store(tmp_path / "memory.sqlite3", "DELETE") as store:
        previous = _seed(store, replace)
        reader = sqlite3.connect(store.path, timeout=0.0)
        try:
            reader.execute("BEGIN")
            reader.execute("SELECT * FROM memory").fetchall()
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                store.put(MemoryRecord(kind="episodic", key="target", value="rejected"))
            if check_immediate:
                assert not store.connection.in_transaction
                assert store.get("target") == previous
        finally:
            reader.close()
        _assert_durable_recovery(store, previous)


@pytest.mark.parametrize("journal", ["WAL", "DELETE"])
@pytest.mark.parametrize("replace", [False, True], ids=["insert", "replace"])
def test_statement_failure_rolls_back_trigger_side_effects(
    tmp_path: Path, journal: str, replace: bool
) -> None:
    with _open_store(tmp_path / "memory.sqlite3", journal) as store:
        previous = _seed(store, replace)
        store.connection.executescript(
            """
            CREATE TABLE trigger_effects (value TEXT NOT NULL);
            CREATE TRIGGER reject_target BEFORE INSERT ON memory
            WHEN NEW.key = 'target'
            BEGIN
                INSERT INTO trigger_effects VALUES ('must-not-survive');
                SELECT RAISE(FAIL, 'rejected by test trigger');
            END;
            """
        )
        with pytest.raises(sqlite3.IntegrityError, match="rejected by test trigger"):
            store.put(MemoryRecord(kind="episodic", key="target", value="rejected"))
        assert not store.connection.in_transaction
        assert store.connection.execute("SELECT * FROM trigger_effects").fetchall() == []
        _assert_durable_recovery(store, previous)
        with MemoryStore(store.path) as observer:
            assert observer.connection.execute("SELECT * FROM trigger_effects").fetchall() == []


@pytest.mark.parametrize("journal", ["WAL", "DELETE"])
@pytest.mark.parametrize("replace", [False, True], ids=["insert", "replace"])
def test_deferred_constraint_commit_failure_is_rolled_back(
    tmp_path: Path, journal: str, replace: bool
) -> None:
    with _open_store(tmp_path / "memory.sqlite3", journal) as store:
        previous = _seed(store, replace)
        store.connection.execute("PRAGMA foreign_keys = ON")
        store.connection.executescript(
            """
            CREATE TABLE parents (id INTEGER PRIMARY KEY);
            CREATE TABLE children (
                parent_id INTEGER REFERENCES parents(id) DEFERRABLE INITIALLY DEFERRED
            );
            CREATE TRIGGER deferred_failure AFTER INSERT ON memory
            WHEN NEW.key = 'target'
            BEGIN
                INSERT INTO children VALUES (999);
            END;
            """
        )
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
            store.put(MemoryRecord(kind="episodic", key="target", value="rejected"))
        assert not store.connection.in_transaction
        assert store.connection.execute("SELECT * FROM children").fetchall() == []
        _assert_durable_recovery(store, previous)


class InterruptingConnection(sqlite3.Connection):
    """Raise after a real write while retaining SQLite's real transaction manager."""

    failure: BaseException | None = None

    def execute(self, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
        cursor = super().execute(sql, parameters)
        if sql.startswith("REPLACE INTO memory") and self.failure is not None:
            failure, self.failure = self.failure, None
            raise failure
        return cursor


@pytest.mark.parametrize("journal", ["WAL", "DELETE"])
@pytest.mark.parametrize("replace", [False, True], ids=["insert", "replace"])
@pytest.mark.parametrize(
    "error_type", [sqlite3.OperationalError, RuntimeError, KeyboardInterrupt, SystemExit]
)
def test_exception_after_real_write_rolls_back_and_preserves_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    journal: str,
    replace: bool,
    error_type: type[BaseException],
) -> None:
    connect = sqlite3.connect

    def interrupted_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        return connect(*args, factory=InterruptingConnection, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", interrupted_connect)
    with _open_store(tmp_path / "memory.sqlite3", journal) as store:
        previous = _seed(store, replace)
        connection = store.connection
        assert isinstance(connection, InterruptingConnection)
        failure = error_type("injected after real SQLite write")
        connection.failure = failure
        with pytest.raises(error_type) as caught:
            store.put(MemoryRecord(kind="episodic", key="target", value="rejected"))
        assert caught.value is failure
        assert not connection.in_transaction
        _assert_durable_recovery(store, previous)


@pytest.mark.parametrize("journal", ["WAL", "DELETE"])
@pytest.mark.parametrize("replace", [False, True], ids=["insert", "replace"])
def test_successful_put_stays_durable_and_preserves_query_behavior(
    tmp_path: Path, journal: str, replace: bool
) -> None:
    path = tmp_path / "memory.sqlite3"
    with _open_store(path, journal) as store:
        _seed(store, replace)
        record = MemoryRecord(
            kind="episodic", key="target", value="accepted", salience=0.9, tags=["one", "two"]
        )
        store.put(record)
        store.put(MemoryRecord(kind="semantic", key="other-kind", value="separate"))
        assert not store.connection.in_transaction
        assert store.get("target") == record
        assert [item.key for item in store.query("episodic")] == ["target", "protected"]
        assert store.get("missing") is None
        assert store.query("missing") == []
        assert store.connection.execute("PRAGMA synchronous").fetchone() == (2,)
    with MemoryStore(path) as observer:
        assert observer.get("target") == record
