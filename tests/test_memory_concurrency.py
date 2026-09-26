"""Event-controlled overlap on real SQLite, without sleep-based race assertions."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, RLock, Thread, current_thread
from typing import Any

import pytest

from olympus.memory.store import MemoryRecord, MemoryStore

WAIT_SECONDS = 10.0


class Worker(Thread):
    """Capture even interruptions and return them to the main test thread."""

    def __init__(self, name: str, action: Callable[[], object], finished: Event) -> None:
        super().__init__(name=name, daemon=True)
        self.action = action
        self.finished = finished
        self.result: object = None
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            self.result = self.action()
        except BaseException as error:
            self.error = error
        finally:
            self.finished.set()


class ObservedLock:
    """Signal B's acquisition attempt, while preserving a real reentrant lock.

    On the unpatched source this unused attribute has no effect and B completes
    instead. Either event lets A proceed, making both executions deterministic.
    The assertions below check persisted values and exception identities, not
    whether this particular lock implementation was used.
    """

    def __init__(self, attempted: Event) -> None:
        self.lock = RLock()
        self.attempted = attempted

    def __enter__(self) -> ObservedLock:
        if current_thread().name == "memory-B":
            self.attempted.set()
        self.lock.acquire()
        return self

    def __exit__(self, *args: object) -> None:
        self.lock.release()


def _overlap(
    first: Callable[[], object],
    second: Callable[[], object],
    entered: Event,
    release: Event,
    attempted_or_finished: Event,
) -> tuple[Worker, Worker]:
    a = Worker("memory-A", first, Event())
    b = Worker("memory-B", second, attempted_or_finished)
    a.start()
    started_b = False
    try:
        assert entered.wait(WAIT_SECONDS), "A did not reach the controlled SQLite pause"
        b.start()
        started_b = True
        assert attempted_or_finished.wait(WAIT_SECONDS), "B neither attempted nor completed"
    finally:
        release.set()
        a.join(WAIT_SECONDS)
        if started_b:
            b.join(WAIT_SECONDS)
    assert not a.is_alive(), "A failed to finish"
    assert not b.is_alive(), "B failed to finish"
    return a, b


def _open_store(path: Path, journal: str) -> MemoryStore:
    store = MemoryStore(path)
    assert store.connection.execute(f"PRAGMA journal_mode = {journal}").fetchone() == (
        journal.lower(),
    )
    assert store.connection.execute("PRAGMA synchronous").fetchone() == (2,)
    return store


def _seed(store: MemoryStore, replace: bool) -> MemoryRecord | None:
    store.put(MemoryRecord(kind="episodic", key="protected", value="keep", salience=0.1))
    previous = None
    if replace:
        previous = MemoryRecord(kind="episodic", key="target", value="old", salience=0.9)
        store.put(previous)
    return previous


def _pause_writes(
    monkeypatch: pytest.MonkeyPatch,
    entered: Event,
    release: Event,
    first_error: BaseException | None,
    second_error: BaseException | None = None,
) -> None:
    connect = sqlite3.connect

    class PausingConnection(sqlite3.Connection):
        def execute(self, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
            cursor = super().execute(sql, parameters)
            if sql.startswith("REPLACE INTO memory"):
                if current_thread().name == "memory-A":
                    entered.set()
                    if not release.wait(WAIT_SECONDS):
                        raise TimeoutError("test did not release first writer")
                    if first_error is not None:
                        raise first_error
                elif current_thread().name == "memory-B" and second_error is not None:
                    raise second_error
            return cursor

    def paused_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        return connect(*args, factory=PausingConnection, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", paused_connect)


@pytest.mark.parametrize("journal", ["WAL", "DELETE"])
@pytest.mark.parametrize("replace", [False, True], ids=["insert", "replace"])
@pytest.mark.parametrize("operation", ["put", "get", "query", "close"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_failed_writer_is_isolated_from_overlapping_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    journal: str,
    replace: bool,
    operation: str,
    error_type: type[BaseException],
) -> None:
    entered, release, attempted = Event(), Event(), Event()
    failure = error_type("reject A after its real SQLite write")
    _pause_writes(monkeypatch, entered, release, failure)
    path = tmp_path / "memory.sqlite3"
    later = MemoryRecord(kind="episodic", key="later", value="accepted", salience=0.7)
    with _open_store(path, journal) as store:
        previous = _seed(store, replace)
        # The baseline does not consult this attribute; the fixed source does.
        monkeypatch.setattr(store, "_lock", ObservedLock(attempted), raising=False)
        actions: dict[str, Callable[[], object]] = {
            "put": lambda: store.put(later),
            "get": lambda: store.get("target"),
            "query": lambda: store.query("episodic"),
            "close": store.close,
        }
        a, b = _overlap(
            lambda: store.put(MemoryRecord(kind="episodic", key="target", value="rejected")),
            actions[operation], entered, release, attempted,
        )
        assert a.error is failure
        assert b.error is None
        if operation == "get":
            assert b.result == previous
        elif operation == "query":
            assert isinstance(b.result, list)
            assert all(record.value != "rejected" for record in b.result)
            assert [record.key for record in b.result] == (
                ["target", "protected"] if replace else ["protected"]
            )
        with MemoryStore(path) as observer:
            assert observer.get("target") == previous
            assert observer.get("later") == (later if operation == "put" else None)
            protected = observer.get("protected")
            assert protected is not None and protected.value == "keep"


@pytest.mark.parametrize("journal", ["WAL", "DELETE"])
@pytest.mark.parametrize("replace", [False, True], ids=["insert", "replace"])
@pytest.mark.parametrize("operation", ["failed-put", "close"])
def test_successful_writer_survives_overlapping_rollback_or_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    journal: str,
    replace: bool,
    operation: str,
) -> None:
    entered, release, attempted = Event(), Event(), Event()
    failure = RuntimeError("reject B after its real SQLite write")
    _pause_writes(monkeypatch, entered, release, None, failure)
    path = tmp_path / "memory.sqlite3"
    accepted = MemoryRecord(kind="episodic", key="target", value="accepted")
    with _open_store(path, journal) as store:
        _seed(store, replace)
        monkeypatch.setattr(store, "_lock", ObservedLock(attempted), raising=False)
        second = (
            (lambda: store.put(MemoryRecord(kind="episodic", key="later", value="rejected")))
            if operation == "failed-put" else store.close
        )
        a, b = _overlap(lambda: store.put(accepted), second, entered, release, attempted)
        assert a.error is None
        assert b.error is (failure if operation == "failed-put" else None)
        with MemoryStore(path) as observer:
            assert observer.get("target") == accepted
            assert observer.get("later") is None
            assert observer.get("protected") is not None


@pytest.mark.parametrize("journal", ["WAL", "DELETE"])
@pytest.mark.parametrize("operation", ["get", "query"])
def test_close_cannot_interrupt_cursor_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, journal: str, operation: str
) -> None:
    entered, release, attempted = Event(), Event(), Event()
    connect = sqlite3.connect

    class PausingCursor(sqlite3.Cursor):
        def _pause(self) -> None:
            if current_thread().name == "memory-A":
                entered.set()
                if not release.wait(WAIT_SECONDS):
                    raise TimeoutError("test did not release cursor")

        def fetchone(self) -> Any:
            self._pause()
            return super().fetchone()

        def fetchall(self) -> list[Any]:
            self._pause()
            return super().fetchall()

    class CursorConnection(sqlite3.Connection):
        def execute(self, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
            if sql.startswith("SELECT kind, key, value, salience, tags FROM memory"):
                return self.cursor(factory=PausingCursor).execute(sql, parameters)
            return super().execute(sql, parameters)

    def cursor_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        return connect(*args, factory=CursorConnection, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", cursor_connect)
    with _open_store(tmp_path / "memory.sqlite3", journal) as store:
        previous = _seed(store, True)
        monkeypatch.setattr(store, "_lock", ObservedLock(attempted), raising=False)
        first = (lambda: store.get("target")) if operation == "get" else (
            lambda: store.query("episodic")
        )
        a, b = _overlap(first, store.close, entered, release, attempted)
        assert a.error is None
        assert b.error is None
        if operation == "get":
            assert a.result == previous
        else:
            assert isinstance(a.result, list)
            assert [record.key for record in a.result] == ["target", "protected"]


@pytest.mark.parametrize("journal", ["WAL", "DELETE"])
def test_separate_stores_do_not_share_a_global_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, journal: str
) -> None:
    entered, release, finished = Event(), Event(), Event()
    _pause_writes(monkeypatch, entered, release, None)
    with _open_store(tmp_path / "a.sqlite3", journal) as first_store:
        with _open_store(tmp_path / "b.sqlite3", journal) as second_store:
            a, b = _overlap(
                lambda: first_store.put(MemoryRecord(kind="episodic", key="a", value="a")),
                lambda: second_store.put(MemoryRecord(kind="episodic", key="b", value="b")),
                entered, release, finished,
            )
            assert a.error is None
            assert b.error is None
            assert first_store.get("a") is not None
            assert second_store.get("b") is not None


@pytest.mark.parametrize("journal", ["WAL", "DELETE"])
def test_shared_store_mixed_threaded_operations(tmp_path: Path, journal: str) -> None:
    start = Barrier(4, timeout=WAIT_SECONDS)
    with _open_store(tmp_path / "memory.sqlite3", journal) as store:
        def worker(number: int) -> None:
            start.wait()
            for index in range(20):
                key = f"{number}-{index}"
                record = MemoryRecord(kind="episodic", key=key, value=key, salience=index / 20)
                store.put(record)
                assert store.get(key) == record
                assert any(item.key == key for item in store.query("episodic"))

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, number) for number in range(4)]
            for future in futures:
                future.result(timeout=WAIT_SECONDS)
        assert len(store.query("episodic")) == 80
        assert not store.connection.in_transaction
    with MemoryStore(tmp_path / "memory.sqlite3") as observer:
        assert len(observer.query("episodic")) == 80


@pytest.mark.parametrize("journal", ["WAL", "DELETE"])
def test_closed_store_errors_remain_errors_and_lock_is_released(
    tmp_path: Path, journal: str
) -> None:
    store = _open_store(tmp_path / "memory.sqlite3", journal)
    store.close()
    actions: list[Callable[[], object]] = [
        lambda: store.put(MemoryRecord(kind="episodic", key="key", value="value")),
        lambda: store.get("key"),
        lambda: store.query("episodic"),
    ]
    for action in actions:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            action()
    # A different thread must also finish after the error paths above.
    worker = Worker("memory-B", store.close, Event())
    worker.start()
    worker.join(WAIT_SECONDS)
    assert not worker.is_alive()
    assert worker.error is None
