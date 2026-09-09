from __future__ import annotations

import fcntl
import json
import logging
import math
import sqlite3
import threading
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import IO
from uuid import uuid4

from olympus.core.schemas import StrictModel
from olympus.foundry.resources import WorkloadTier, assess_workload, memory_snapshot
from olympus.foundry.service import FoundryCancelled, FoundryService
from olympus.foundry.worker import run_isolated_verification

LOGGER = logging.getLogger(__name__)


class JobStatus(StrEnum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class FoundryJobSnapshot(StrictModel):
    job_id: str | None
    status: JobStatus
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    result: dict[str, object] | None = None
    admission: dict[str, object] | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat()


class JobJournal:
    """Durable latest state plus a kernel lease released automatically on process death."""

    def __init__(self, root: Path) -> None:
        self.path = root / "jobs.sqlite3"
        self.lock_path = root / "jobs.lock"
        with closing(sqlite3.connect(self.path, timeout=10)) as connection, connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS jobs "
                "(job_id TEXT PRIMARY KEY, snapshot TEXT NOT NULL)"
            )

    def acquire(self) -> IO[str]:
        handle = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            handle.close()
            raise
        return handle

    def read(self) -> FoundryJobSnapshot:
        with closing(sqlite3.connect(self.path, timeout=10)) as connection:
            row = connection.execute(
                "SELECT snapshot FROM jobs ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
        return (
            FoundryJobSnapshot.model_validate_json(row[0])
            if row else FoundryJobSnapshot(job_id=None, status=JobStatus.IDLE)
        )

    def history(self, *, limit: int = 50, offset: int = 0) -> list[FoundryJobSnapshot]:
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("job history requires limit 1..100 and non-negative offset")
        with closing(sqlite3.connect(self.path, timeout=10)) as connection:
            rows = connection.execute(
                "SELECT snapshot FROM jobs ORDER BY rowid DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [FoundryJobSnapshot.model_validate_json(row[0]) for row in rows]

    def save(self, snapshot: FoundryJobSnapshot) -> None:
        if snapshot.job_id is None:
            return
        with closing(sqlite3.connect(self.path, timeout=10)) as connection, connection:
            connection.execute(
                "INSERT INTO jobs(job_id, snapshot) VALUES (?, ?) "
                "ON CONFLICT(job_id) DO UPDATE SET snapshot=excluded.snapshot",
                (snapshot.job_id, snapshot.model_dump_json()),
            )

    def request_cancel(self) -> FoundryJobSnapshot:
        # Serialize cancellation with completion: a late cancel cannot resurrect a job.
        with closing(sqlite3.connect(self.path, timeout=10)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT snapshot FROM jobs ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            snapshot = FoundryJobSnapshot.model_validate_json(row[0]) if row else None
            if snapshot is None or snapshot.status not in {
                JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED,
            }:
                raise RuntimeError("no active foundry workload to cancel")
            snapshot = snapshot.model_copy(update={"status": JobStatus.CANCEL_REQUESTED})
            connection.execute(
                "UPDATE jobs SET snapshot=? WHERE job_id=?",
                (snapshot.model_dump_json(), snapshot.job_id),
            )
            return snapshot

    def reconcile_abandoned(self) -> FoundryJobSnapshot:
        """Reconcile only while the caller holds this journal's exclusive lease."""
        snapshot = self.read()
        if snapshot.status in {JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED}:
            snapshot = snapshot.model_copy(update={
                "status": JobStatus.FAILED,
                "finished_at": _now(),
                "error": "Worker interrupted before completion; explicit retry required.",
                "result": None,
            })
            self.save(snapshot)
        return snapshot


class FoundryJobManager:
    """One bounded local workload, with cooperative cancellation and explicit state."""

    def __init__(
        self,
        service: FoundryService,
        sample_path: Path,
        *,
        runner: Callable[[Callable[[], bool]], dict[str, object]] | None = None,
        worker_timeout_seconds: float = 300,
    ) -> None:
        if not math.isfinite(worker_timeout_seconds) or not 0 < worker_timeout_seconds <= 3600:
            raise ValueError("worker timeout must be finite and in (0, 3600] seconds")
        self.worker_timeout_seconds = worker_timeout_seconds
        self.service = service
        self.sample_path = sample_path
        self._runner = runner or self._run_service
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._shutdown = False
        self._journal = JobJournal(service.root)
        self._lease: IO[str] | None = None
        # Only the lease owner may reconcile an abandoned RUNNING record.
        try:
            recovery_lease = self._journal.acquire()
        except BlockingIOError:
            self._snapshot = self._journal.read()
        else:
            try:
                self._snapshot = self._journal.reconcile_abandoned()
            finally:
                recovery_lease.close()

    def _run_service(self, cancelled: Callable[[], bool]) -> dict[str, object]:
        return run_isolated_verification(
            self.service.root, self.sample_path, cancelled,
            timeout_seconds=self.worker_timeout_seconds,
        )

    def snapshot(self) -> FoundryJobSnapshot:
        with self._lock:
            try:
                lease = self._journal.acquire()
            except BlockingIOError:
                return self._journal.read()
            with lease:
                return self._journal.reconcile_abandoned()

    def history(self, *, limit: int = 50, offset: int = 0) -> list[FoundryJobSnapshot]:
        self.snapshot()
        return self._journal.history(limit=limit, offset=offset)

    def _cancelled(self) -> bool:
        return self._cancel.is_set() or self._journal.read().status is JobStatus.CANCEL_REQUESTED

    def start(self) -> FoundryJobSnapshot:
        with self._lock:
            if self._shutdown:
                raise RuntimeError("foundry job manager is shut down")
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("a foundry workload is already active")
            decision = assess_workload(WorkloadTier.SMALL, memory_snapshot())
            if not decision.admitted:
                raise RuntimeError(f"resource admission refused: {decision.reason}")
            try:
                self._lease = self._journal.acquire()
            except BlockingIOError as error:
                raise RuntimeError("a foundry workload is already active") from error
            try:
                self._journal.reconcile_abandoned()
            except BaseException:
                self._lease.close()
                self._lease = None
                raise
            self._cancel.clear()
            self._snapshot = FoundryJobSnapshot(
                job_id=f"job-{uuid4().hex[:16]}",
                status=JobStatus.RUNNING,
                started_at=_now(),
                admission={
                    "tier": decision.policy.tier.value,
                    "reason": decision.reason,
                    "estimated_peak_bytes": decision.policy.estimated_peak_bytes,
                    "memory": decision.snapshot.as_dict(),
                },
            )
            thread = threading.Thread(target=self._execute, daemon=True)
            self._thread = thread
            try:
                self._journal.save(self._snapshot)
                thread.start()
            except Exception as error:
                LOGGER.error(
                    "Unable to start Foundry workload (%s)",
                    type(error).__name__,
                )
                self._thread = None
                self._snapshot = self._snapshot.model_copy(
                    update={
                        "status": JobStatus.FAILED,
                        "finished_at": _now(),
                        "error": "Foundry workload could not be started.",
                    }
                )
                try:
                    self._journal.save(self._snapshot)
                finally:
                    self._lease.close()
                    self._lease = None
                raise RuntimeError("unable to start foundry workload") from error
            return self._snapshot.model_copy(deep=True)

    def _execute(self) -> None:
        result: dict[str, object] | None
        failure: str | None
        try:
            result = self._runner(self._cancelled)
            if not isinstance(result, dict) or any(
                not isinstance(key, str) for key in result
            ):
                raise TypeError("Foundry runner must return an object with string keys")
            json.dumps(result, allow_nan=False)
        except FoundryCancelled:
            status = JobStatus.CANCELLED
            result = None
            failure = "Foundry workload was cancelled."
        except TimeoutError:
            status = JobStatus.FAILED
            result = None
            failure = "Foundry workload exceeded its execution deadline."
        except Exception as error:
            LOGGER.error(
                "Foundry background workload failed (%s)",
                type(error).__name__,
            )
            status = JobStatus.FAILED
            result = None
            failure = "Foundry workload failed; inspect server logs for details."
        else:
            status = JobStatus.SUCCEEDED
            failure = None
        with self._lock:
            self._snapshot = self._snapshot.model_copy(
                update={
                    "status": status,
                    "finished_at": _now(),
                    "error": failure,
                    "result": result,
                }
            )
            try:
                self._journal.save(self._snapshot)
            finally:
                if self._lease is not None:
                    self._lease.close()
                    self._lease = None
                self._thread = None

    def cancel(self) -> FoundryJobSnapshot:
        with self._lock:
            self._snapshot = self._journal.request_cancel()
            if self._thread is not None:
                self._cancel.set()
            return self._snapshot.model_copy(deep=True)

    def retry(self) -> FoundryJobSnapshot:
        with self._lock:
            if self.snapshot().status not in {JobStatus.CANCELLED, JobStatus.FAILED}:
                raise RuntimeError("only cancelled or failed workloads can be retried")
        return self.start()

    def shutdown(self, *, timeout_seconds: float = 30.0) -> bool:
        """Stop accepting work and wait for the active cooperative job.

        The service backing this manager must not be closed until this method
        returns ``True``. A runner that ignores cancellation remains visible to
        the caller instead of racing a closed registry connection.
        """

        if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
            raise ValueError("timeout_seconds must be finite and non-negative")
        with self._lock:
            self._shutdown = True
            thread = self._thread
            if thread is not None and thread.is_alive():
                self._cancel.set()
                self._snapshot = self._snapshot.model_copy(
                    update={"status": JobStatus.CANCEL_REQUESTED}
                )
                self._journal.save(self._snapshot)
        if thread is None:
            return True
        if thread is threading.current_thread():
            return False
        thread.join(timeout_seconds)
        return not thread.is_alive()
