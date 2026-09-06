from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from olympus.core.schemas import StrictModel
from olympus.foundry.resources import WorkloadTier, assess_workload, memory_snapshot
from olympus.foundry.service import FoundryCancelled, FoundryService


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


class FoundryJobManager:
    """One bounded local workload, with cooperative cancellation and explicit state."""

    def __init__(
        self,
        service: FoundryService,
        sample_path: Path,
        *,
        runner: Callable[[Callable[[], bool]], dict[str, object]] | None = None,
    ) -> None:
        self.service = service
        self.sample_path = sample_path
        self._runner = runner or self._run_service
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._snapshot = FoundryJobSnapshot(job_id=None, status=JobStatus.IDLE)

    def _run_service(self, cancelled: Callable[[], bool]) -> dict[str, object]:
        return self.service.run_verification_pipeline(
            self.sample_path, cancelled=cancelled
        ).model_dump(mode="json")

    def snapshot(self) -> FoundryJobSnapshot:
        with self._lock:
            return self._snapshot.model_copy(deep=True)

    def start(self) -> FoundryJobSnapshot:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("a foundry workload is already active")
            decision = assess_workload(WorkloadTier.SMALL, memory_snapshot())
            if not decision.admitted:
                raise RuntimeError(f"resource admission refused: {decision.reason}")
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
            self._thread = threading.Thread(target=self._execute, daemon=True)
            self._thread.start()
            return self._snapshot.model_copy(deep=True)

    def _execute(self) -> None:
        result: dict[str, object] | None
        failure: str | None
        try:
            result = self._runner(self._cancel.is_set)
        except FoundryCancelled as error:
            status = JobStatus.CANCELLED
            result = None
            failure = str(error)
        except Exception as error:
            status = JobStatus.FAILED
            result = None
            failure = f"{type(error).__name__}: {error}"
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

    def cancel(self) -> FoundryJobSnapshot:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                raise RuntimeError("no active foundry workload to cancel")
            self._cancel.set()
            self._snapshot = self._snapshot.model_copy(
                update={"status": JobStatus.CANCEL_REQUESTED}
            )
            return self._snapshot.model_copy(deep=True)

    def retry(self) -> FoundryJobSnapshot:
        with self._lock:
            if self._snapshot.status not in {JobStatus.CANCELLED, JobStatus.FAILED}:
                raise RuntimeError("only cancelled or failed workloads can be retried")
        return self.start()
