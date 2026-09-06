import time
from collections.abc import Callable
from pathlib import Path

import pytest

from olympus.foundry import jobs
from olympus.foundry.jobs import FoundryJobManager, JobStatus
from olympus.foundry.resources import MemorySnapshot
from olympus.foundry.service import FoundryCancelled, FoundryService


def _safe_memory() -> MemorySnapshot:
    return MemorySnapshot(
        total_bytes=16 * 1024**3,
        available_bytes=10 * 1024**3,
        swap_total_bytes=4 * 1024**3,
        swap_used_bytes=0,
        process_peak_rss_bytes=1_000,
    )


def _wait(manager: FoundryJobManager) -> JobStatus:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        status = manager.snapshot().status
        if status not in {JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED}:
            return status
        time.sleep(0.005)
    raise AssertionError("job did not finish within the test bound")


def test_foundry_job_runs_one_admitted_workload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jobs, "memory_snapshot", _safe_memory)
    with FoundryService(tmp_path / "foundry") as service:
        manager = FoundryJobManager(
            service,
            tmp_path / "unused.txt",
            runner=lambda cancelled: {"cancelled": cancelled()},
        )
        started = manager.start()
        assert started.status is JobStatus.RUNNING
        assert started.admission is not None
        assert _wait(manager) is JobStatus.SUCCEEDED
        assert manager.snapshot().result == {"cancelled": False}
        with pytest.raises(RuntimeError, match="cancelled or failed"):
            manager.retry()


def test_foundry_job_cancel_and_retry_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jobs, "memory_snapshot", _safe_memory)

    def cancellable(cancelled: Callable[[], bool]) -> dict[str, object]:
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            if cancelled():
                raise FoundryCancelled("cancelled by test")
            time.sleep(0.005)
        return {"unexpected": True}

    with FoundryService(tmp_path / "foundry") as service:
        manager = FoundryJobManager(
            service,
            tmp_path / "unused.txt",
            runner=cancellable,
        )
        manager.start()
        assert manager.cancel().status is JobStatus.CANCEL_REQUESTED
        assert _wait(manager) is JobStatus.CANCELLED
        retried = manager.retry()
        assert retried.status is JobStatus.RUNNING
        manager.cancel()
        assert _wait(manager) is JobStatus.CANCELLED


def test_foundry_job_refuses_unsafe_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unsafe = MemorySnapshot(16 * 1024**3, 1, 4 * 1024**3, 0, 1_000)
    monkeypatch.setattr(jobs, "memory_snapshot", lambda: unsafe)
    with FoundryService(tmp_path / "foundry") as service:
        manager = FoundryJobManager(service, tmp_path / "unused.txt")
        with pytest.raises(RuntimeError, match="resource admission refused"):
            manager.start()
