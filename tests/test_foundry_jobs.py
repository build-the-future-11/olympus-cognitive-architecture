import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from olympus.foundry import jobs
from olympus.foundry.jobs import FoundryJobManager, FoundryJobSnapshot, JobJournal, JobStatus
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


def test_killed_worker_is_reconciled_with_persisted_identity(tmp_path: Path) -> None:
    script = """
import sys, threading
from pathlib import Path
from olympus.foundry import jobs
from olympus.foundry.resources import MemorySnapshot
from olympus.foundry.service import FoundryService
jobs.memory_snapshot = lambda: MemorySnapshot(16*1024**3, 10*1024**3, 0, 0, 1000)
assert 'olympus.api' not in sys.modules
assert 'torch' not in sys.modules
service = FoundryService(Path(sys.argv[1]))
manager = jobs.FoundryJobManager(service, Path('unused'),
    runner=lambda cancelled: threading.Event().wait())
job = manager.start()
print(job.job_id, flush=True)
threading.Event().wait()
"""
    with subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ) as worker:
        try:
            deadline = time.monotonic() + 10
            journal = JobJournal(tmp_path)
            while journal.read().status is JobStatus.IDLE and time.monotonic() < deadline:
                assert worker.poll() is None
                time.sleep(0.01)
            snapshot = journal.read()
            assert snapshot.status is JobStatus.RUNNING
            with FoundryService(tmp_path) as service:
                observer = FoundryJobManager(service, tmp_path / "unused")
                assert observer.snapshot().status is JobStatus.RUNNING
                worker.kill()
                worker.wait(timeout=5)
                # A live observer must recover without requiring a server restart.
                assert observer.snapshot().job_id == snapshot.job_id
                assert observer.snapshot().status is JobStatus.FAILED
                assert observer.history()[0].finished_at is not None
            with FoundryService(tmp_path) as service:
                recovered = FoundryJobManager(service, tmp_path / "unused")
                assert recovered.snapshot().job_id == snapshot.job_id
                assert recovered.snapshot().status is JobStatus.FAILED
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.wait(timeout=5)


def _wait(manager: FoundryJobManager) -> JobStatus:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        status = manager.snapshot().status
        if status not in {JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED}:
            return status
        time.sleep(0.005)
    raise AssertionError("job did not finish within the test bound")


@pytest.mark.parametrize("abandoned_status", [JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED])
def test_start_preserves_abandoned_job_as_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, abandoned_status: JobStatus,
) -> None:
    monkeypatch.setattr(jobs, "memory_snapshot", _safe_memory)
    with FoundryService(tmp_path) as service:
        manager = FoundryJobManager(service, tmp_path / "unused", runner=lambda _: {})
        journal = JobJournal(tmp_path)
        journal.save(FoundryJobSnapshot(job_id="abandoned", status=abandoned_status))
        manager.start()
        assert _wait(manager) is JobStatus.SUCCEEDED
        history = manager.history()
        assert len(history) == 2
        assert history[1].job_id == "abandoned"
        assert history[1].status is JobStatus.FAILED
        assert history[1].finished_at is not None


def test_jobs_survive_recreation_and_reconcile_interrupted_workers(tmp_path: Path) -> None:
    with FoundryService(tmp_path) as service:
        journal = JobJournal(tmp_path)
        journal.save(FoundryJobSnapshot(job_id="interrupted", status=JobStatus.RUNNING))
        manager = FoundryJobManager(service, tmp_path / "unused")
        assert manager.snapshot().status is JobStatus.FAILED
        assert "interrupted" in (manager.snapshot().error or "")
        recreated = FoundryJobManager(service, tmp_path / "unused")
        assert recreated.snapshot() == manager.snapshot()
        assert recreated.history() == [manager.snapshot()]
        assert recreated.history(offset=1) == []
        with pytest.raises(ValueError, match="limit"):
            recreated.history(limit=101)


def test_managers_share_lease_and_cross_worker_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jobs, "memory_snapshot", _safe_memory)

    def runner(cancelled: Callable[[], bool]) -> dict[str, object]:
        deadline = time.monotonic() + 2
        while not cancelled() and time.monotonic() < deadline:
            time.sleep(0.005)
        if cancelled():
            raise FoundryCancelled("cancelled")
        return {}

    with FoundryService(tmp_path) as service:
        first = FoundryJobManager(service, tmp_path / "unused", runner=runner)
        first.start()
        second = FoundryJobManager(service, tmp_path / "unused", runner=runner)
        assert second.snapshot().status is JobStatus.RUNNING
        with pytest.raises(RuntimeError, match="already active"):
            second.start()
        second.cancel()
        assert _wait(first) is JobStatus.CANCELLED
        assert second.snapshot().status is JobStatus.CANCELLED


def test_verification_entrypoints_share_resource_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from olympus.foundry import resources

    monkeypatch.setattr(resources, "memory_snapshot", _safe_memory)
    with FoundryService(tmp_path) as service:
        with resources.ResourceGovernor(tmp_path / "verification.lock"):
            with pytest.raises(RuntimeError, match="holds the lock"):
                service.run_verification_pipeline(tmp_path / "not-read")
        monkeypatch.setattr(
            resources, "memory_snapshot", lambda: MemorySnapshot(1, 0, 0, 0, 0)
        )
        with pytest.raises(RuntimeError, match="safety floor"):
            service.run_verification_pipeline(tmp_path / "not-read")


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


def test_foundry_job_redacts_internal_failure_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(jobs, "memory_snapshot", _safe_memory)
    secret = "PRIVATE_FOUNDRY_FAILURE_DETAIL"

    def fail(_cancelled: Callable[[], bool]) -> dict[str, object]:
        raise RuntimeError(secret)

    with FoundryService(tmp_path / "foundry") as service:
        manager = FoundryJobManager(service, tmp_path / "unused.txt", runner=fail)
        manager.start()
        assert _wait(manager) is JobStatus.FAILED
        snapshot = manager.snapshot()

    assert snapshot.error == "Foundry workload failed; inspect server logs for details."
    assert secret not in (snapshot.error or "")
    assert secret not in caplog.text
    assert "RuntimeError" in caplog.text


def test_foundry_job_rejects_nonfinite_runner_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(jobs, "memory_snapshot", _safe_memory)
    with FoundryService(tmp_path / "foundry") as service:
        manager = FoundryJobManager(
            service,
            tmp_path / "unused.txt",
            runner=lambda _: {"metric": float("nan")},
        )
        manager.start()
        assert _wait(manager) is JobStatus.FAILED
        assert manager.snapshot().result is None


def test_foundry_job_shutdown_waits_for_cooperative_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(jobs, "memory_snapshot", _safe_memory)
    started = threading.Event()

    def cooperative(cancelled: Callable[[], bool]) -> dict[str, object]:
        started.set()
        while not cancelled():
            time.sleep(0.001)
        raise FoundryCancelled("private cancellation detail")

    with FoundryService(tmp_path / "foundry") as service:
        manager = FoundryJobManager(
            service,
            tmp_path / "unused.txt",
            runner=cooperative,
        )
        manager.start()
        assert started.wait(timeout=1)
        assert manager.shutdown(timeout_seconds=1) is True
        snapshot = manager.snapshot()
        assert snapshot.status is JobStatus.CANCELLED
        assert snapshot.error == "Foundry workload was cancelled."
        with pytest.raises(RuntimeError, match="shut down"):
            manager.start()

    with pytest.raises(ValueError, match="finite and non-negative"):
        manager.shutdown(timeout_seconds=float("nan"))
