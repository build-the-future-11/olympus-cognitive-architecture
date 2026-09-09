from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

import pytest

from olympus.foundry import jobs, worker
from olympus.foundry.attestation import load_strict_json_object
from olympus.foundry.jobs import FoundryJobManager
from olympus.foundry.resources import MemorySnapshot
from olympus.foundry.service import FoundryCancelled, FoundryService


def test_real_isolated_worker_completes_registry_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shadow = tmp_path / "olympus"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("raise RuntimeError('shadow package executed')\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    sample = Path(worker.__file__).with_name("foundry_verification.txt")
    with FoundryService(tmp_path) as service:
        result = worker.run_isolated_verification(
            tmp_path, sample, lambda: False, timeout_seconds=30
        )
        assert result["model"]
        assert service.status()["integrity"] == "ok"
        assert len(service.list_models()) == 1
    assert not list(tmp_path.glob(".worker-*"))


@pytest.mark.parametrize("cancel", [False, True])
def test_uncooperative_worker_is_stopped_and_reaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    ready = tmp_path / "ready"
    script = (
        "import os,signal,time,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"pathlib.Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    monkeypatch.setattr(worker, "_command", lambda *args: [sys.executable, "-c", script])
    start = time.monotonic()
    with pytest.raises(FoundryCancelled if cancel else TimeoutError):
        worker.run_isolated_verification(
            tmp_path, tmp_path / "unused", lambda: cancel and ready.exists(), timeout_seconds=1,
        )
    assert time.monotonic() - start < 5
    assert ready.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int(ready.read_text()), 0)
    assert not list(tmp_path.glob(".worker-*"))


def test_worker_failure_and_prestart_cancel_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(FoundryCancelled):
        worker.run_isolated_verification(tmp_path, tmp_path / "unused", lambda: True,
                                         timeout_seconds=1)
    monkeypatch.setattr(
        worker, "_command", lambda *args: [sys.executable, "-c", "raise SystemExit(4)"]
    )
    with pytest.raises(RuntimeError, match="code 4"):
        worker.run_isolated_verification(tmp_path, tmp_path / "unused", lambda: False,
                                         timeout_seconds=2)


def test_manager_rejects_unbounded_deadlines(tmp_path: Path) -> None:
    with FoundryService(tmp_path) as service:
        for value in (0, -1, float("inf"), float("nan"), 3601):
            with pytest.raises(ValueError, match="timeout"):
                FoundryJobManager(service, tmp_path / "unused", worker_timeout_seconds=value)
            with pytest.raises(ValueError, match="timeout"):
                worker.run_isolated_verification(tmp_path, tmp_path / "unused", lambda: False,
                                                 timeout_seconds=value)


def test_parent_watchdog_terminates_its_group_when_parent_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parents = iter([10, 11])
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "getppid", lambda: next(parents))
    monkeypatch.setattr(os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    worker._watch_parent(10, 60)
    assert killed == [(os.getpid(), signal.SIGKILL)]


def test_manager_records_worker_timeout_without_closing_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(jobs, "memory_snapshot", lambda: MemorySnapshot(16 << 30, 8 << 30, 0, 0, 1))
    monkeypatch.setattr(worker, "_command", lambda *args: [sys.executable, "-c",
                                                         "import time; time.sleep(60)"])
    with FoundryService(tmp_path) as service:
        manager = FoundryJobManager(service, tmp_path / "unused", worker_timeout_seconds=0.2)
        manager.start()
        deadline = time.monotonic() + 5
        while manager.snapshot().status in {
            jobs.JobStatus.RUNNING, jobs.JobStatus.CANCEL_REQUESTED,
        }:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert manager.snapshot().status is jobs.JobStatus.FAILED
        assert "execution deadline" in (manager.snapshot().error or "")
        assert manager.shutdown(timeout_seconds=2)
        assert service.status()["integrity"] == "ok"


def test_worker_entrypoint_validates_session_and_writes_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = Path(worker.__file__).with_name("foundry_verification.txt")
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["worker", "--root", str(tmp_path / "registry"),
                                     "--sample", str(sample), "--output", str(output),
                                     "--parent-pid", str(os.getpid()), "--timeout", "30"])
    monkeypatch.setattr(os, "getpgrp", lambda: -1)
    with pytest.raises(SystemExit) as error:
        worker.main()
    assert error.value.code == 2
    assert not output.exists()
    monkeypatch.setattr(os, "getpgrp", os.getpid)
    # Do not run a process watchdog inside the test runner's own process group.
    monkeypatch.setattr("olympus.foundry.worker.threading.Thread.start", lambda self: None)
    worker.main()
    assert load_strict_json_object(output)["model"]
