"""Bounded subprocess boundary for the built-in Foundry verification workload."""

from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

from olympus.foundry.attestation import load_strict_json_object
from olympus.foundry.schemas import FoundryPipelineResult
from olympus.foundry.service import FoundryCancelled, FoundryService, _atomic_write


def _command(root: Path, sample: Path, output: Path, timeout_seconds: float) -> list[str]:
    bootstrap = (
        "import sys; sys.path.insert(0, sys.argv.pop(1)); "
        "from olympus.foundry.worker import main; main()"
    )
    # Do not resolve code through cwd, PYTHONPATH, or a different installed Olympus.
    package_root = Path(__file__).resolve().parents[2]
    return [sys.executable, "-I", "-c", bootstrap, str(package_root), "--root", str(root),
            "--sample", str(sample), "--output", str(output),
            "--parent-pid", str(os.getpid()), "--timeout", str(timeout_seconds)]


def _watch_parent(parent_pid: int, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while os.getppid() == parent_pid and time.monotonic() < deadline:
        time.sleep(0.1)
    # A worker owns its process group; do not leave it running after supervisor death.
    os.killpg(os.getpid(), signal.SIGKILL)


def _stop(process: subprocess.Popen[bytes]) -> None:
    # This group belongs exclusively to the child we started with a new session.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=2)


def run_isolated_verification(
    root: Path, sample: Path, cancelled: Callable[[], bool], *, timeout_seconds: float
) -> dict[str, object]:
    if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 3600:
        raise ValueError("worker timeout must be finite and in (0, 3600] seconds")
    if cancelled():
        raise FoundryCancelled("cancelled before worker startup")
    with TemporaryDirectory(prefix=".worker-", dir=root) as temporary:
        output = Path(temporary) / "result.json"
        deadline = time.monotonic() + timeout_seconds
        with subprocess.Popen(
            _command(root, sample.resolve(), output, timeout_seconds),
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        ) as process:
            try:
                while process.poll() is None:
                    if cancelled():
                        raise FoundryCancelled("worker cancellation requested")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("worker execution deadline exceeded")
                    time.sleep(0.05)
                if process.returncode != 0:
                    raise RuntimeError(f"verification worker exited with code {process.returncode}")
                result = FoundryPipelineResult.model_validate(load_strict_json_object(output))
                return result.model_dump(mode="json")
            finally:
                if process.poll() is None:
                    _stop(process)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--parent-pid", required=True, type=int)
    parser.add_argument("--timeout", required=True, type=float)
    args = parser.parse_args()
    if os.getpgrp() != os.getpid():
        parser.error("internal verification workers require a dedicated process session")
    if not 0 < args.timeout <= 3600 or args.parent_pid < 1:
        parser.error("invalid parent or timeout")
    threading.Thread(
        target=_watch_parent, args=(args.parent_pid, args.timeout), daemon=True
    ).start()
    with FoundryService(args.root) as service:
        result = service.run_verification_pipeline(args.sample)
    _atomic_write(args.output, result.model_dump_json().encode("utf-8"))


if __name__ == "__main__":
    main()
