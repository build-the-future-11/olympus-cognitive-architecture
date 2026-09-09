from __future__ import annotations

import fcntl
import os
import platform
import re
import resource
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import IO, Self


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    total_bytes: int
    available_bytes: int
    swap_total_bytes: int
    swap_used_bytes: int
    process_peak_rss_bytes: int

    @property
    def swap_fraction(self) -> float:
        return 0.0 if self.swap_total_bytes == 0 else self.swap_used_bytes / self.swap_total_bytes

    def as_dict(self) -> dict[str, int | float]:
        return {
            "total_bytes": self.total_bytes,
            "available_bytes": self.available_bytes,
            "swap_total_bytes": self.swap_total_bytes,
            "swap_used_bytes": self.swap_used_bytes,
            "swap_fraction": self.swap_fraction,
            "process_peak_rss_bytes": self.process_peak_rss_bytes,
        }


class WorkloadTier(StrEnum):
    SMALL = "small"
    MEDIUM = "medium"


@dataclass(frozen=True, slots=True)
class WorkloadPolicy:
    tier: WorkloadTier
    estimated_peak_bytes: int
    minimum_available_bytes: int
    maximum_swap_fraction: float
    maximum_active_runs: int
    maximum_queued_runs: int


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    admitted: bool
    reason: str
    policy: WorkloadPolicy
    snapshot: MemorySnapshot


WORKLOAD_POLICIES = {
    WorkloadTier.SMALL: WorkloadPolicy(
        tier=WorkloadTier.SMALL,
        estimated_peak_bytes=1 * 1024**3,
        minimum_available_bytes=2 * 1024**3,
        maximum_swap_fraction=0.90,
        maximum_active_runs=1,
        maximum_queued_runs=1,
    ),
    WorkloadTier.MEDIUM: WorkloadPolicy(
        tier=WorkloadTier.MEDIUM,
        estimated_peak_bytes=6 * 1024**3,
        minimum_available_bytes=8 * 1024**3,
        maximum_swap_fraction=0.50,
        maximum_active_runs=1,
        maximum_queued_runs=0,
    ),
}


def assess_workload(
    tier: WorkloadTier,
    snapshot: MemorySnapshot,
    *,
    active_runs: int = 0,
    queued_runs: int = 0,
) -> AdmissionDecision:
    policy = WORKLOAD_POLICIES[tier]
    if active_runs >= policy.maximum_active_runs:
        reason = "active-run limit reached"
    elif queued_runs > policy.maximum_queued_runs:
        reason = "queue limit exceeded"
    elif snapshot.available_bytes < policy.minimum_available_bytes:
        reason = "available memory is below the tier safety floor"
    elif snapshot.swap_fraction > policy.maximum_swap_fraction:
        reason = "swap utilization exceeds the tier safety ceiling"
    else:
        return AdmissionDecision(True, "admitted", policy, snapshot)
    return AdmissionDecision(False, reason, policy, snapshot)


def _run(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
    return result.stdout


def _try_run(command: list[str]) -> str | None:
    """Return command output when an optional host probe is available.

    Resource discovery runs in CI, containers, and OS sandboxes where individual
    host-information commands may be present but denied. A denied optional probe
    must not make otherwise measurable memory unavailable.
    """

    try:
        return _run(command)
    except (OSError, subprocess.SubprocessError):
        return None


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def _macos_memory() -> MemorySnapshot:
    total_output = _try_run(["sysctl", "-n", "hw.memsize"])
    if total_output is not None:
        total = int(total_output.strip())
    else:
        try:
            total = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError) as error:
            raise RuntimeError("cannot determine total macOS memory") from error
    vm_output = _run(["vm_stat"])
    page_size_match = re.search(r"page size of (\d+) bytes", vm_output)
    if page_size_match is None:
        raise RuntimeError("cannot parse macOS VM page size")
    page_size = int(page_size_match.group(1))
    pages: dict[str, int] = {}
    for line in vm_output.splitlines():
        match = re.match(r"([^:]+):\s+(\d+)\.", line)
        if match:
            pages[match.group(1)] = int(match.group(2))
    available_pages = sum(
        pages.get(name, 0)
        for name in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
    )
    swap_output = _try_run(["sysctl", "vm.swapusage"])
    swap_match = (
        re.search(r"total = ([0-9.]+)M\s+used = ([0-9.]+)M", swap_output)
        if swap_output is not None
        else None
    )
    # macOS does not expose swap totals through a stable non-sysctl interface.
    # A zero total means "not observable" to MemorySnapshot and deliberately
    # disables the ratio gate while retaining the available-memory safety gate.
    swap_total = int(float(swap_match.group(1)) * 1024 * 1024) if swap_match else 0
    swap_used = int(float(swap_match.group(2)) * 1024 * 1024) if swap_match else 0
    return MemorySnapshot(
        total_bytes=total,
        available_bytes=available_pages * page_size,
        swap_total_bytes=swap_total,
        swap_used_bytes=swap_used,
        process_peak_rss_bytes=_peak_rss_bytes(),
    )


def _linux_memory() -> MemorySnapshot:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0]) * 1024
    return MemorySnapshot(
        total_bytes=values["MemTotal"],
        available_bytes=values["MemAvailable"],
        swap_total_bytes=values["SwapTotal"],
        swap_used_bytes=values["SwapTotal"] - values["SwapFree"],
        process_peak_rss_bytes=_peak_rss_bytes(),
    )


def memory_snapshot() -> MemorySnapshot:
    system = platform.system()
    if system == "Darwin":
        return _macos_memory()
    if system == "Linux":
        return _linux_memory()
    raise RuntimeError(f"resource governor does not support {system}")


class WorkloadUnavailable(RuntimeError):
    """Admission was refused without starting a workload."""


class ResourceGovernor:
    """Exclusive local workload lock plus preflight memory and swap enforcement."""

    def __init__(
        self,
        lock_path: Path,
        *,
        min_available_bytes: int = 2 * 1024**3,
        max_swap_fraction: float = 0.90,
    ) -> None:
        if min_available_bytes < 256 * 1024**2:
            raise ValueError("min_available_bytes must be at least 256 MiB")
        if not 0.0 < max_swap_fraction <= 1.0:
            raise ValueError("max_swap_fraction must be in (0, 1]")
        self.lock_path = lock_path.resolve()
        self.min_available_bytes = min_available_bytes
        self.max_swap_fraction = max_swap_fraction
        self._handle: IO[str] | None = None
        self.preflight: MemorySnapshot | None = None

    def acquire(self) -> MemorySnapshot:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.close()
            raise WorkloadUnavailable(
                "another model-loading or training workload holds the lock"
            ) from error

        try:
            snapshot = memory_snapshot()
            if snapshot.available_bytes < self.min_available_bytes:
                raise WorkloadUnavailable(
                    f"available memory {snapshot.available_bytes} is below the "
                    f"{self.min_available_bytes}-byte safety floor"
                )
            if snapshot.swap_fraction > self.max_swap_fraction:
                raise WorkloadUnavailable(
                    f"swap utilization {snapshot.swap_fraction:.1%} exceeds the "
                    f"{self.max_swap_fraction:.1%} safety ceiling"
                )
            handle.seek(0)
            handle.truncate()
            handle.write(f"pid={os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
            raise

        self._handle = handle
        self.preflight = snapshot
        return snapshot

    def release(self) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
