from __future__ import annotations

import fcntl
import os
import platform
import re
import resource
import subprocess
from dataclasses import dataclass
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


def _run(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
    return result.stdout


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def _macos_memory() -> MemorySnapshot:
    total = int(_run(["sysctl", "-n", "hw.memsize"]).strip())
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
    swap_output = _run(["sysctl", "vm.swapusage"])
    swap_match = re.search(r"total = ([0-9.]+)M\s+used = ([0-9.]+)M", swap_output)
    if swap_match is None:
        raise RuntimeError("cannot parse macOS swap usage")
    return MemorySnapshot(
        total_bytes=total,
        available_bytes=available_pages * page_size,
        swap_total_bytes=int(float(swap_match.group(1)) * 1024 * 1024),
        swap_used_bytes=int(float(swap_match.group(2)) * 1024 * 1024),
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
            raise RuntimeError(
                "another model-loading or training workload holds the lock"
            ) from error
        snapshot = memory_snapshot()
        if snapshot.available_bytes < self.min_available_bytes:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
            raise RuntimeError(
                f"available memory {snapshot.available_bytes} is below the "
                f"{self.min_available_bytes}-byte safety floor"
            )
        if snapshot.swap_fraction > self.max_swap_fraction:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
            raise RuntimeError(
                f"swap utilization {snapshot.swap_fraction:.1%} exceeds the "
                f"{self.max_swap_fraction:.1%} safety ceiling"
            )
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        os.fsync(handle.fileno())
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
