from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def workflow() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "verify_foundry_admission.py"
    spec = importlib.util.spec_from_file_location("foundry_workflow_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((400, 1000, 20, 30, 40, 10), 300),
        ((900, 1000, 100, 200, 150, 50), 400),
        ((900, 1000, 400, 300, 200, 200), 0),
        ((10, 100, 0, 0, 10, 0), 0),
        ((-1, 1000, 0, 0, 0, 0), 0),
    ],
)
@pytest.mark.parametrize("version", [1, 2])
def test_headroom_credit_excludes_unavailable_cache(
    tmp_path: Path, values: tuple[int, ...], expected: int, version: int
) -> None:
    names = ("inactive_file", "file", "shmem", "file_mapped", "file_dirty", "file_writeback")
    if version == 1:
        names = (
            "total_inactive_file",
            "total_cache",
            "total_shmem",
            "total_mapped_file",
            "total_dirty",
            "total_writeback",
        )
    (tmp_path / "memory.stat").write_text(
        "\n".join(f"{key} {value}" for key, value in zip(names, values, strict=True))
    )
    assert workflow().cache_credit(tmp_path, version) == expected


def test_missing_cgroup_counters_receive_no_headroom_credit(tmp_path: Path) -> None:
    module = workflow()
    assert module.cache_credit(tmp_path, 2) == 0
    (tmp_path / "memory.stat").write_text("inactive_file 400\nfile 1000\n")
    assert module.cache_credit(tmp_path, 2) == 0
