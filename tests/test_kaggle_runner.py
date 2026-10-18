"""Exercise generated notebook bootstrap against real archives, without Torch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from notebooks.build_kaggle_runner import build


def _setup(tmp_path: Path) -> tuple[str, Path]:
    notebook = build(tmp_path / "package")
    value = json.loads(notebook.read_text())
    for cell in value["cells"]:
        if cell["cell_type"] == "code":
            compile(cell["source"], "<generated>", "exec")
    source = value["cells"][1]["source"]
    source = source.replace(
        "SOURCE_ARCHIVE = None",
        f"SOURCE_ARCHIVE = {str(notebook.parent / 'olympus-source.tar.gz')!r}",
        1,
    ).replace("/kaggle/working/olympus", str(tmp_path / "working"))
    return source, tmp_path / "working"


def test_bootstrap_reentry_and_source_tampering(tmp_path: Path) -> None:
    source, working = _setup(tmp_path)
    exec(source, {})
    exec(source, {})
    (working / "source/olympus/__init__.py").write_text("# tampered\n")
    with pytest.raises(ValueError, match="restored file changed"):
        exec(source, {})


def test_archive_tampering_fails_before_extraction(tmp_path: Path) -> None:
    source, working = _setup(tmp_path)
    (tmp_path / "package/olympus-source.tar.gz").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        exec(source, {})
    assert not (working / "source").exists()


def test_recovery_preserves_partial_outputs(tmp_path: Path) -> None:
    source, working = _setup(tmp_path)
    namespace: dict[str, Any] = {}
    exec(source, namespace)
    (working / "results/partial.json").write_text('{"step": 4}')
    namespace["persist"]()
    archive = next(working.glob("recovery-*.tar.gz"))
    restored = tmp_path / "restored"
    restore_code = (
        source.replace(str(working), str(restored))
        .replace("RECOVERY_ARCHIVE = None", f"RECOVERY_ARCHIVE = {str(archive)!r}", 1)
        .replace("RECOVERY_SHA256 = None", f"RECOVERY_SHA256 = {namespace['sha'](archive)!r}", 1)
    )
    exec(restore_code, {})
    assert (restored / "results/partial.json").read_text() == '{"step": 4}'
    with pytest.raises(FileExistsError, match="Restore needs fresh"):
        exec(restore_code, {})


def test_build_refuses_to_overwrite_package(tmp_path: Path) -> None:
    build(tmp_path / "package")
    with pytest.raises(FileExistsError):
        build(tmp_path / "package")


def test_embedded_gpu_source_roundtrip(tmp_path: Path) -> None:
    notebook = build(tmp_path / "embedded", device="cuda", embed_source=True)
    value = json.loads(notebook.read_text())
    source = value["cells"][1]["source"].replace("/kaggle/working", str(tmp_path))
    namespace: dict[str, Any] = {}
    exec(source, namespace)
    assert namespace["DEVICE"] == "cuda"
    assert namespace["RUN_DEVELOPMENT"] is False
    assert (tmp_path / "olympus-source.tar.gz").read_bytes() == (
        notebook.parent / "olympus-source.tar.gz"
    ).read_bytes()


def test_invalid_device_does_not_create_package(tmp_path: Path) -> None:
    target = tmp_path / "invalid"
    with pytest.raises(ValueError, match="device"):
        build(target, device="auto")
    assert not target.exists()
