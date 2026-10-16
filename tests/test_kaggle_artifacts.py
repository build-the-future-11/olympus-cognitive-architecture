from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from notebooks.kaggle_support import artifacts


def _write(root: Path, relative: str, content: bytes = b"fixture\n") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _source(root: Path) -> None:
    for name in (
        "olympus/__init__.py",
        "olympus/training/core.py",
        "scripts/validate_olympus_campaign.py",
        "pyproject.toml",
        "requirements-kaggle.txt",
        "configs/kaggle/smoke-cuda.json",
        "tests/test_olympus_system.py",
        "tests/test_olympus_resume.py",
        "tests/test_olympus_dataset.py",
        "tests/test_controlled_fusion.py",
        "tests/test_runtime_hardening.py",
        "tests/conftest.py",
        "notebooks/kaggle_support/artifacts.py",
    ):
        _write(root, name)
    for name in (
        ".env",
        ".git/config",
        "research/history.json",
        "datasets/protected.pt",
        "README.md",
        "notebooks/campaign.ipynb",
        "tests/test_unrelated.py",
    ):
        _write(root, name, b"must not appear in source snapshot")


def _malicious_archive(
    path: Path,
    name: str,
    *,
    kind: bytes = tarfile.REGTYPE,
    payload: bytes = b"expected",
    content_digest: str | None = None,
    duplicate: bool = False,
) -> None:
    manifest = {
        "schema_version": 1,
        "kind": "evidence",
        "excluded_cache_paths": [],
        "files": [
            {
                "path": "run/checkpoint.pt",
                "size": len(payload),
                "sha256": content_digest or hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    encoded = json.dumps(manifest).encode()
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo(artifacts.MANIFEST_NAME)
        info.size = len(encoded)
        archive.addfile(info, io.BytesIO(encoded))
        for _ in range(2 if duplicate else 1):
            info = tarfile.TarInfo(name)
            info.type = kind
            info.size = len(payload) if kind == tarfile.REGTYPE else 0
            if kind in {tarfile.SYMTYPE, tarfile.LNKTYPE}:
                info.linkname = "../../outside"
            archive.addfile(info, io.BytesIO(payload) if info.size else None)


def test_source_bundle_exact_identity_and_allowlist(tmp_path: Path) -> None:
    root = tmp_path / "source"
    _source(root)
    output = tmp_path / "source.tar.gz"
    result = artifacts.build_source_bundle(root, output)
    expected = {
        str(path.relative_to(root)): artifacts.file_sha256(path)
        for path in [
            *root.glob("olympus/**/*.py"),
            *root.glob("scripts/*.py"),
            root / "pyproject.toml",
        ]
    }
    source_sha = hashlib.sha256(
        json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert result["source_sha256"] == source_sha
    names = {entry["path"] for entry in result["manifest"]["files"]}
    assert {"tests/conftest.py", "tests/test_olympus_resume.py"} <= names
    assert not names & {
        ".env",
        ".git/config",
        "research/history.json",
        "datasets/protected.pt",
        "README.md",
        "notebooks/campaign.ipynb",
        "tests/test_unrelated.py",
    }
    restored = tmp_path / "restored"
    artifacts.restore_archive(output, restored, expected_sha256=result["archive_sha256"])
    assert (
        artifacts.verify_restored(restored, expected_archive_sha256=result["archive_sha256"])[
            "source_sha256"
        ]
        == source_sha
    )
    assert (restored / "olympus/training/core.py").read_bytes() == b"fixture\n"


def test_archive_deterministic_and_never_overwrites(tmp_path: Path) -> None:
    root = tmp_path / "source"
    _source(root)
    first = artifacts.build_source_bundle(root, tmp_path / "a.tar.gz")
    second = artifacts.build_source_bundle(root, tmp_path / "b.tar.gz")
    assert first["archive_sha256"] == second["archive_sha256"]
    with pytest.raises(FileExistsError):
        artifacts.build_source_bundle(root, tmp_path / "a.tar.gz")
    inside = artifacts.build_source_bundle(root, root / "output" / "source.tar.gz")
    assert inside["archive_sha256"] == first["archive_sha256"]


def test_evidence_retains_failed_partial_full_checkpoints_and_data(tmp_path: Path) -> None:
    root = tmp_path / "generated-results"
    names = [
        "campaign.json",
        "data/train.pt",
        "data/manifest_development.json",
        "runs/campaign_state.json",
        "runs/cells/pilot/status.json",
        "runs/cells/pilot/metrics.jsonl",
        "runs/cells/pilot/checkpoints/latest.json",
        "runs/cells/pilot/checkpoints/best.json",
        "runs/cells/pilot/checkpoints/step-1-old.pt",
        "runs/cells/pilot/checkpoints/step-2-new.pt",
        "runs/failed-attempt.json",
    ]
    for name in names:
        _write(root, name, name.encode())
    _write(root, "__pycache__/ignored.pyc")
    output = tmp_path / "evidence.tar.gz"
    result = artifacts.archive_tree(root, output)
    assert {entry["path"] for entry in result["manifest"]["files"]} == set(names)
    assert result["manifest"]["excluded_cache_paths"] == ["__pycache__/"]
    restored = tmp_path / "resumed"
    artifacts.restore_archive(output, restored, expected_sha256=result["archive_sha256"])
    for name in names:
        assert (restored / name).read_bytes() == name.encode()
    with pytest.raises(FileExistsError):
        artifacts.restore_archive(output, restored)
    (restored / names[0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="restored file changed"):
        artifacts.verify_restored(restored)


@pytest.mark.parametrize(
    "name,kind,duplicate",
    [
        ("../outside", tarfile.REGTYPE, False),
        ("/absolute", tarfile.REGTYPE, False),
        ("files/run/checkpoint.pt", tarfile.SYMTYPE, False),
        ("files/run/checkpoint.pt", tarfile.LNKTYPE, False),
        ("files/run/checkpoint.pt", tarfile.REGTYPE, True),
        ("files/uninventoried", tarfile.REGTYPE, False),
    ],
)
def test_rejects_unsafe_tar_members_before_creating_restore_target(
    name: str,
    kind: bytes,
    duplicate: bool,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    _malicious_archive(archive, name, kind=kind, duplicate=duplicate)
    target = tmp_path / "restore"
    with pytest.raises(ValueError):
        artifacts.restore_archive(archive, target)
    assert not target.exists()
    assert not (tmp_path / "outside").exists()


def test_rejects_content_and_archive_hash_tampering(tmp_path: Path) -> None:
    archive = tmp_path / "tampered.tar.gz"
    _malicious_archive(archive, "files/run/checkpoint.pt", content_digest="0" * 64)
    with pytest.raises(ValueError, match="file digest mismatch"):
        artifacts.verify_archive(archive)
    with pytest.raises(ValueError, match="independently pinned"):
        artifacts.verify_archive(archive, expected_sha256="0" * 64)


def test_rejects_source_symlink_and_secret_evidence(tmp_path: Path) -> None:
    root = tmp_path / "source"
    _source(root)
    (root / "olympus/link.py").symlink_to(root / ".env")
    with pytest.raises(ValueError, match="symlink"):
        artifacts.build_source_bundle(root, tmp_path / "source.tar.gz")
    evidence = tmp_path / "generated-results"
    _write(evidence, "run/report.json")
    _write(evidence, "run/.env", b"TOKEN=not-for-export")
    with pytest.raises(ValueError, match="secret-like"):
        artifacts.archive_tree(evidence, tmp_path / "evidence.tar.gz")


def test_rejects_archive_recursion_checkout_root_and_size_overflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "generated-results"
    _write(root, "checkpoint.pt", b"12345678")
    with pytest.raises(ValueError, match="outside"):
        artifacts.archive_tree(root, root / "archive.tar.gz")
    monkeypatch.setattr(artifacts, "MAX_BYTES", 4)
    with pytest.raises(ValueError, match="budget"):
        artifacts.archive_tree(root, tmp_path / "archive.tar.gz")
    _write(root, "pyproject.toml")
    with pytest.raises(ValueError, match="designated"):
        artifacts.archive_tree(root, tmp_path / "archive.tar.gz")


def test_packaging_import_never_imports_torch() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import notebooks.kaggle_support.artifacts; "
            "assert 'torch' not in sys.modules",
        ],
        check=True,
        capture_output=True,
    )
