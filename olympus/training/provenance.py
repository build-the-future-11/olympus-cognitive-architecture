"""Small, Torch-free identity and durable-write helpers for local campaigns."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def digest_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    """Replace metadata only after its complete contents have reached the filesystem."""
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def source_identity() -> dict[str, Any]:
    """Hash actual source bytes, including uncommitted/untracked implementation files.

    Git metadata is informational; the source digest is stable across output writes,
    repository relocation and commits that do not change implementation bytes.
    No environment variables, remote URLs or arbitrary workspace files are recorded.
    """
    root = Path(__file__).resolve().parents[2]
    files = sorted([*root.glob("olympus/**/*.py"), *root.glob("scripts/*.py")])
    files.extend(path for path in [root / "pyproject.toml"] if path.is_file())
    hashes = {str(path.relative_to(root)): file_sha256(path) for path in files}

    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=root, stderr=subprocess.DEVNULL, text=True, timeout=5
            ).strip()
        except (OSError, subprocess.SubprocessError):  # fmt: skip
            return None

    status = git("status", "--porcelain", "--untracked-files=normal")
    return {
        "sha256": digest_json(hashes),
        "files": hashes,
        "head": git("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
    }
