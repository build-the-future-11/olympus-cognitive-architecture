"""Torch-free, allowlisted source snapshots and checksummed campaign evidence.

Evidence roots must be explicitly designated generated-results directories. They
must never be a checkout, home directory, or general working directory. Filename
screening is a safeguard, not a detector for secrets embedded in arbitrary data.
Pin the returned archive SHA-256 independently when transferring an archive.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import IO, Any

MAX_BYTES = 1024**3
MAX_FILES = 10_000
MAX_MANIFEST_BYTES = 4 * 1024**2
MANIFEST_NAME = "BUNDLE_MANIFEST.json"
RESTORE_RECEIPT = ".kaggle_restore_complete.json"
_VALIDATION_TESTS = frozenset(
    {
        "test_olympus_system.py",
        "test_olympus_dataset.py",
        "test_controlled_fusion.py",
        "test_runtime_hardening.py",
        "test_olympus_resume.py",
        "conftest.py",
        "__init__.py",
    }
)
_CACHE_NAMES = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".cache",
        ".git",
        ".venv",
        "venv",
        "node_modules",
    }
)
_SECRET_NAMES = frozenset(
    {
        ".aws",
        ".ssh",
        ".kaggle",
        ".config",
        ".credentials",
        "credentials.json",
        "kaggle.json",
        "token.json",
        "tokens.json",
        "secrets.json",
        "secrets.toml",
        "id_rsa",
        "id_ed25519",
        ".netrc",
        ".npmrc",
        ".pypirc",
    }
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _relative(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise ValueError("invalid archive path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or "\x00" in value
        or str(path) != value
        or value == "."
        or ":" in path.parts[0]
    ):
        raise ValueError("archive paths must be canonical contained relative paths")
    return value


def _secret(path: PurePosixPath) -> bool:
    return any(
        part.lower() in _SECRET_NAMES
        or part.lower().startswith((".env", "client_secret"))
        or part.lower().endswith((".pem", ".key", ".p12", ".pfx"))
        for part in path.parts
    )


def _source_allowed(relative: str) -> bool:
    path = PurePosixPath(relative)
    parts = path.parts
    if _secret(path) or any(part in _CACHE_NAMES for part in parts):
        return False
    if relative in {"pyproject.toml", "requirements-kaggle.txt"}:
        return True
    if len(parts) >= 2 and parts[0] == "olympus" and path.suffix == ".py":
        return True
    if len(parts) == 2 and parts[0] == "scripts" and path.suffix == ".py":
        return True
    if len(parts) == 3 and parts[:2] == ("configs", "kaggle") and path.suffix == ".json":
        return True
    if len(parts) == 2 and parts[0] == "tests" and parts[1] in _VALIDATION_TESTS:
        return True
    return len(parts) == 3 and parts[:2] == ("notebooks", "kaggle_support") and path.suffix == ".py"


def _identity_file(relative: str) -> bool:
    parts = PurePosixPath(relative).parts
    return (
        relative == "pyproject.toml"
        or (len(parts) >= 2 and parts[0] == "olympus" and relative.endswith(".py"))
        or (len(parts) == 2 and parts[0] == "scripts" and relative.endswith(".py"))
    )


def _regular(root: Path, relative: str) -> Path:
    relative = _relative(relative)
    path = root / relative
    current = path
    while current != root:
        if current.is_symlink():
            raise ValueError(f"symlink is not an admissible artifact: {relative}")
        current = current.parent
    if not path.is_file():
        raise ValueError(f"artifact must be a regular file: {relative}")
    return path


def _inventory(root: Path, paths: list[str]) -> list[dict[str, Any]]:
    if not paths or len(paths) > MAX_FILES:
        raise ValueError("artifact count is outside the archive budget")
    result = []
    total = 0
    for relative in sorted(paths):
        path = _regular(root, relative)
        size = path.stat().st_size
        total += size
        if total > MAX_BYTES:
            raise ValueError("archive exceeds the 1 GiB uncompressed budget")
        result.append({"path": relative, "size": size, "sha256": file_sha256(path)})
    return result


def _source_paths(root: Path) -> list[str]:
    candidates: set[str] = set()
    for pattern in (
        "olympus/**/*.py",
        "scripts/*.py",
        "configs/kaggle/*.json",
        "notebooks/kaggle_support/*.py",
    ):
        candidates.update(str(path.relative_to(root)) for path in root.glob(pattern))
    candidates.update(
        name for name in ("pyproject.toml", "requirements-kaggle.txt") if (root / name).is_file()
    )
    candidates.update(
        f"tests/{name}" for name in _VALIDATION_TESTS if (root / "tests" / name).is_file()
    )
    required = {"olympus/__init__.py", "pyproject.toml", "requirements-kaggle.txt"}
    if "scripts/validate_olympus_campaign.py" in candidates:
        required.update(
            f"tests/{name}"
            for name in _VALIDATION_TESTS
            if name not in {"conftest.py", "__init__.py"}
        )
    if not required <= candidates:
        raise ValueError("source root is missing required Olympus source/environment files")
    if any(not _source_allowed(name) for name in candidates):
        raise ValueError("source candidate violates the explicit allowlist")
    # Glob recursion does not follow directory links. Reject them instead of silently
    # omitting source, which would produce a misleading source identity.
    for selected in ("olympus", "scripts", "configs/kaggle", "notebooks/kaggle_support"):
        base = root / selected
        if base.is_symlink():
            raise ValueError("source directory must not be a symlink")
        if base.exists():
            for directory, directories, _files in os.walk(base, followlinks=False):
                if any((Path(directory) / name).is_symlink() for name in directories):
                    raise ValueError("source directory contains a symlink")
    return sorted(candidates)


def _evidence_paths(root: Path) -> tuple[list[str], list[str]]:
    if (
        root == Path.home().resolve()
        or (root / "pyproject.toml").exists()
        or (root / ".git").exists()
    ):
        raise ValueError("evidence root must be a designated generated-results directory")
    files: list[str] = []
    excluded: list[str] = []
    for directory, directories, names in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in list(directories):
            relative = (base / name).relative_to(root).as_posix()
            if (base / name).is_symlink():
                raise ValueError(f"evidence directory contains a symlink: {relative}")
            if _secret(PurePosixPath(relative)):
                raise ValueError(f"secret-like path is forbidden in evidence: {relative}")
            if name in _CACHE_NAMES:
                directories.remove(name)
                excluded.append(relative + "/")
        for name in names:
            relative = (base / name).relative_to(root).as_posix()
            if (base / name).is_symlink():
                raise ValueError(f"evidence file is a symlink: {relative}")
            if _secret(PurePosixPath(relative)):
                raise ValueError(f"secret-like path is forbidden in evidence: {relative}")
            if name == RESTORE_RECEIPT or name.endswith((".pyc", ".pyo")):
                excluded.append(relative)
            else:
                files.append(relative)
    return files, sorted(excluded)


def _json_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON keys in archive manifest")
        result[key] = value
    return result


def _validate_manifest(value: object) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
    ):
        raise ValueError("unsupported archive manifest")
    if value.get("kind") not in {"source", "evidence"}:
        raise ValueError("unsupported archive purpose")
    allowed = {"schema_version", "kind", "files"}
    allowed.add("source_identity" if value["kind"] == "source" else "excluded_cache_paths")
    if set(value) != allowed:
        raise ValueError("unexpected or missing archive manifest fields")
    files = value.get("files")
    if not isinstance(files, list) or not 1 <= len(files) <= MAX_FILES:
        raise ValueError("invalid archive file inventory")
    names: set[str] = set()
    total = 0
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"}:
            raise ValueError("invalid inventory entry")
        name = _relative(entry["path"])
        if name in names or name == RESTORE_RECEIPT:
            raise ValueError("duplicate or reserved inventory path")
        names.add(name)
        if _secret(PurePosixPath(name)) or any(
            part in _CACHE_NAMES for part in PurePosixPath(name).parts
        ):
            raise ValueError("secret/cache path in archive")
        if value["kind"] == "source" and not _source_allowed(name):
            raise ValueError("source archive contains a non-allowlisted path")
        if type(entry["size"]) is not int or entry["size"] < 0:
            raise ValueError("invalid inventory size")
        total += entry["size"]
        if total > MAX_BYTES:
            raise ValueError("archive exceeds uncompressed size budget")
        if not isinstance(entry["sha256"], str) or not re.fullmatch(
            r"[0-9a-f]{64}", entry["sha256"]
        ):
            raise ValueError("invalid inventory digest")
    if any(str(parent) in names for name in names for parent in PurePosixPath(name).parents):
        raise ValueError("archive inventory has a file/directory path collision")
    if value["kind"] == "source":
        hashes = {
            entry["path"]: entry["sha256"] for entry in files if _identity_file(entry["path"])
        }
        identity = value.get("source_identity")
        if identity != {"sha256": _digest_json(hashes), "files": hashes}:
            raise ValueError("source identity does not match archived implementation bytes")
        if not {"olympus/__init__.py", "pyproject.toml", "requirements-kaggle.txt"} <= names:
            raise ValueError("source archive is missing its required files")
    return value


def _read_manifest(archive: tarfile.TarFile) -> dict[str, Any]:
    member = archive.next()
    if (
        member is None
        or member.name != MANIFEST_NAME
        or not member.isfile()
        or member.size > MAX_MANIFEST_BYTES
    ):
        raise ValueError("archive must begin with its bounded regular-file manifest")
    handle = archive.extractfile(member)
    if handle is None:
        raise ValueError("archive manifest is unreadable")
    with handle:
        value = json.load(handle, object_pairs_hook=_json_no_duplicates)
    return _validate_manifest(value)


def _stream_digest(handle: IO[bytes], destination: IO[bytes] | None = None) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
        if destination is not None:
            destination.write(chunk)
    return digest.hexdigest()


def _check_members(
    archive: tarfile.TarFile, manifest: dict[str, Any], target: Path | None = None
) -> None:
    expected = {f"files/{entry['path']}": entry for entry in manifest["files"]}
    seen: set[str] = set()
    while (member := archive.next()) is not None:
        if member.name in seen or member.name not in expected or not member.isfile():
            raise ValueError("archive contains duplicate, unexpected, or non-regular members")
        seen.add(member.name)
        entry = expected[member.name]
        if member.size != entry["size"] or member.sparse is not None:
            raise ValueError("archive member size/type differs from inventory")
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError("archive file is unreadable")
        with handle:
            if target is None:
                observed = _stream_digest(handle)
            else:
                path = target / entry["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as output:
                    observed = _stream_digest(handle, output)
                    output.flush()
                    os.fsync(output.fileno())
        if observed != entry["sha256"]:
            raise ValueError(f"archive file digest mismatch: {entry['path']}")
    if seen != set(expected):
        raise ValueError("archive is missing inventoried files")


def verify_archive(archive_path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ValueError("archive must be a regular non-symlink file")
    if archive_path.stat().st_size > MAX_BYTES + MAX_MANIFEST_BYTES + MAX_FILES * 1024:
        raise ValueError("compressed archive exceeds the bounded input size")
    observed = file_sha256(archive_path)
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError("archive SHA-256 differs from the independently pinned digest")
    with tarfile.open(archive_path, "r:gz") as archive:
        manifest = _read_manifest(archive)
        _check_members(archive, manifest)
    return {
        "archive_sha256": observed,
        "kind": manifest["kind"],
        "manifest": manifest,
        "source_sha256": manifest.get("source_identity", {}).get("sha256"),
        "inventory_sha256": _digest_json(manifest["files"]),
        "file_count": len(manifest["files"]),
        "total_bytes": sum(entry["size"] for entry in manifest["files"]),
    }


def _write_bundle(root: Path, output: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError("archive destination already exists")
    if output.resolve().is_relative_to(root):
        relative_output = output.resolve().relative_to(root).as_posix()
        if manifest["kind"] == "evidence" or _source_allowed(relative_output):
            raise ValueError("archive output must be outside the archived file selection")
    encoded = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise ValueError("archive inventory exceeds its size budget")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(dir=output.parent, prefix=".bundle-", delete=False) as raw:
            temporary = Path(raw.name)
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
                with tarfile.open(
                    fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
                ) as archive:
                    info = tarfile.TarInfo(MANIFEST_NAME)
                    info.size, info.mode, info.mtime = len(encoded), 0o644, 0
                    archive.addfile(info, io.BytesIO(encoded))
                    for entry in manifest["files"]:
                        info = tarfile.TarInfo(f"files/{entry['path']}")
                        info.size, info.mode, info.mtime = entry["size"], 0o644, 0
                        with _regular(root, entry["path"]).open("rb") as handle:
                            archive.addfile(info, handle)
            raw.flush()
            os.fsync(raw.fileno())
        result = verify_archive(temporary)
        os.link(temporary, output)  # Exclusive atomic publication; never replaces a prior archive.
        descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return {**result, "path": str(output.resolve())}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_source_bundle(repo_root: Path, output: Path) -> dict[str, Any]:
    if repo_root.is_symlink() or not repo_root.is_dir():
        raise ValueError("source root must be a regular directory")
    root = repo_root.resolve()
    files = _inventory(root, _source_paths(root))
    hashes = {entry["path"]: entry["sha256"] for entry in files if _identity_file(entry["path"])}
    manifest = {
        "schema_version": 1,
        "kind": "source",
        "files": files,
        "source_identity": {"sha256": _digest_json(hashes), "files": hashes},
    }
    return _write_bundle(root, output, manifest)


def archive_tree(root: Path, output: Path) -> dict[str, Any]:
    """Archive a designated evidence root, retaining partial/failed runs and every checkpoint."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("evidence root must be a regular directory")
    root = root.resolve()
    paths, excluded = _evidence_paths(root)
    manifest = {
        "schema_version": 1,
        "kind": "evidence",
        "files": _inventory(root, paths),
        "excluded_cache_paths": excluded,
    }
    return _write_bundle(root, output, manifest)


def restore_archive(
    archive_path: Path, target: Path, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    if target.exists() or target.is_symlink():
        raise FileExistsError("restore requires a fresh, nonexistent target directory")
    verified = verify_archive(archive_path, expected_sha256=expected_sha256)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            manifest = _read_manifest(archive)
            if manifest != verified["manifest"]:
                raise ValueError("archive changed between verification and restoration")
            _check_members(archive, manifest, target)
        if file_sha256(archive_path) != verified["archive_sha256"]:
            raise ValueError("archive changed during restoration")
        with (target / RESTORE_RECEIPT).open("x", encoding="utf-8") as handle:
            json.dump(verified, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return {**verified, "target": str(target.resolve())}
    except BaseException:
        shutil.rmtree(target)
        raise


def verify_restored(root: Path, *, expected_archive_sha256: str | None = None) -> dict[str, Any]:
    receipt = _regular(root.resolve(), RESTORE_RECEIPT)
    result = json.loads(receipt.read_text(encoding="utf-8"), object_pairs_hook=_json_no_duplicates)
    if not isinstance(result, dict):
        raise ValueError("restoration receipt must be an object")
    if (
        expected_archive_sha256 is not None
        and result.get("archive_sha256") != expected_archive_sha256
    ):
        raise ValueError("restored receipt differs from the pinned archive digest")
    manifest = _validate_manifest(result.get("manifest"))
    for entry in manifest["files"]:
        path = _regular(root.resolve(), entry["path"])
        if path.stat().st_size != entry["size"] or file_sha256(path) != entry["sha256"]:
            raise ValueError(f"restored file changed: {entry['path']}")
    return result
