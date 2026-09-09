"""Match built package/test/example Python payloads to their tracked sources."""

from __future__ import annotations

import argparse
import stat
import subprocess
import tarfile
import zipfile
from collections.abc import Sequence, Set
from pathlib import Path, PurePosixPath

_PYTHON_PAYLOAD_PREFIXES = ("olympus/", "tests/", "examples/", "scripts/")
_FORBIDDEN_WHEEL_SUFFIXES = (".dll", ".dylib", ".pth", ".pyd", ".so")


def _is_python_payload(path: str) -> bool:
    return path.endswith(".py") and path.startswith(_PYTHON_PAYLOAD_PREFIXES)


def _safe_posix_path(raw: str, *, directory: bool = False) -> PurePosixPath:
    candidate = raw[:-1] if directory and raw.endswith("/") else raw
    path = PurePosixPath(candidate)
    if (
        not candidate
        or "\\" in candidate
        or "\x00" in candidate
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != candidate
    ):
        raise ValueError(f"distribution contains an unsafe member path: {raw!r}")
    return path


def _reject_duplicate_path(path: PurePosixPath, *, seen: set[str]) -> None:
    canonical = path.as_posix()
    if canonical in seen:
        raise ValueError(f"distribution contains a duplicate member path: {canonical}")
    seen.add(canonical)


def _reject_internal_outreach_document(path: PurePosixPath) -> None:
    if (
        len(path.parts) >= 2
        and path.parts[0] == "docs"
        and path.suffix.lower() == ".md"
        and "OUTREACH" in path.name.upper()
    ):
        raise ValueError(f"source distribution contains an internal outreach document: {path}")


def _tracked_paths(repository_root: Path) -> set[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository_root,
        check=True,
        stdout=subprocess.PIPE,
    )
    paths = completed.stdout.decode("utf-8", errors="surrogateescape").split("\0")
    return {path for path in paths if path}


def _wheel_paths(wheel_path: Path) -> set[str]:
    with zipfile.ZipFile(wheel_path) as archive:
        members: list[PurePosixPath] = []
        seen: set[str] = set()
        for member in archive.infolist():
            path = _safe_posix_path(member.filename, directory=member.is_dir())
            _reject_duplicate_path(path, seen=seen)
            unix_mode = member.external_attr >> 16
            if unix_mode and stat.S_ISLNK(unix_mode):
                raise ValueError(f"wheel contains a symbolic link: {path}")
            if not member.is_dir():
                if path.suffix.lower() in _FORBIDDEN_WHEEL_SUFFIXES:
                    raise ValueError(f"pure-Python wheel contains a forbidden payload: {path}")
                if any(part.endswith(".data") for part in path.parts) and "scripts" in path.parts:
                    raise ValueError(f"wheel contains an undeclared installed script: {path}")
                members.append(path)
    return {path.as_posix() for path in members}


def _sdist_paths(sdist_path: Path) -> set[str]:
    with tarfile.open(sdist_path, mode="r:*") as archive:
        members: list[PurePosixPath] = []
        seen: set[str] = set()
        for member in archive.getmembers():
            path = _safe_posix_path(member.name, directory=member.isdir())
            _reject_duplicate_path(path, seen=seen)
            if not (member.isfile() or member.isdir()):
                raise ValueError(
                    "source distribution members must be regular files or directories: "
                    f"{path}"
                )
            if member.isfile():
                members.append(path)
    roots = {path.parts[0] for path in members if path.parts}
    if len(roots) != 1:
        raise ValueError("source distribution must contain exactly one top-level directory")
    paths: set[str] = set()
    for member_path in members:
        if len(member_path.parts) < 2:
            continue
        relative_path = PurePosixPath(*member_path.parts[1:])
        _reject_internal_outreach_document(relative_path)
        relative = relative_path.as_posix()
        if relative_path.suffix.lower() == ".pth":
            raise ValueError(f"source distribution contains a forbidden .pth file: {relative}")
        if relative_path.suffix.lower() == ".py" and not _is_python_payload(relative):
            raise ValueError(
                f"source distribution contains Python outside approved roots: {relative}"
            )
        paths.add(relative)
    return paths


_GENERATED_SDIST_ROOT_FILES = {"PKG-INFO", "setup.cfg"}
_GENERATED_EGG_INFO_FILES = {
    "PKG-INFO",
    "SOURCES.txt",
    "dependency_links.txt",
    "entry_points.txt",
    "requires.txt",
    "top_level.txt",
}
_GENERATED_DIST_INFO_FILES = {
    "METADATA",
    "RECORD",
    "WHEEL",
    "entry_points.txt",
    "licenses/LICENSE",
    "top_level.txt",
}


def _is_generated_sdist_path(path: str) -> bool:
    parsed = PurePosixPath(path)
    if path in _GENERATED_SDIST_ROOT_FILES:
        return True
    return (
        len(parsed.parts) == 2
        and parsed.parts[0] == "olympus_cognitive_architecture.egg-info"
        and parsed.parts[1] in _GENERATED_EGG_INFO_FILES
    )


def _is_generated_wheel_path(path: str) -> bool:
    parsed = PurePosixPath(path)
    if len(parsed.parts) < 2:
        return False
    root = parsed.parts[0]
    relative = PurePosixPath(*parsed.parts[1:]).as_posix()
    if root.startswith("olympus_cognitive_architecture-") and root.endswith(".dist-info"):
        return relative in _GENERATED_DIST_INFO_FILES
    return (
        root.startswith("olympus_cognitive_architecture-")
        and root.endswith(".data")
        and relative == "data/share/olympus/datasets/source.jsonl"
    )


def _payload_differences(
    *, label: str, expected: Set[str], actual: Set[str]
) -> list[str]:
    issues: list[str] = []
    unexpected = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unexpected:
        issues.append(f"unexpected {label} Python payload: {', '.join(unexpected)}")
    if missing:
        issues.append(f"missing {label} Python payload: {', '.join(missing)}")
    return issues


def validate_distribution_payload(
    wheel_path: Path,
    sdist_path: Path,
    *,
    repository_root: Path,
    tracked_python_paths: Set[str] | None = None,
) -> None:
    """Require configured wheel/sdist Python scopes to match tracked sources exactly."""

    tracked = (
        set(tracked_python_paths)
        if tracked_python_paths is not None
        else _tracked_paths(repository_root)
    )
    wheel_paths = _wheel_paths(wheel_path)
    sdist_paths = _sdist_paths(sdist_path)
    expected_wheel = {path for path in tracked if path.startswith("olympus/")}
    actual_wheel = {path for path in wheel_paths if path.startswith("olympus/")}
    expected_sdist = {path for path in tracked if _is_python_payload(path)}
    actual_sdist = {path for path in sdist_paths if _is_python_payload(path)}
    issues = [
        *_payload_differences(
            label="wheel", expected=expected_wheel, actual=actual_wheel
        ),
        *_payload_differences(
            label="sdist", expected=expected_sdist, actual=actual_sdist
        ),
    ]
    untracked_wheel = sorted(
        path
        for path in wheel_paths - actual_wheel
        if not _is_generated_wheel_path(path)
    )
    untracked_sdist = sorted(
        path
        for path in sdist_paths - tracked
        if not _is_generated_sdist_path(path)
    )
    if untracked_wheel:
        issues.append(f"unexpected non-package wheel payload: {', '.join(untracked_wheel)}")
    if untracked_sdist:
        issues.append(f"untracked source-distribution payload: {', '.join(untracked_sdist)}")
    if issues:
        raise ValueError("\n".join(issues))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("sdist", type=Path)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    arguments = parser.parse_args(argv)
    validate_distribution_payload(
        arguments.wheel.resolve(),
        arguments.sdist.resolve(),
        repository_root=arguments.repository_root.resolve(),
    )
    print("distribution Python payload matches tracked sources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
