"""Build deterministic Python distributions from the locked development environment."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path, PurePosixPath


def _source_date_epoch(explicit: int | None, repository_root: Path) -> int:
    if explicit is not None:
        epoch = explicit
    elif raw_epoch := os.environ.get("SOURCE_DATE_EPOCH"):
        try:
            epoch = int(raw_epoch)
        except ValueError as exc:
            raise ValueError("SOURCE_DATE_EPOCH must be an integer Unix timestamp") from exc
    else:
        completed = subprocess.run(
            ["git", "show", "-s", "--format=%ct", "HEAD"],
            cwd=repository_root,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            epoch = int(completed.stdout.strip())
        except ValueError as exc:
            raise ValueError("Git did not return a valid commit timestamp") from exc
    # ZIP timestamps cannot represent dates before 1980. Keeping the bound here
    # makes wheel-build failures explicit rather than platform-dependent.
    if epoch < 315_532_800:
        raise ValueError("source date epoch must be on or after 1980-01-01")
    return epoch


def _canonical_member_name(raw: str) -> str:
    if not raw or "\\" in raw or "\x00" in raw:
        raise ValueError(f"source distribution has an unsafe member path: {raw!r}")
    candidate = raw[:-1] if raw.endswith("/") else raw
    path = PurePosixPath(candidate)
    if (
        not candidate
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != candidate
    ):
        raise ValueError(f"source distribution has an unsafe member path: {raw!r}")
    return path.as_posix()


def normalize_source_distribution(path: Path, *, epoch: int) -> None:
    """Rewrite an sdist with stable order, ownership, modes, timestamps, and gzip header."""

    source = path.resolve(strict=True)
    members: list[tuple[tarfile.TarInfo, bytes | None]] = []
    seen: set[str] = set()
    with tarfile.open(source, mode="r:*") as archive:
        for member in archive.getmembers():
            name = _canonical_member_name(member.name)
            if name in seen:
                raise ValueError(f"source distribution has a duplicate member path: {name}")
            seen.add(name)
            if not (member.isfile() or member.isdir()):
                raise ValueError(
                    f"source distribution member must be a regular file or directory: {name}"
                )
            payload: bytes | None = None
            if member.isfile():
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"could not read source distribution member: {name}")
                payload = extracted.read()
            members.append((member, payload))

    temporary = source.with_name(f".{source.name}.normalized")
    try:
        with temporary.open("wb") as raw_output:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=raw_output,
                mtime=epoch,
            ) as compressed:
                with tarfile.open(
                    fileobj=compressed,
                    mode="w",
                    format=tarfile.PAX_FORMAT,
                ) as output:
                    for member, payload in sorted(members, key=lambda item: item[0].name):
                        normalized = tarfile.TarInfo(name=_canonical_member_name(member.name))
                        normalized.type = member.type
                        normalized.mode = (
                            0o755
                            if member.isdir() or member.mode & 0o111
                            else 0o644
                        )
                        normalized.uid = 0
                        normalized.gid = 0
                        normalized.uname = ""
                        normalized.gname = ""
                        normalized.mtime = epoch
                        if payload is not None:
                            normalized.size = len(payload)
                            output.addfile(normalized, BytesIO(payload))
                        else:
                            output.addfile(normalized)
        temporary.chmod(0o644)
        os.replace(temporary, source)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_once(repository_root: Path, *, epoch: int, output: Path) -> dict[str, Path]:
    environment = os.environ.copy()
    environment["SOURCE_DATE_EPOCH"] = str(epoch)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--outdir",
            str(output),
            str(repository_root),
        ],
        cwd=repository_root,
        env=environment,
        check=True,
    )
    wheels = sorted(output.glob("*.whl"))
    sdists = sorted(output.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError(
            "release build must produce exactly one wheel and one .tar.gz source distribution"
        )
    normalize_source_distribution(sdists[0], epoch=epoch)
    return {artifact.name: artifact for artifact in (*wheels, *sdists)}


def build_release_artifacts(
    *,
    repository_root: Path,
    output_directory: Path,
    epoch: int,
    verify_reproducible: bool,
) -> dict[str, str]:
    """Build into temporary directories and publish only a complete verified pair."""

    output = output_directory.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"release output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="olympus-release-build-") as temporary:
        temporary_root = Path(temporary)
        first = _build_once(repository_root, epoch=epoch, output=temporary_root / "first")
        first_hashes = {name: _sha256(path) for name, path in first.items()}
        if verify_reproducible:
            second = _build_once(repository_root, epoch=epoch, output=temporary_root / "second")
            second_hashes = {name: _sha256(path) for name, path in second.items()}
            if first_hashes != second_hashes:
                raise ValueError(
                    "release distributions are not reproducible across two clean builds: "
                    f"first={first_hashes}, second={second_hashes}"
                )
        for name, source in first.items():
            shutil.copyfile(source, output / name)
            (output / name).chmod(0o644)
    return first_hashes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("dist"))
    parser.add_argument("--source-date-epoch", type=int)
    parser.add_argument("--verify-reproducible", action="store_true")
    arguments = parser.parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    epoch = _source_date_epoch(arguments.source_date_epoch, repository_root)
    hashes = build_release_artifacts(
        repository_root=repository_root,
        output_directory=arguments.outdir,
        epoch=epoch,
        verify_reproducible=arguments.verify_reproducible,
    )
    print(
        json.dumps(
            {"source_date_epoch": epoch, "sha256": hashes},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
