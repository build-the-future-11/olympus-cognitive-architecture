"""Build distributions twice and retain them only when every byte is reproducible."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory


def _artifacts(directory: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def _normalize_source_archive(directory: Path) -> None:
    epoch_raw = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch_raw is None or not epoch_raw.isdigit():
        raise ValueError("SOURCE_DATE_EPOCH must be a non-negative integer")
    epoch = int(epoch_raw)
    archives = list(directory.glob("*.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected exactly one source archive, found {len(archives)}")
    archive = archives[0]
    original_tar = io.BytesIO(gzip.decompress(archive.read_bytes()))
    canonical_tar = io.BytesIO()
    with tarfile.open(fileobj=original_tar, mode="r:") as source, tarfile.open(
        fileobj=canonical_tar,
        mode="w:",
        format=tarfile.GNU_FORMAT,
    ) as destination:
        for member in sorted(source.getmembers(), key=lambda item: item.name):
            normalized = tarfile.TarInfo(member.name)
            normalized.type = member.type
            normalized.mode = member.mode
            normalized.mtime = epoch
            normalized.uid = 0
            normalized.gid = 0
            normalized.uname = ""
            normalized.gname = ""
            normalized.linkname = member.linkname
            if member.isfile():
                extracted = source.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"source archive member cannot be read: {member.name}")
                payload = extracted.read()
                normalized.size = len(payload)
                destination.addfile(normalized, io.BytesIO(payload))
            elif member.isdir() or member.issym() or member.islnk():
                normalized.size = 0
                destination.addfile(normalized)
            else:
                raise RuntimeError(f"unsupported source archive member type: {member.name}")
    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as compressed:
            compressed.write(canonical_tar.getvalue())


def verify_reproducible_build(output: Path) -> dict[str, str]:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"reproducible-build output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="olympus-build-a-") as first_raw, TemporaryDirectory(
        prefix="olympus-build-b-"
    ) as second_raw:
        first = Path(first_raw)
        second = Path(second_raw)
        for index, destination in enumerate((first, second), 1):
            print(f"building distribution pass {index}/2", flush=True)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--no-isolation",
                    "--outdir",
                    str(destination),
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"distribution build failed:\n{completed.stdout}")
            _normalize_source_archive(destination)
        first_hashes = _artifacts(first)
        second_hashes = _artifacts(second)
        if not first_hashes or first_hashes != second_hashes:
            raise RuntimeError(
                "distribution builds are not byte reproducible: "
                f"first={first_hashes} second={second_hashes}"
            )
        for name in first_hashes:
            shutil.copyfile(first / name, output / name)
    return first_hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("dist"))
    args = parser.parse_args()
    hashes = verify_reproducible_build(args.output)
    for name, digest in hashes.items():
        print(f"{digest}  {name}")


if __name__ == "__main__":
    main()
