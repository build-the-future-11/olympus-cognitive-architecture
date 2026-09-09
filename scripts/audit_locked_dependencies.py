"""Audit the exact locked runtime dependency graph without re-resolving the project."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("columns", "json", "cyclonedx-json", "cyclonedx-xml"))
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    uv = shutil.which("uv")
    if uv is None:
        print("uv is required to export the locked dependency graph", file=sys.stderr)
        return 2

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="olympus-dependency-audit-") as temporary:
        requirements = Path(temporary) / "requirements.txt"
        export = subprocess.run(
            [
                uv,
                "export",
                "--locked",
                "--no-dev",
                "--no-emit-project",
                "--format",
                "requirements-txt",
                "--output-file",
                str(requirements),
                "--quiet",
            ],
            cwd=repository_root,
            check=False,
        )
        if export.returncode != 0:
            return export.returncode

        audit = [
            sys.executable,
            "-m",
            "pip_audit",
            "--strict",
            "-r",
            str(requirements),
            "--require-hashes",
            "--disable-pip",
        ]
        if args.format is not None:
            audit.extend(("--format", args.format))
        if args.output is not None:
            audit.extend(("--output", str(args.output.resolve())))
        return subprocess.run(audit, cwd=repository_root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
