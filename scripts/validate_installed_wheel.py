"""Exercise a built wheel without importing Olympus from the source checkout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

_CLI_BOOTSTRAP = """
from pathlib import Path
import sys

target = Path(sys.argv[1]).resolve()
source_root = Path(sys.argv[2]).resolve()
if Path.cwd().resolve().is_relative_to(source_root):
    raise RuntimeError("installed-wheel command ran inside the source checkout")
sys.path.insert(0, str(target))

import olympus

package_file = Path(olympus.__file__).resolve()
if not package_file.is_relative_to(target):
    raise RuntimeError(f"Olympus imported from {package_file}, not {target}")

from olympus.cli import app

sys.argv = ["olympus", *sys.argv[3:]]
app()
"""

_IDENTITY_BOOTSTRAP = """
import json
from importlib.metadata import version
from pathlib import Path
import sys

target = Path(sys.argv[1]).resolve()
source_root = Path(sys.argv[2]).resolve()
if Path.cwd().resolve().is_relative_to(source_root):
    raise RuntimeError("installed-wheel command ran inside the source checkout")
sys.path.insert(0, str(target))

import olympus
from olympus.models.composition_smoke import reference_replay_source_sha256

package_file = Path(olympus.__file__).resolve()
if not package_file.is_relative_to(target):
    raise RuntimeError(f"Olympus imported from {package_file}, not {target}")
print(json.dumps({
    "package_file": str(package_file),
    "version": version("olympus-cognitive-architecture"),
    "source_sha256": reference_replay_source_sha256(),
}))
"""

_WORKER_BOOTSTRAP = """
import json
from pathlib import Path
import sys

target = Path(sys.argv[1]).resolve()
source_root = Path(sys.argv[2]).resolve()
if Path.cwd().resolve().is_relative_to(source_root):
    raise RuntimeError("installed worker ran inside the source checkout")
sys.path.insert(0, str(target))
from olympus.foundry import worker
from olympus.foundry.service import FoundryService

if not Path(worker.__file__).resolve().is_relative_to(target):
    raise RuntimeError("worker did not import from installed wheel")
root = Path.cwd() / "isolated-foundry"
with FoundryService(root) as service:
    result = worker.run_isolated_verification(
        root, Path(worker.__file__).with_name("foundry_verification.txt"),
        lambda: False, timeout_seconds=30,
    )
    if result["evaluation"]["passed"] is not True:
        raise RuntimeError("installed worker evaluation failed")
    status = service.status()
if list(root.glob(".worker-*")):
    raise RuntimeError("worker left temporary output directories")
print(json.dumps(status))
"""


def _isolated_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_CACHE_DIR": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def _run(
    command: Sequence[str], *, cwd: Path, environment: Mapping[str, str],
    timeout_seconds: float = 120,
) -> str:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(environment),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
        timeout=timeout_seconds,
    )
    return completed.stdout


def _json_object(raw: str, *, label: str) -> dict[str, object]:
    value: object = json.loads(raw)
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} did not return a JSON object")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _run_installed_cli(
    target: Path,
    source_root: Path,
    run_root: Path,
    arguments: Sequence[str],
    *,
    environment: Mapping[str, str],
) -> dict[str, object]:
    stdout = _run(
        [
            sys.executable,
            "-I",
            "-c",
            _CLI_BOOTSTRAP,
            str(target),
            str(source_root),
            *arguments,
        ],
        cwd=run_root,
        environment=environment,
    )
    return _json_object(stdout, label="installed-wheel CLI")


def _select_temporary_parent(source_root: Path, requested: Path | None) -> Path:
    candidates = [requested] if requested is not None else []
    runner_temp = os.environ.get("RUNNER_TEMP")
    if runner_temp:
        candidates.append(Path(runner_temp))
    candidates.extend((Path(tempfile.gettempdir()), Path("/tmp")))
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.expanduser().resolve()
        if resolved == source_root or resolved.is_relative_to(source_root):
            continue
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved
    raise ValueError("no temporary directory outside the source checkout is available")


def validate_installed_wheel(
    wheel_path: Path,
    *,
    temporary_parent: Path | None = None,
) -> dict[str, object]:
    """Install and exercise one wheel in fresh processes outside the checkout."""

    wheel = wheel_path.resolve(strict=True)
    source_root = Path(__file__).resolve().parents[1]
    parent = _select_temporary_parent(source_root, temporary_parent)
    environment = _isolated_environment()

    with tempfile.TemporaryDirectory(prefix="olympus-wheel-gate-", dir=parent) as raw_root:
        run_root = Path(raw_root).resolve()
        target = run_root / "target"
        target.mkdir()
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-compile",
                "--no-deps",
                "--no-index",
                "--target",
                str(target),
                str(wheel),
            ],
            cwd=run_root,
            environment=environment,
        )

        identity = _json_object(
            _run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    _IDENTITY_BOOTSTRAP,
                    str(target),
                    str(source_root),
                ],
                cwd=run_root,
                environment=environment,
            ),
            label="installed-wheel identity",
        )
        status = _run_installed_cli(
            target,
            source_root,
            run_root,
            ["models", "status"],
            environment=environment,
        )
        families = _sequence(status.get("families"), label="model families")
        if len(families) != 6:
            raise ValueError(f"installed wheel reported {len(families)} model families, expected 6")
        summary = _mapping(status.get("summary"), label="model status summary")
        if summary.get("implementations") != "IMPLEMENTED":
            raise ValueError("installed wheel does not report implemented reference components")

        composition_root = run_root / "composition-smoke"
        composition = _run_installed_cli(
            target,
            source_root,
            run_root,
            [
                "models",
                "composition-smoke",
                "--output-dir",
                str(composition_root),
                "--seed",
                "20260906",
            ],
            environment=environment,
        )
        checks = _mapping(
            composition.get("deterministic_checks"),
            label="composition deterministic checks",
        )
        if composition.get("passed") is not True or not checks:
            raise ValueError("installed-wheel composition smoke did not pass")
        if any(value is not True for value in checks.values()):
            raise ValueError("installed-wheel composition smoke has a failed deterministic check")
        if composition.get("promotion_authorized") is not False:
            raise ValueError("composition smoke unexpectedly authorized promotion")
        if composition.get("qualifying_result") is not False:
            raise ValueError("composition smoke unexpectedly reported a qualifying result")
        if composition.get("source_sha256") != identity.get("source_sha256"):
            raise ValueError("installed identity and composition manifest source digests differ")

        foundry_root = run_root / "foundry"
        foundry_run = _run_installed_cli(
            target,
            source_root,
            run_root,
            ["foundry", "verify-pipeline", "--root", str(foundry_root)],
            environment=environment,
        )
        evaluation = _mapping(foundry_run.get("evaluation"), label="Foundry evaluation")
        if evaluation.get("passed") is not True:
            raise ValueError("installed-wheel Foundry verification did not pass")

        # This intentionally starts another interpreter to prove persisted state reopens.
        foundry_status = _run_installed_cli(
            target,
            source_root,
            run_root,
            ["foundry", "status", "--root", str(foundry_root)],
            environment=environment,
        )
        expected_counts = {
            "datasets": 1,
            "experiments": 1,
            "checkpoints": 1,
            "evaluations": 1,
            "models": 1,
            "evidence_events": 5,
        }
        if foundry_status.get("integrity") != "ok" or any(
            foundry_status.get(key) != value for key, value in expected_counts.items()
        ):
            raise ValueError("fresh-process Foundry status did not reopen the verified state")

        worker_status = _json_object(
            _run(
                [sys.executable, "-I", "-c", _WORKER_BOOTSTRAP,
                 str(target), str(source_root)],
                cwd=run_root, environment=environment,
            ),
            label="installed-wheel worker",
        )
        if worker_status.get("integrity") != "ok" or any(
            worker_status.get(key) != value for key, value in expected_counts.items()
        ):
            raise ValueError("installed-wheel worker did not persist the complete lifecycle")

        package_file = identity.get("package_file")
        version = identity.get("version")
        source_sha256 = identity.get("source_sha256")
        if not all(isinstance(value, str) and value for value in (package_file, version)):
            raise ValueError("installed-wheel identity is incomplete")
        if not isinstance(source_sha256, str) or len(source_sha256) != 64:
            raise ValueError("installed-wheel source digest is invalid")
        return {
            "wheel": wheel.name,
            "package_file": package_file,
            "version": version,
            "model_families": len(families),
            "composition_checks": len(checks),
            "composition_source_sha256": source_sha256,
            "foundry_status": foundry_status,
            "isolated_worker_status": worker_status,
            "passed": True,
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--temporary-parent", type=Path)
    arguments = parser.parse_args(argv)
    result = validate_installed_wheel(
        arguments.wheel,
        temporary_parent=arguments.temporary_parent,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
