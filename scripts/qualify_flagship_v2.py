"""Run bounded artificial flagship qualification after host admission.

Usage: python -m scripts.qualify_flagship_v2 --output NEW_DIRECTORY --device cpu
The same command works offline on Kaggle; use cuda only for an allocated GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from olympus.foundry.resources import ResourceGovernor
from olympus.training.provenance import atomic_json, file_sha256, source_identity


def _macos_vm_counters() -> tuple[int, int]:
    result = subprocess.run(
        ["vm_stat"], check=True, capture_output=True, text=True, timeout=5
    ).stdout
    page_size = re.search(r"page size of (\d+) bytes", result)
    free = re.search(r"^Pages free:\s+(\d+)\.", result, re.MULTILINE)
    pageouts = re.search(r"^Pageouts:\s+(\d+)\.", result, re.MULTILINE)
    if page_size is None or free is None or pageouts is None:
        raise RuntimeError("cannot inspect macOS free pages and pageouts")
    return int(page_size[1]) * int(free[1]), int(pageouts[1])


def local_tensor_admission() -> dict[str, object]:
    """Retain the local 512 MiB/stable-pageout floor before loading Torch.

    This is additional to (not a replacement for) ResourceGovernor's 2 GiB
    available-memory and 90 percent swap ceilings. Linux uses that governor's
    MemAvailable admission; raw macOS free pages are not interchangeable with it.
    """
    if platform.system() != "Darwin":
        return {"platform": platform.system(), "extra_macos_guard": "NOT_APPLICABLE"}
    first = _macos_vm_counters()
    time.sleep(1)
    second = _macos_vm_counters()
    if min(first[0], second[0]) < 512 * 1024**2 or second[1] > first[1]:
        raise RuntimeError(
            f"local tensor admission refused: free bytes {first[0]}, {second[0]}; "
            f"pageouts {first[1]}, {second[1]}; require >=512 MiB and no rising pageouts"
        )
    return {
        "platform": "Darwin",
        "free_bytes": [first[0], second[0]],
        "pageouts": list((first[1], second[1])),
    }


def qualify(output: Path, *, device: str = "cpu", wall_seconds: int = 120) -> dict[str, Any]:
    if device not in {"cpu", "cuda"} or not 10 <= wall_seconds <= 120:
        raise ValueError("device must be cpu/cuda and wall budget must be 10..120 seconds")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    protocol = root / "research/flagship_engineering_v2/PROTOCOL.md"
    source = source_identity()
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "study_id": "flagship-engineering-v2",
        "status": "STARTED",
        "device_requested": device,
        "source": source,
        "protocol_sha256": file_sha256(protocol),
        "regression_source_sha256": file_sha256(root / "tests/test_flagship_v2.py"),
        "max_wall_seconds": wall_seconds,
        "scientific_evaluation": False,
        "promotion_authorized": False,
        "target_scale_training": False,
    }
    atomic_json(output / "receipt.json", receipt)
    try:
        receipt["local_tensor_admission"] = local_tensor_admission()
        with ResourceGovernor(root / "artifacts/locks/flagship-qualification.lock") as governor:
            receipt["resource_preflight"] = (
                governor.preflight.as_dict() if governor.preflight else None
            )
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            import torch

            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable; CPU fallback is forbidden")
            receipt["runtime"] = {
                "python": sys.version,
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "device": device,
                "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
            }
            tests = [str(root / "tests/test_flagship_v2.py")]
            if device == "cpu":
                tests += ["-k", "not cuda"]
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        *tests,
                        "--junitxml",
                        str(output.resolve() / "regressions.xml"),
                    ],
                    cwd=root,
                    env=dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"),
                    capture_output=True,
                    text=True,
                    timeout=max(1, wall_seconds - (time.monotonic() - started)),
                )
            except subprocess.TimeoutExpired as error:
                stdout = error.stdout or b""
                stderr = error.stderr or b""
                (output / "regressions.log").write_bytes(
                    (stdout.encode() if isinstance(stdout, str) else stdout)
                    + (stderr.encode() if isinstance(stderr, str) else stderr)
                )
                raise
            (output / "regressions.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(f"flagship regressions failed with exit {result.returncode}")
            # Artificial optimizer qualification, no protected dataset or benchmark.
            from scripts.flagship_smoke import train_smoke

            receipt["optimizer_smoke"] = train_smoke(
                output / "smoke.pt",
                device=device,
                deadline=started + wall_seconds,
            )
            if time.monotonic() - started > wall_seconds:
                raise TimeoutError("qualification exceeded the frozen wall budget")
            receipt["status"] = "ENGINEERING_QUALIFIED"
    except Exception as error:
        receipt["status"] = "FAILED_OR_RESOURCE_BLOCKED"
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["artifacts"] = {
            str(path.relative_to(output)): file_sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "receipt.json"
        }
        atomic_json(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    print(json.dumps(qualify(args.output, device=args.device), indent=2))


if __name__ == "__main__":
    main()
