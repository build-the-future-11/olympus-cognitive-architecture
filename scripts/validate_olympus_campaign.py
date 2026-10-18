"""Run bounded numerical regression checks only after the host resource gate admits them.

Launch from the checkout with ``python -m scripts.validate_olympus_campaign``.
This does not launch a research campaign, use a network service or acquire compute.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from olympus.foundry.resources import ResourceGovernor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--junitxml", type=Path, help="Persist numerical regression results")
    parser.add_argument(
        "--lock", type=Path, default=Path("artifacts/locks/campaign-validation.lock")
    )
    args = parser.parse_args()
    # Admission happens before importing pytest plugins or any tensor library.
    with ResourceGovernor(args.lock):
        os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import pytest
        import torch

        from olympus.models.runtime import resolve_olympus_device

        resolve_olympus_device(args.device)
        torch.set_num_threads(1)
        root = Path(__file__).resolve().parents[1]
        tests = [
            "test_olympus_system.py",
            "test_olympus_dataset.py",
            "test_controlled_fusion.py",
            "test_runtime_hardening.py",
            "test_olympus_resume.py",
        ]
        arguments = ["-q", *[str(root / "tests" / name) for name in tests]]
        if args.junitxml is not None:
            arguments.extend(["--junitxml", str(args.junitxml.resolve())])
        if args.device == "cuda":
            arguments.extend(["-k", "cuda"])
        raise SystemExit(pytest.main(arguments))


if __name__ == "__main__":
    main()
