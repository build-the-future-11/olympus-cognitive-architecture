from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "entrypoint",
    [
        "from olympus import app; assert app.title == 'Olympus Forge API'",
        "from olympus.cli import app; assert app is not None",
    ],
)
def test_foundry_entrypoints_do_not_load_tensor_backends(entrypoint: str) -> None:
    # Refuse Torch before it can allocate anything. This also makes the
    # predecessor regression safe to run on a memory-constrained host.
    script = """
import importlib.abc
import sys

class RejectTensorBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise RuntimeError('unrelated tensor backend loaded during Foundry startup')
        return None

sys.meta_path.insert(0, RejectTensorBackend())
""" + entrypoint
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository,
        env={**os.environ, "PYTHONPATH": str(repository)},
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
