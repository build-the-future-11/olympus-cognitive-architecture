"""Run real Foundry tests when optional web dependencies are unavailable.

Only the top-level package namespace is provided here, avoiding its eager API
import. Every tested Foundry/core module is loaded unchanged from source. This
does not verify the ordinary package import, API, CLI, or web integrations.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

root = Path(os.environ.get("OLYMPUS_SOURCE_ROOT", Path(__file__).resolve().parents[3]))
package = types.ModuleType("olympus")
package.__path__ = [str(root / "olympus")]
sys.modules["olympus"] = package
sys.path.insert(0, str(root))
raise SystemExit(pytest.main(sys.argv[1:]))
