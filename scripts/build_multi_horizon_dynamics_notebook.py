"""Build the private Kaggle notebook for the frozen multi-horizon campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(source: Path, protocol: Path, output: Path) -> None:
    source_text = source.read_text(encoding="utf-8")
    protocol_text = protocol.read_text(encoding="utf-8")
    cell = "\n".join(
        (
            "from pathlib import Path",
            "import os",
            "import runpy",
            "",
            "root = Path('/kaggle/working/multi_horizon_frozen_source')",
            "script = root / 'scripts/run_multi_horizon_dynamics_v1.py'",
            "protocol = root / 'research/multi_horizon_dynamics_v1/PROTOCOL.md'",
            "script.parent.mkdir(parents=True, exist_ok=True)",
            "protocol.parent.mkdir(parents=True, exist_ok=True)",
            f"script.write_text({source_text!r}, encoding='utf-8')",
            f"protocol.write_text({protocol_text!r}, encoding='utf-8')",
            "os.environ['MULTI_HORIZON_OUTPUT'] = "
            "'/kaggle/working/multi_horizon_dynamics_v1'",
            "runpy.run_path(str(script), run_name='__main__')",
        )
    )
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Multi-horizon dynamics v1\n",
                    "Private, frozen, CPU-only scientific execution. Internet must remain off.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [f"{line}\n" for line in cell.splitlines()],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    build(arguments.source, arguments.protocol, arguments.output)


if __name__ == "__main__":
    main()
