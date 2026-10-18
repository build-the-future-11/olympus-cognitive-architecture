"""Build the maintained, offline Olympus Kaggle smoke notebook and source bundle.

Run from the checkout: python -m notebooks.build_kaggle_runner --output-dir <new-dir>.
The dated, superseded launch draft is intentionally preserved.
"""

from __future__ import annotations

import argparse
import base64
import json
import textwrap
from pathlib import Path
from typing import Any

from notebooks.kaggle_support.artifacts import build_source_bundle


def build(output_dir: Path, *, device: str = "cpu", embed_source: bool = False) -> Path:
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    output_dir.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    bundle = output_dir / "olympus-source.tar.gz"
    receipt = build_source_bundle(repo, bundle)
    (output_dir / "source-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    cells: list[dict[str, Any]] = []

    def cell(kind: str, source: str) -> None:
        source = textwrap.dedent(source).strip() + "\n"
        value: dict[str, Any] = {
            "cell_type": kind,
            "id": f"cell-{len(cells):03d}",
            "metadata": {},
            "source": source,
        }
        if kind == "code":
            compile(source, "<kaggle-cell>", "exec")
            value.update(execution_count=None, outputs=[])
        cells.append(value)

    cell(
        "markdown",
        """
        # Olympus Kaggle runner

        Attach the accompanying **olympus-source.tar.gz** as a private Kaggle input.
        This notebook runs an eight-step synthetic engineering smoke, not a scientific
        confirmation. Set DEVICE to `cuda` only in a GPU session; CPU is the default.
        Keep the image's CUDA Torch. Do not pip-install this repository: its full app
        requires Python 3.14; this source-only runner supports Python 3.11+.
        Missing dependencies are reported before execution; requirements-kaggle.txt
        lists the minimal environment. No network or credentials are required.

        All outputs live in /kaggle/working. Save/download them before ending the
        session. A stopped Python process does not stop the platform GPU allocation.
        The six-cell development matrix is opt-in after smoke validation.
        Re-run setup after a kernel restart. To resume in a new session, attach the
        recovery archive, enter its SHA256 below, and keep the same source and device.
        Historical negative studies remain unchanged.
    """,
    )
    cell(
        "code",
        f"""
        from pathlib import Path
        import hashlib, json, os, subprocess, sys, tarfile, time, types

        DEVICE = "cpu"  # "cuda" requires an enabled GPU; no fallback.
        RUN_DEVELOPMENT = False
        SOURCE_ARCHIVE = None  # Optional exact /kaggle/input/.../olympus-source.tar.gz
        RECOVERY_ARCHIVE = None  # Optional attached recovery archive in a fresh session.
        RECOVERY_SHA256 = None  # Required external digest when restoring.
        EXPECTED_SHA256 = {receipt["archive_sha256"]!r}
        BASE = Path("/kaggle/working/olympus")
        SOURCE, RESULTS = BASE / "source", BASE / "results"
        if DEVICE not in {{"cpu", "cuda"}}:
            raise ValueError("DEVICE must be cpu or cuda")
        if sys.version_info < (3, 11):
            raise RuntimeError("Python 3.11+ required")
        if SOURCE_ARCHIVE is None:
            candidates = list(Path("/kaggle/input").rglob("olympus-source.tar.gz"))
            if len(candidates) != 1:
                raise RuntimeError("Attach one source bundle or set SOURCE_ARCHIVE explicitly")
            SOURCE_ARCHIVE = candidates[0]
        SOURCE_ARCHIVE = Path(SOURCE_ARCHIVE)
        def sha(path):
            with Path(path).open("rb") as stream:
                return hashlib.file_digest(stream, "sha256").hexdigest()
        if sha(SOURCE_ARCHIVE) != EXPECTED_SHA256:
            raise RuntimeError("Source archive checksum mismatch")
        # Only execute the verifier from the independently hash-pinned archive.
        with tarfile.open(SOURCE_ARCHIVE, "r:gz") as archive:
            member = archive.getmember("files/notebooks/kaggle_support/artifacts.py")
            if not member.isfile() or member.size > 100000:
                raise ValueError("Invalid archive verifier")
            verifier_source = archive.extractfile(member).read().decode()
        verifier = types.ModuleType("olympus_archive_verifier")
        sys.modules[verifier.__name__] = verifier
        exec(compile(verifier_source, "<pinned-verifier>", "exec"), verifier.__dict__)
        if not SOURCE.exists():
            verifier.restore_archive(SOURCE_ARCHIVE, SOURCE, expected_sha256=EXPECTED_SHA256)
        verifier.verify_restored(SOURCE, expected_archive_sha256=EXPECTED_SHA256)
        if RECOVERY_ARCHIVE is not None:
            if not RECOVERY_SHA256:
                raise ValueError("Recovery requires its independently saved SHA256")
            if RESULTS.exists():
                raise FileExistsError("Restore needs fresh results; clear RECOVERY_ARCHIVE "
                                      "to continue existing results")
            verifier.restore_archive(Path(RECOVERY_ARCHIVE), RESULTS,
                                     expected_sha256=RECOVERY_SHA256)
        RESULTS.mkdir(parents=True, exist_ok=True)
        ENV = dict(os.environ, PYTHONPATH=str(SOURCE), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                   CUBLAS_WORKSPACE_CONFIG=":4096:8")
        def command(*args, allowed=(0,)):
            result = subprocess.run([sys.executable, *map(str, args)], cwd=SOURCE, env=ENV)
            if result.returncode not in allowed:
                raise RuntimeError(f"Command failed ({{result.returncode}}): {{args}}")
            return result.returncode
        def campaign(action, root, *args, **kwargs):
            return command("-m", "olympus.training.campaign", action,
                           "--run-root", root / "runs", *args, **kwargs)
        def persist():
            archive = BASE / f"recovery-{{time.time_ns()}}.tar.gz"
            verifier.archive_tree(RESULTS, archive)
            print("SAVE/DOWNLOAD recovery:", archive, "SHA256:", sha(archive), flush=True)
        print("Verified source:", EXPECTED_SHA256)
    """,
    )
    cells[1]["source"] = cells[1]["source"].replace('DEVICE = "cpu"', f'DEVICE = "{device}"', 1)
    if embed_source:
        encoded = base64.b64encode(bundle.read_bytes()).decode("ascii")
        cells[0]["source"] = cells[0]["source"].replace(
            "Attach the accompanying **olympus-source.tar.gz** as a private Kaggle input.",
            "The verified source archive is embedded; no separate input upload is needed.",
        )
        cells[1]["source"] = (
            "from pathlib import Path\nimport base64\n"
            "Path('/kaggle/working/olympus-source.tar.gz').write_bytes("
            f"base64.b64decode({encoded!r}))\n"
            + cells[1]["source"].replace(
                "SOURCE_ARCHIVE = None",
                "SOURCE_ARCHIVE = '/kaggle/working/olympus-source.tar.gz'",
                1,
            )
        )
    cell(
        "code",
        """
        # Admission precedes all Torch imports, including regression test collection.
        campaign("preflight", RESULTS / "preflight", "--device", DEVICE)
        try:
            command("-m", "scripts.validate_olympus_campaign", "--device", DEVICE,
                    "--junitxml", RESULTS / ("regressions-" + DEVICE + ".xml"))
        finally:
            if any(RESULTS.rglob("*.xml")):
                persist()
    """,
    )
    cell(
        "code",
        """
        def execute_profile(profile):
            root = RESULTS / (profile + "-" + DEVICE)
            root.mkdir(parents=True, exist_ok=True)
            template = SOURCE / "configs/kaggle" / (profile + "-" + DEVICE + ".json")
            manifest = root / "campaign.json"
            if manifest.exists():
                if manifest.read_bytes() != template.read_bytes():
                    raise RuntimeError("Existing manifest differs; use a new results directory")
            else:
                manifest.write_bytes(template.read_bytes())
            try:
                campaign("campaign", root, "--manifest", manifest, "--dry-run")
                if not (root / "data").exists():
                    campaign("prepare-" + profile, root, "--output", root / "data")
                campaign("campaign", root, "--manifest", manifest)
                campaign("analyze", root)
            finally:
                # Includes partial checkpoints and data on failures, not just completions.
                persist()
        execute_profile("smoke")
    """,
    )
    cell(
        "code",
        """
        if RUN_DEVELOPMENT:
            if DEVICE != "cuda":
                raise ValueError("The development manifest requires cuda")
            execute_profile("development")
        else:
            print("Smoke finished. Development matrix disabled.")
        print("Save outputs and explicitly stop the Kaggle session when finished.")
    """,
    )
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"isGpuEnabled": False, "isInternetEnabled": False},
        },
    }
    output = output_dir / "olympus_kaggle.ipynb"
    output.write_text(json.dumps(notebook, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--embed-source", action="store_true", help="Embed the verified source bundle"
    )
    args = parser.parse_args()
    print(build(args.output_dir, device=args.device, embed_source=args.embed_source))
