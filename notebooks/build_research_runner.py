"""Build exact-source Kaggle notebooks for the frozen v5 and EditCLEVR studies."""

from __future__ import annotations

import argparse
import base64
import json
import textwrap
from pathlib import Path
from typing import Any

from notebooks.kaggle_support.artifacts import build_source_bundle

CAMPAIGNS = ("v5", "editclevr")


def build(output_dir: Path, *, campaign: str, embed_source: bool = True) -> Path:
    if campaign not in CAMPAIGNS:
        raise ValueError(f"campaign must be one of {CAMPAIGNS}")
    output_dir.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    bundle = output_dir / "olympus-source.tar.gz"
    receipt = build_source_bundle(repo, bundle)
    (output_dir / "source-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    cells: list[dict[str, Any]] = []

    def add(kind: str, source: str) -> None:
        normalized = textwrap.dedent(source).strip() + "\n"
        cell: dict[str, Any] = {
            "cell_type": kind,
            "id": f"cell-{len(cells):03d}",
            "metadata": {},
            "source": normalized,
        }
        if kind == "code":
            compile(normalized, "<research-runner-cell>", "exec")
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    title = "Robust fusion v5" if campaign == "v5" else "EditCLEVR v1"
    add(
        "markdown",
        f"""
        # Olympus {title}

        This notebook restores a checksummed source snapshot, runs qualification,
        executes the frozen protocol, verifies the resulting evidence, and exports
        a checksummed recovery archive. A successful process is engineering evidence;
        the protocol's primary predicate decides whether the scientific hypothesis
        is supported. Save the archive before stopping the session.
        """,
    )
    embedded = base64.b64encode(bundle.read_bytes()).decode("ascii") if embed_source else None
    source_setup = (
        "Path('/kaggle/working/olympus-source.tar.gz').write_bytes("
        f"base64.b64decode({embedded!r}))\n"
        "SOURCE_ARCHIVE = Path('/kaggle/working/olympus-source.tar.gz')"
        if embedded is not None
        else """
        candidates = list(Path('/kaggle/input').rglob('olympus-source.tar.gz'))
        if len(candidates) != 1:
            raise RuntimeError('Attach exactly one olympus-source.tar.gz input')
        SOURCE_ARCHIVE = candidates[0]
        """
    )
    source_setup = textwrap.dedent(source_setup).strip()
    setup_cell = textwrap.dedent(
        f"""
        from pathlib import Path
        import base64, hashlib, json, os, subprocess, sys, tarfile, types

        __SOURCE_SETUP__
        EXPECTED_SHA256 = {receipt['archive_sha256']!r}
        BASE = Path('/kaggle/working/olympus-{campaign}')
        SOURCE = BASE / 'source'
        def sha(path):
            with Path(path).open('rb') as stream:
                return hashlib.file_digest(stream, 'sha256').hexdigest()
        if sha(SOURCE_ARCHIVE) != EXPECTED_SHA256:
            raise RuntimeError('source archive checksum mismatch')
        with tarfile.open(SOURCE_ARCHIVE, 'r:gz') as archive:
            member = archive.getmember('files/notebooks/kaggle_support/artifacts.py')
            verifier_source = archive.extractfile(member).read().decode()
        verifier = types.ModuleType('olympus_archive_verifier')
        sys.modules[verifier.__name__] = verifier
        exec(compile(verifier_source, '<pinned-verifier>', 'exec'), verifier.__dict__)
        verifier.restore_archive(SOURCE_ARCHIVE, SOURCE, expected_sha256=EXPECTED_SHA256)
        verifier.verify_restored(SOURCE, expected_archive_sha256=EXPECTED_SHA256)
        ENV = dict(os.environ, PYTHONPATH=str(SOURCE), PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8')
        def command(*arguments):
            result = subprocess.run([sys.executable, *map(str, arguments)], cwd=SOURCE, env=ENV)
            if result.returncode:
                raise RuntimeError(f'command failed {{result.returncode}}: {{arguments}}')
        print(json.dumps({{'source_archive_sha256': EXPECTED_SHA256,
                          'source_identity_sha256': {receipt['source_sha256']!r}}}, indent=2))
        """
    ).replace("__SOURCE_SETUP__", source_setup)
    add(
        "code",
        setup_cell,
    )
    if campaign == "v5":
        add(
            "code",
            """
            qualification = SOURCE / 'artifacts/v5-qualification'
            result = SOURCE / 'artifacts/robust-fusion-v5'
            command('-m', 'scripts.run_robust_fusion_v5', '--output', qualification, 'test')
            command('-m', 'scripts.run_robust_fusion_v5', '--output', result, 'run')
            command('-m', 'scripts.verify_robust_fusion_v5', SOURCE)
            """,
        )
    else:
        add(
            "code",
            """
            registries = list(Path('/kaggle/input').rglob('splits.json'))
            payloads = list(Path('/kaggle/input').rglob('payload-734efa9'))
            if len(registries) != 1 or len(payloads) != 1:
                raise RuntimeError('Attach one exact EditCLEVR payload containing splits.json')
            target = SOURCE / 'datasets/external/editclevr'
            target.mkdir(parents=True, exist_ok=False)
            os.symlink(payloads[0], target / 'payload-734efa9', target_is_directory=True)
            os.symlink(registries[0], target / 'splits.json')
            command('-m', 'pytest', '-q', 'tests/test_editclevr_model.py',
                    'tests/test_editclevr_dataset.py', '--override-ini=addopts=')
            result = SOURCE / 'artifacts/editclevr-v1'
            command('-m', 'scripts.run_editclevr_v1', '--output', result, 'run')
            command('-m', 'scripts.verify_editclevr_v1', SOURCE)
            """,
        )
    add(
        "code",
        f"""
        evidence = BASE / '{campaign}-evidence.tar.gz'
        archive = verifier.archive_tree(result, evidence)
        verifier.verify_archive(evidence, expected_sha256=archive['archive_sha256'])
        print(json.dumps({{'evidence_path': str(evidence),
                          'evidence_sha256': archive['archive_sha256'],
                          'source_sha256': EXPECTED_SHA256}}, indent=2))
        """,
    )
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {
                "isGpuEnabled": campaign == "editclevr",
                "isInternetEnabled": False,
            },
        },
        "cells": cells,
    }
    output = output_dir / f"olympus_{campaign}_runner.ipynb"
    output.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--campaign", choices=CAMPAIGNS, required=True)
    parser.add_argument("--external-source", action="store_true")
    arguments = parser.parse_args()
    print(
        build(
            arguments.output_dir,
            campaign=arguments.campaign,
            embed_source=not arguments.external_source,
        )
    )
