"""Build a self-contained, source-pinned CPU Kaggle handoff for routing v6.

The default runs artificial qualification only. A second scientific campaign
requires a separately admitted reproduction budget; it cannot replace the
retained first campaign. No network, credentials or external weights are used.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

from scripts.run_conservative_routing_v6 import ROOT, sources


def build(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    paths = sorted(set(sources()) | {"scripts/__init__.py", "tests/__init__.py", "LICENSE"})
    payload = {name: base64.b64encode((ROOT / name).read_bytes()).decode() for name in paths}
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    setup = f'''from pathlib import Path
import base64, hashlib, json, os, subprocess, sys, tarfile, time

PAYLOAD = {payload!r}
HASHES = {hashes!r}
EXPECTED = {digest!r}
assert hashlib.sha256(json.dumps(HASHES, sort_keys=True).encode()).hexdigest() == EXPECTED
BASE = Path('/kaggle/working') / ('olympus-v6-' + EXPECTED[:12])
SOURCE, RESULTS = BASE / 'source', BASE / 'results'
SOURCE.mkdir(parents=True, exist_ok=True)
for name, encoded in PAYLOAD.items():
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('unsafe source member')
    value = base64.b64decode(encoded)
    assert hashlib.sha256(value).hexdigest() == HASHES[name]
    destination = SOURCE / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert destination.read_bytes() == value, 'Existing source differs'
    else:
        destination.write_bytes(value)
RESULTS.mkdir(parents=True, exist_ok=True)
ENV = dict(os.environ, PYTHONPATH=str(SOURCE), PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
def command(action, destination, *extra):
    subprocess.run([sys.executable, '-m', 'scripts.run_conservative_routing_v6',
                    action, '--output', str(destination), *map(str, extra)],
                   cwd=SOURCE, env=ENV, check=True)
def preserve():
    archive = BASE / ('retained-' + str(time.time_ns()) + '.tar.gz')
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(RESULTS, arcname='results')
    print('RETAINED', archive, hashlib.sha256(archive.read_bytes()).hexdigest())
print('SOURCE VERIFIED', EXPECTED)
'''
    run = '''# Keep False: the first campaign has already been admitted on GitHub CPU.
# Enable only for a separately authorized independent reproduction, never a
# favorable replacement result. Missing dependencies fail without downloads.
RUN_INDEPENDENT_REPRODUCTION = False
qualification = RESULTS / 'qualification'
try:
    command('qualify', qualification)
    if RUN_INDEPENDENT_REPRODUCTION:
        command('run', RESULTS / 'independent-reproduction', '--qualification', qualification)
finally:
    preserve()
'''
    for value in (setup, run):
        compile(value, "<routing-v6-kaggle>", "exec")
    document = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"isGpuEnabled": False, "isInternetEnabled": False},
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": "scope",
                "metadata": {},
                "source": "# Olympus routing v6: retained-source handoff\n\n"
                "CPU and internet disabled. The complete source is embedded and checked. "
                "Requires Python >=3.11, Torch >=2.4, NumPy, Pydantic >=2.10 and pytest. "
                "The default performs artificial qualification only. Scientific reproduction "
                "requires separate admission. Save the retained archive before ending the session. "
                "A generated notebook is not Kaggle execution evidence.\n",
            },
            *[
                {
                    "cell_type": "code",
                    "id": name,
                    "metadata": {},
                    "source": value,
                    "outputs": [],
                    "execution_count": None,
                }
                for name, value in (("restore", setup), ("qualify", run))
            ],
        ],
    }
    notebook = output / "olympus-routing-v6.ipynb"
    notebook.write_text(json.dumps(document, indent=2) + "\n")
    (output / "source-receipt.json").write_text(
        json.dumps(
            {
                "source_manifest_sha256": digest,
                "files": hashes,
                "notebook_sha256": hashlib.sha256(notebook.read_bytes()).hexdigest(),
                "execution": "NOT_EXECUTED_ON_KAGGLE",
            },
            indent=2,
        )
        + "\n"
    )
    return notebook


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(build(parser.parse_args().output))
