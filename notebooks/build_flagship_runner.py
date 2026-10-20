"""Build the offline, hash-pinned six-family v2 Kaggle qualification notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from notebooks.build_kaggle_runner import build as build_base


def build(
    output_dir: Path, *, device: Literal["cpu", "cuda"] = "cpu", embed_source: bool = False
) -> Path:
    notebook = build_base(output_dir, device=device, embed_source=embed_source)
    document = json.loads(notebook.read_text())
    bundle_hash = json.loads((output_dir / "source-receipt.json").read_text())["archive_sha256"]
    document["cells"][1]["source"] = document["cells"][1]["source"].replace(
        'BASE = Path("/kaggle/working/olympus")',
        f'BASE = Path("/kaggle/working/olympus-flagship-{bundle_hash[:12]}-{device}")',
    )
    if device == "cuda":
        # Kaggle may provision a pair of GPUs. Admit only physical device zero
        # to the qualification subprocess and every regression child process.
        document["cells"][1]["source"] += (
            "\nENV['CUDA_VISIBLE_DEVICES'] = '0'\n"
            "print('CUDA qualification is restricted to physical GPU 0', flush=True)\n"
        )
    document["cells"][0]["source"] = (
        "# Olympus six flagship v2 engineering qualification\n\n"
        + (
            "The source archive is embedded and hash-verified. "
            if embed_source
            else "Attach olympus-source.tar.gz as a private input. The setup verifies its hash. "
        )
        + "This runs bounded artificial tests and eight tiny optimizer steps. It reports actual "
        "parameters and reload parity; the 12–20B Prometheus and 8T Kronos targets "
        "remain separate from this qualification. "
        "Keep the platform CUDA Torch. Network access stays disabled. Save outputs and stop "
        "the session after completion.\n"
    )
    source = (
        "qualification = RESULTS / 'flagship-v2'\n"
        "try:\n"
        "    command('-m', 'scripts.qualify_flagship_v2', '--device', DEVICE,\n"
        "            '--output', qualification)\n"
        "finally:\n"
        "    if qualification.exists():\n"
        "        log = qualification / 'regressions.log'\n"
        "        if log.exists():\n"
        "            print(log.read_text()[-20000:], flush=True)\n"
        "        receipt_file = qualification / 'receipt.json'\n"
        "        if receipt_file.exists():\n"
        "            receipt = json.loads(receipt_file.read_text())\n"
        "            print('QUALIFICATION_RECEIPT', json.dumps({k: v for k, v in receipt.items()\n"
        "                  if k != 'source'}, indent=2), flush=True)\n"
        "            print('SOURCE_SHA256', receipt['source']['sha256'], flush=True)\n"
        "        persist()\n"
    )
    compile(source, "<flagship-kaggle-cell>", "exec")
    document["cells"] = document["cells"][:2] + [
        {
            "cell_type": "code",
            "id": "flagship-qualify",
            "metadata": {},
            "source": source,
            "outputs": [],
            "execution_count": None,
        }
    ]
    document["metadata"]["kaggle"] = {"isGpuEnabled": device == "cuda", "isInternetEnabled": False}
    for cell in document["cells"]:
        if cell["cell_type"] == "code":
            compile(cell["source"], "<flagship-kaggle-cell>", "exec")
    notebook.write_text(json.dumps(document, indent=2) + "\n")
    return notebook


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--embed-source", action="store_true")
    args = parser.parse_args()
    print(build(args.output_dir, device=args.device, embed_source=args.embed_source))
