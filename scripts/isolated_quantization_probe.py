"""Measure retained engineering artifacts in separate Linux CPU processes.

This reuses the existing synthetic checkpoint. It never trains, promotes a model,
or executes a protected scientific campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

VARIANTS = {
    "float32": "int4/float32-model.pt",
    "int4": "int4/model-int4.pt",
    "int8": "int8/model-int8.pt",
}
PROMPT = "Explain why a measured zero differs from an absent observation."


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(root: Path, variant: str) -> dict[str, Any]:
    import gc
    import resource

    import torch

    from olympus.foundry.quantization import load_quantized_model
    from olympus.foundry.sft import ByteTokenizer, TinyCausalLM, TinyModelConfig

    if platform.system() != "Linux":
        raise RuntimeError("This RSS protocol requires Linux /proc semantics")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(31)
    source = json.loads((root / "probe.json").read_text())
    config = TinyModelConfig.model_validate(source["configuration"]["model"])
    artifact = root / VARIANTS[variant]
    expected = source["artifact_sha256"][VARIANTS[variant]]
    if sha256(artifact) != expected:
        raise ValueError("Retained artifact SHA-256 does not match its probe receipt")

    def rss() -> int:
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")

    gc.collect()
    before_rss = rss()
    before_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    start = time.perf_counter_ns()
    if variant == "float32":
        state = torch.load(artifact, map_location="cpu", weights_only=True)
        model = TinyCausalLM(config)
        model.load_state_dict(state)
        del state
        model.eval()
    else:
        model = load_quantized_model(artifact)
    load_ms = (time.perf_counter_ns() - start) / 1_000_000
    gc.collect()
    after_load_rss = rss()
    ids = ByteTokenizer.encode(PROMPT, bos=True)[: config.max_sequence_tokens]
    inputs = torch.tensor([ids], dtype=torch.long)
    with torch.inference_mode():
        model(inputs)
        model(inputs)
        samples = []
        for _ in range(7):
            start = time.perf_counter_ns()
            logits = model(inputs)
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
    if not torch.isfinite(logits).all().item():
        raise RuntimeError("Forward output contains non-finite values")
    dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    result = {
        "variant": variant,
        "artifact_relative_path": VARIANTS[variant],
        "artifact_sha256": expected,
        "artifact_bytes": artifact.stat().st_size,
        "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "buffer_bytes": sum(b.numel() * b.element_size() for b in model.buffers()),
        "parameter_dtypes": dtypes,
        "rss_before_load_bytes": before_rss,
        "rss_after_load_bytes": after_load_rss,
        "rss_load_delta_bytes": after_load_rss - before_rss,
        "rss_after_forward_bytes": rss(),
        "process_peak_before_load_bytes": before_peak,
        "process_peak_after_forward_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "load_ms": load_ms,
        "forward_samples_ms": samples,
        "forward_median_ms": statistics.median(samples),
        "input_tokens": len(ids),
        "input_token_ids_sha256": hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "output_shape": list(logits.shape),
        "output_finite": True,
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "python": platform.python_version(),
        "pid": os.getpid(),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("retained_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--worker", choices=tuple(VARIANTS))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.retained_root.resolve(), args.worker)))
        return 0
    if args.output.exists():
        parser.error("output must be a new directory")
    root = args.retained_root.resolve()
    original = json.loads((root / "probe.json").read_text())
    for relative in VARIANTS.values():
        if sha256(root / relative) != original["artifact_sha256"][relative]:
            raise ValueError(f"Retained artifact digest mismatch: {relative}")
    args.output.mkdir(parents=True)
    protocol = {
        "schema": "olympus.isolated_quantization_probe.v1",
        "purpose": "Separate stored artifact size from loaded float32 model and process RSS",
        "evidence_type": "retained 36-record constructed engineering fixture",
        "retained_probe_sha256": sha256(root / "probe.json"),
        "script_sha256": sha256(Path(__file__)),
        "repetitions": 3,
        "variant_orders": [
            ["float32", "int4", "int8"],
            ["int4", "int8", "float32"],
            ["int8", "float32", "int4"],
        ],
        "process_isolation": "One fresh Python interpreter per variant and repetition",
        "timing_scope": "Load excludes Python/torch import; forward excludes two warmup calls",
        "rss_scope": "Linux resident pages including shared libraries; not proportional set size",
        "peak_scope": "Each child process high-water RSS, including interpreter and imports",
        "cache_policy": "OS file caches are not cleared; these are not cold-storage measurements",
        "inference_execution": "All three variants execute float32 CPU kernels",
        "scientific_outcome_campaign": False,
        "new_training": False,
        "model_promotion_authorized": False,
        "research_claims_authorized": False,
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rows = []
    for trial, order in enumerate(protocol["variant_orders"]):
        for variant in order:
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), str(root), str(args.output),
                 "--worker", variant],
                check=True, capture_output=True, text=True, timeout=60,
            )
            row = json.loads(completed.stdout)
            row["trial"] = trial
            rows.append(row)
            (args.output / f"trial-{trial}-{variant}.json").write_text(
                json.dumps(row, indent=2) + "\n"
            )
    summary = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        summary[variant] = {
            field: statistics.median(row[field] for row in selected)
            for field in ["artifact_bytes", "parameter_bytes", "buffer_bytes", "load_ms",
                          "rss_before_load_bytes", "rss_after_load_bytes", "rss_load_delta_bytes",
                          "rss_after_forward_bytes", "process_peak_after_forward_bytes",
                          "forward_median_ms"]
        }
    result = {"protocol": protocol, "platform": platform.platform(), "rows": rows,
              "medians_over_three_processes": summary,
              "source_files_sha256": {
                  path.as_posix(): sha256(path)
                  for path in [Path("olympus/foundry/sft.py"),
                               Path("olympus/foundry/quantization.py")]
              }}
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
