# Quantized storage: CPU timing protocol

The foundry stores weight tensors as 4-bit or 8-bit integers, then reconstructs
float32 tensors when loading the model. This can reduce the stored artifact size.
It does not currently execute integer inference kernels.

The previous report timed one generation call for each model in a fixed order.
That could mix cold-start effects, different generated sequence lengths, and
ordering effects with the comparison. New schema-v2 reports instead time the same
fixed-context CPU forward workload, warm both models, and retain seven measured
rounds with alternating order. The models use inference mode, identical token
inputs, and the same thread settings. The prompt is truncated to the smaller
supported context, and the actual token IDs are bound by a digest.

Each report contains all samples, median and p90 latency, timing order, warmup
policy, context length, parameter dtypes, thread counts, and PyTorch version.
`source_latency_ms` and `quantized_latency_ms` now mean the respective median
fixed-context forward latency. Schema v1 used a single generation latency. Old
reports remain readable with `inference_execution=legacy_unspecified` and no
invented timing metadata; consumers must check the schema before comparison.

`startup_latency_ms` remains one artifact load including dequantization.
`peak_rss_bytes` remains the cumulative process peak and is not an isolated
per-model memory comparison. No latency ratio is a validated systems speedup.
The quality gate still measures the existing loss/context criterion and does not
promote a model or authorize a paper's broad quality claim.

## Reproduce the bounded probe

Use the project's Python 3.14 environment and installed dependencies. From the
repository root:

```bash
PYTHONPATH=. python scripts/quantization_timing_probe.py /tmp/olympus-timing-new
python -m pytest tests/test_foundry_latency.py tests/test_deep_foundry.py -q
```

The probe trains the small existing 36-record engineering fixture for one epoch,
stores both quantized variants, reloads them, and evaluates the new timing
protocol on CPU with one intra-op and one inter-op thread. It saves a checkpoint,
both quantized artifacts, raw reports, data manifests, and source/output hashes.
The output path must be new. This fixture is synthetic infrastructure evidence;
it does not modify or rerun any frozen research experiment.

The retained [October 7 probe](evidence/quantization-timing-2026-10-07/probe.json)
records the actual runtime, sample arrays, artifact sizes, losses, and execution
semantics. Rerunning can change timings and serialized checkpoint bytes across
platforms; the retained per-run hashes identify exactly what was measured.

MLSys, ACL, or another research route still requires real representative
workloads, meaningful task-quality baselines, isolated memory and cost
measurements, and a substantiated contribution. This repair supplies a reliable
measurement component and a runnable engineering example.
