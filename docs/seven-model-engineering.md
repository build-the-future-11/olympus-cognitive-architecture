# Seven-model engineering contract

Olympus maintains seven bounded reference implementations:

| ID | Implementation | Engineering role |
|---|---|---|
| `hermes` | `HermesReferenceModel` | causal language state with tool and abstention heads |
| `prometheus` | `PrometheusReferenceModel` | domain adapters and evidence classification |
| `perseus` | `PerseusReferenceModel` | sparse top-k expert dispatch and routing records |
| `atlas` | `AtlasReferenceModel` | text-image grounding and retrieval embeddings |
| `kronos` | `KronosReferenceModel` | temporal prediction, uncertainty, value, and planning |
| `aion` | `AionReferenceModel` | host-constrained research transitions and forced STOP |
| `jepa_upscaler` | `JEPAUpscaler` | masked high-resolution detail prediction and reconstruction |

Inspect the fast, non-executing inventory:

```bash
.venv/bin/olympus bu1ld seven-model-status
```

Run the complete bounded contract explicitly:

```bash
.venv/bin/olympus bu1ld seven-model-contract
```

The contract command builds all seven references in canonical order and performs a
deterministic artificial forward pass, finite-output check, and backward pass.
It reports exact reference and trainable parameter counts and fails unless every
trainable parameter tensor receives a finite gradient. It does not load a
scientific dataset, train a checkpoint, inspect protected evidence, or authorize
promotion.

## Admission boundaries

- Language attention masks and all authority or trajectory masks must be
  Boolean and reside on the same device as their inputs.
- Prometheus domain identifiers must be integral and in range.
- Atlas images must be finite normalized floating-point tensors with parameter-
  compatible dtype and device; timestamps must be finite floating-point values.
- Kronos states and actions must be finite floating-point tensors with common
  dtype and device. All exposed time-indexed outputs are exactly zero on padded
  steps.
- Aion state, transition authority, and risk tensors are typed separately. It
  never coerces a numeric tensor into an authority mask, and a forced STOP must
  already be host-authorized.
- JEPA images must be finite normalized tensors with matching dtype and device;
  context masks are Boolean. Checkpoint loading can bind both the checkpoint
  digest and the source dataset-manifest digest.

## Meaning of complete

`ENGINEERING_REFERENCE_COMPLETE_NOT_PROMOTED` means that the checked-in bounded
module has an executable, differentiable, fail-closed software contract. It does
not mean a target-scale model has been trained or that a capability, robustness,
safety, super-resolution, or deployment claim has been established.

The frozen acceptance criteria and budgets are in
`research/seven_model_engineering_v1/PROTOCOL.md`. The canonical receipt records
the exact final verification without changing any predecessor result.
