# Result audit

## Disposition

- Central Olympus scientific hypothesis: **INCONCLUSIVE / NOT EVALUATED**.
- Bounded instruction-model engineering study: **MIXED DEVELOPMENT EVIDENCE**.
- Promotion: **NOT AUTHORIZED**.
- Confirmatory protocol: **NOT FROZEN** because no valid aligned corpus or untouched
  protected test exists.

## What ran

The corrected frozen matrix completed 24 of 24 CPU runs with no runtime failures. It
covered 900 total attempted epochs: five paired seeds each for cosine scheduling,
constant learning rate, and no validation-best restoration; three seeds each for packing,
medium scale, and small scale. Every checkpoint, log, exploratory evaluation, configuration
hash, and aggregate is retained under `artifacts/research/olympus-irp-v1-corrected`.

The first matrix attempt is separately retained. All 24 runs were rejected before
training because swap utilization exceeded the frozen resource ceiling, after which an
all-failure plotting defect made that attempt invalid. The defect was repaired without
editing the failed artifacts or weakening the guard.

## Observations and explanations

### Schedule

Cosine scheduling produced mean observed test loss 3.13889 versus 3.14858 for constant
learning rate. The paired difference (cosine minus constant) was -0.00969 with a
deterministic bootstrap 95% interval [-0.01607, -0.00564]. The difference is consistent
across the five frozen seeds but is practically small and is measured on an already
observed 12-record smoke test.

Observation -> cosine is slightly better. Hypothesized cause -> reducing the learning
rate late in training reduces overshoot. Proposed change -> keep cosine as the
development default. Predicted consequence -> lower held-out loss at equal epochs.
Discriminating experiment -> repeat on new source-independent data. Result -> not yet
available.

### Validation-best restoration

The mean observed test loss was 3.13889 with restoration versus 3.16673 at the final
epoch. The paired difference was -0.02784 with bootstrap interval
[-0.03644, -0.02087]. This supports restoration as an engineering safeguard in this
bounded study, but it does not establish external generalization.

### Scale

Mean observed test loss decreased from 4.61653 (35,552 parameters) to 3.44510 (145,664)
to 3.13889 (884,480). Format compliance was unstable for the smallest model (mean
0.3889) and 0.8333 for medium and large variants. The comparison is not compute matched,
so it does not isolate a scaling law.

### Packing

Packing-on and packing-off runs were bit-identical for the three shared seeds. Inspection
showed that individual serialized examples already approach the 192-token cap, so the
packing algorithm cannot combine them. The ablation is non-informative for this corpus;
it is not evidence that packing never matters.

### Capability ceiling

Exact match was 0% for every variant and every seed. This decisive negative result
prevents a capability or model-quality claim even though loss and format measures
improved. The corpus is also too small for scientifically meaningful generalization.

## What remains unresolved

The results say nothing about heterogeneous fusion, leave-one-component robustness,
risk calibration, OOD behavior, governed transitions in realistic settings, or JEPA
super-resolution quality. Those questions require the aligned corpus and protected
protocol described in `PROTOCOL.md`.
