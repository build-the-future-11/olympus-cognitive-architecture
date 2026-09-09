# Next Experiments

The highest-value research experiment is a confirmatory Pantheon study with
agents that pass a prespecified, externally timestamped capability admission
floor. The next Foundry experiment is a
revision-pinned Qwen3-0.6B-Base QLoRA run, but it remains blocked on license
approval, data scale, provenance, and compute admission. Neither experiment may
be represented as completed by smoke-test evidence.

## Priority 1: Pantheon confirmation

- Freeze task selection, exclusions, adjudication, capability thresholds, and
  statistical analysis before running agents.
- Use independently implemented or provider-diverse agents and preserve every
  trajectory, execution package, environment description, and checksum.
- Require each agent condition to pass the capability floor before estimating
  false consensus or auditing advantage.
- Add blinded human review and multiple natural replication failures; the WDBC
  example alone is not sufficient external validation.
- Treat the existing CORE-Bench run as a capability-floor negative result, not
  evidence about capable autonomous researchers.

## Priority 2: Foundry external-base admission

- Approve and record base-model training and redistribution rights.
- Reach 10,000 reviewed training records and 1,200 untouched test records, with
  at least 100 test records in every capability category.
- Preserve source hashes, document-level split isolation, contamination scans,
  and the current protected test set.
- Admit the medium workload only with at least 8 GiB available memory and at
  most 50% swap use; otherwise move the run to an approved larger host.

## Frozen comparison

Use seeds 17, 31, and 47 with equal parameter-step budgets for the frozen base,
simple SFT/QLoRA, Olympus reference condition, and packing/mixing ablations.
Report every seed and failure. Do not interpret a mechanism when its ablation
does not change the implementation, as happened in the current packing smoke.

## Success and stop rules

Promotion remains fail-closed: exact task >=80%, structured format >=95%, every
workflow >=75%, quantized tool exact >=75%, no category regression over 5%,
context enforcement, fresh-process serving, and a hash-bound model card. Stop
on license ambiguity, split contamination, budget mismatch, resource refusal,
or two materially identical infrastructure failures.

Pantheon withholds consensus estimation and interpretation after inference if
the measured capability thresholds fail. Foundry stops before training if
provenance, license, split isolation, or resource admission fails. Negative
results are retained and reported without relabeling them as successful model
or agent evaluations.
