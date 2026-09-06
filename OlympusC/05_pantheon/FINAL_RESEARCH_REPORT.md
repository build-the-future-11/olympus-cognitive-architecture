# Final Research Report

## Executive Summary

The supplied Pantheon repository was a concept-only README. It is now an executable research prototype with frozen claim packages, separate-process replication, provenance capture, hash-based auditing, canary contamination checks, numerical disagreement metrics, a rule adjudicator, a controlled seeded-fault benchmark, a public-data cross-implementation case study, tests, plots, and a paper draft.

The strongest result is narrow but real: on 160 controlled fault comparisons, the full canonical audit localized all injected classes, compared with 62.5% for pairwise artifact comparison and 12.5% for numeric-only comparison. The benchmark is intentionally diagnostic and therefore cannot be treated as an estimate of real-world accuracy.

## Research Question
Can canonical frozen-package auditing improve reproducibility-failure localization relative to pairwise or numerical agreement alone?

## What Was Implemented
- package schema and freezer;
- fresh temporary-workspace subprocess runner;
- canonical and observed hash capture;
- provenance files;
- independence vector;
- hidden-state canary check;
- disagreement metrics;
- numeric-only, pairwise-artifact, and canonical Pantheon adjudicators;
- seeded fault suite;
- public-data cross-implementation experiment;
- paper asset generator and evidence validator.

## Experiments Actually Run

### Controlled seeded faults
200 subprocess executions; 20 synthetic package seeds; 8 conditions; 160 comparisons; 480 adjudicator/baseline evaluations. Total measured subprocess runtime recorded in the registry: 0.633 seconds (the unrounded per-run values are in `results/execution_registry.csv`).

### Public-data case study
Ten prespecified stratified splits; proposer is scikit-learn L-BFGS logistic regression; replicator is an independent numpy batch-gradient-descent implementation. Twenty training/evaluation runs total.

## Main Results

| Method | Localization accuracy | Correct / total |
|---|---:|---:|
| Numeric only | 0.125 | 20 / 160 |
| Pairwise artifacts | 0.625 | 100 / 160 |
| Full Pantheon | 1.000 | 160 / 160 |

## Public Case

| Quantity | Proposer | Replicator |
|---|---:|---:|
| Mean accuracy | 0.9727 ± 0.0134 | 0.9783 ± 0.0101 |
| Mean ROC AUC | 0.9872 ± 0.0101 | 0.9948 ± 0.0050 |
| Mean accuracy improvement over majority | 0.3434 ± 0.0134 | 0.3490 ± 0.0101 |

Direction agreement is 1.000; mean absolute effect difference is 0.006993. No formal significance claim is made because the ten train/test splits overlap and are not independent studies.

## Ablations
Removing canonical reference information reduces localization from 1.000 to 0.625; removing artifact evidence entirely reduces it to 0.125.

## Robustness
The controlled benchmark is repeated over twenty independently generated synthetic data seeds; the public case uses ten prespecified split seeds and two different optimization implementations. No cross-hardware or cross-model robustness experiment was run.

## Failure Analysis
The central negative result is that the repository does not yet validate the original cross-LLM thesis. Process isolation is real; model/context independence between external agents is not.

## Statistical Evidence
Summary means and standard deviations are reported for the ten public-data splits. No p-value is reported because overlapping splits violate a simple independence interpretation. The controlled fault benchmark reports exact counts rather than inferential statistics.

## Strongest Supported Conclusion
Canonical frozen-package auditing adds information that proposer-vs-replicator agreement alone cannot provide, especially under correlated implementation drift and hidden-state contamination.

## Unsupported Original Claims
Any claim of general cross-agent reliability, reduced false consensus in real LLM systems, or end-to-end replication of a published paper remains unsupported.

## Remaining Limitations
- no external LLM research agents;
- no third-party frozen research package;
- engineered fault taxonomy;
- no adversarial package generation;
- no human adjudicator comparison;
- no container/hardware independence beyond separate fresh temp workspaces/processes.

## Reproduction Instructions

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
./scripts/reproduce_all.sh
```

## File / Evidence Map
See `EVIDENCE_LEDGER.md`, `RESEARCH_TRUTH.md`, `results/`, `runs/`, and `figures/`.

## Verdict
**SUBMISSION-CANDIDATE — strong protocol prototype and controlled evidence, but not yet defensible as a full cross-agent/NeurIPS-main-track validation without real multi-agent and third-party package experiments.**
