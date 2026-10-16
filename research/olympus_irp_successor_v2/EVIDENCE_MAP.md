# Evidence map

| Claim | Experiment | Raw artifact | Analysis | Figure/table |
|---|---|---|---|---|
| Full does not improve clean accuracy over uniform | five paired seeds, clean protected split | `runs/confirmatory/{full,uniform}/*/clean_test.json` | `CONFIRMATORY_SUMMARY.json` | `confirmatory_metrics.csv`, `confirmatory_accuracy.png` |
| Full underperforms uniform on corrupted selective accuracy in the point estimate | five paired seeds, corrupted protected split | `runs/confirmatory/{full,uniform}/*/corrupted_test.json` | `CONFIRMATORY_SUMMARY.json` | `confirmatory_metrics.csv` |
| The registered conjunctive hypothesis is negative | both primary endpoints and invariant audit | all 75 confirmatory JSON records | `CONFIRMATORY_SUMMARY.json` | primary-result table in `RESULT_AUDIT.md` |
| Learned routing moderately reweights components rather than collapsing | full, corrupted protected split | `runs/confirmatory/full/*/corrupted_test.json` | routing aggregation in result audit | `corrupted_gate_weights.png` |
| Host-risk and transition invariants held | all variants, seeds, regimes | invariant counts in every confirmatory JSON | `CONFIRMATORY_SUMMARY.json` | result audit |
| Findings do not establish real-world or JEPA performance | registered scope boundary | `PROTOCOL.md`, dataset receipt, freeze receipt | `RESULT_AUDIT.md` | limitations text |

The run root is `artifacts/research/olympus-controlled-fusion-v2/runs`. Its locked summary digest is recorded in `paper/figures/render_receipt.json`. No row above licenses a natural-data, deployment, target-scale, or JEPA efficacy claim.
