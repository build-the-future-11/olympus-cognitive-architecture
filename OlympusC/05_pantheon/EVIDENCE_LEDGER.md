# Evidence Ledger

| Claim | Result | Experiment ID / scope | Evidence file | Status |
|---|---|---|---|---|
| Clean R0 packages reproduce exactly | 20/20 controlled seeds | seeded-fault / clean | `results/main_results.csv`; `runs/seeded_faults/` | VERIFIED |
| Full Pantheon localizes seeded faults | 160/160 = 1.000 | 20 seeds × 8 conditions | `results/seeded_faults_predictions.csv` | VERIFIED |
| Numeric-only localization is weaker | 20/160 = 0.125 | same frozen comparisons | `results/baselines.csv` | VERIFIED |
| Pairwise artifact localization is weaker | 100/160 = 0.625 | same frozen comparisons | `results/baselines.csv` | VERIFIED |
| Canonical auditing detects correlated shared bug | 20/20; both weaker baselines call reproduced | shared_bug | `results/seeded_faults_predictions.csv` | VERIFIED |
| Canary detects seeded hidden-state exposure | 20/20 | hidden_contamination | `results/seeded_faults_predictions.csv` | VERIFIED |
| Two public-data implementations agree on improvement direction | 10/10 | WDBC, 10 splits | `results/public_case_paired.csv` | VERIFIED |
| Mean absolute public-case effect difference | 0.006993 | WDBC, 10 splits | `results/public_case_summary.json` | VERIFIED |
| Pantheon reduces false consensus among real LLM agents | Not tested | external agents absent | none | UNVERIFIED |
| A published research package is independently replicated | Not tested | no package supplied | none | UNVERIFIED |
| Controlled localization generalizes to real failures | Not established | benchmark engineered | `research/FALSIFICATION_REPORT.md` | UNVERIFIED |
