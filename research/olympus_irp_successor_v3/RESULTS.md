# Olympus v3 reliability-fusion results

Frozen disposition: **NEGATIVE**.

Primary full-minus-semantic corrupted selective-accuracy delta: -0.0125 (95% bootstrap CI -0.0281 to 0.0031; required effect 0.0200).

## Confirmatory means across five seeds

| Variant | Regime | Accuracy | Selective@50 | Reliability MAE | Reliability separation |
|---|---|---:|---:|---:|---:|
| full | clean_test | 1.0000 | 1.0000 | 0.2868 | n/a |
| full | corrupted_test | 0.8727 | 0.8969 | 0.4453 | 0.0600 |
| full | ood_test | 0.7852 | 0.8016 | 0.4374 | 0.0789 |
| no_reliability_supervision | clean_test | 1.0000 | 1.0000 | 0.5275 | n/a |
| no_reliability_supervision | corrupted_test | 0.8773 | 0.9000 | 0.4073 | -0.0066 |
| no_reliability_supervision | ood_test | 0.7969 | 0.8219 | 0.4166 | -0.0051 |
| semantic | clean_test | 1.0000 | 1.0000 | 0.4897 | n/a |
| semantic | corrupted_test | 0.8836 | 0.9094 | 0.4146 | -0.0005 |
| semantic | ood_test | 0.8047 | 0.8219 | 0.4240 | 0.0006 |
| uniform | clean_test | 1.0000 | 1.0000 | 0.4955 | n/a |
| uniform | corrupted_test | 0.8766 | 0.9109 | 0.4124 | -0.0001 |
| uniform | ood_test | 0.8047 | 0.8328 | 0.4224 | 0.0001 |
| reliability_only | clean_test | 1.0000 | 1.0000 | 0.3571 | n/a |
| reliability_only | corrupted_test | 0.8742 | 0.9016 | 0.4396 | 0.0308 |
| reliability_only | ood_test | 0.7945 | 0.8234 | 0.4424 | 0.0336 |

## Integrity audit

- Protected receipts: 75 of 75.
- Distinct checkpoint bindings: 25 of 25.
- Parameter count: [27962] (matched=True).
- Risk-monotonicity violations: 0.
- Transition-legality violations: 0.
- Secondary results are descriptive and do not alter the negative disposition.
- Scope is controlled synthetic evidence only.
