# Ablation Analysis

All ablations are evaluated on the same 40 frozen seeded-fault package comparisons (20 seeds × 8 conditions), so no result is inferred from an unexecuted configuration.

| Adjudication signals | Localization accuracy | n |
|---|---:|---:|
| Numeric only / no artifact hashes | 0.125 | 160 |
| Pairwise artifacts / no canonical manifest | 0.625 | 160 |
| Full Pantheon canonical audit | 1.000 | 160 |

The largest qualitative gap is the **shared bug** condition. Proposer and replicator execute the same corrupted implementation, so pairwise hashes and outputs agree. Both weaker baselines label all twenty shared-bug cases as reproduced. Pantheon compares both runs to the frozen canonical package and identifies implementation drift in all twenty.

The canary condition similarly demonstrates why pairwise artifact equality is insufficient for independence auditing: hidden-state contamination is visible in replicator output but not in ordinary file deltas.

These are controlled diagnostic ablations. They establish that each evidence channel is useful on the faults designed to exercise it; they do not estimate performance on naturally occurring research failures.
