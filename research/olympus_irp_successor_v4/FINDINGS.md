# Olympus v4: completed oracle-routing diagnostic and matched ablations

Executed 2026-10-17 in private Kaggle CPU notebook `ryangomez010/olympus-v4`, version **359270139**. Runtime **1080.4 seconds**. The complete recovery archive was downloaded and SHA-256 verified. All **1,102 inventoried prior files remain byte-identical**, including prior negative results and preexisting working-tree edits.

**Verdict:** training corruption coverage is the largest demonstrated repair. Dynamic semantic routing adds value. The proposed utility router fails its primary comparison, and removing the counterfactual penalty does not meaningfully improve accuracy. This is a completed synthetic mechanism diagnostic, not a SOTA result.

## What was implemented

- A separate v4 generator, expert-bank model, runner, tests, protocol, and evidence directory. Prior studies were not rewritten.
- Independent domain IDs and target-independent action masks remove the known metadata answer shortcuts. The constant all-legal mask deliberately makes no policy-enforcement efficacy claim.
- Clean nonlinear latent futures replace the old target that could be recovered with `observed_state + 0.1 * padded_action`.
- Five independently supervised expert heads and probability fusion expose whether useful expert predictions exist before judging the router.
- Missing-only versus mixed missing/perturbed training is paired on the same latent examples and training budget.
- Two-fold cross-fitting trains routers on predictions for examples excluded from their expert bank's fit. The final model averages the two frozen half-trained banks; there is no full-data refit.
- Uniform, learned fixed, semantic, utility, corruption-proxy, best-single, and semantic-training-mean controls share those banks. Privileged corruption and task oracles are diagnostic-only.
- Development selection and source/checkpoint hashes are frozen before protected clean, corrupted, and OOD evaluation. No tuning followed outcome inspection.

## Measured improvements and failures

Accuracy below is a percentage. Differences are percentage points (pp), with descriptive 95% intervals from five dataset clusters, averaging two paired training seeds per cluster.

| Matched corrupted-input comparison | Difference (pp) | Descriptive 95% interval | Interpretation |
|---|---:|---:|---|
| Mixed vs missing-only training, semantic router | **+27.42** | [+25.30, +29.28] | Training coverage repair works: **68.32% → 95.74%** |
| Semantic vs learned fixed routing | **+10.21** | [+9.62, +10.76] | Input-dependent routing adds value |
| Semantic vs its own mean training gates | **+11.54** | [+10.27, +13.18] | Gain is not explained by a constant preferred mixture |
| Utility vs development-selected strongest control (semantic) | **−6.82** | [−7.71, −5.89] | **Primary hypothesis not supported**; all five cluster deltas are negative |
| Utility vs corruption proxy | +1.32 | [+0.63, +1.89] | Predicting task correctness helps relative to the proxy, but is insufficient |
| Corruption oracle vs uniform | +14.82 | [+13.53, +16.64] | Privileged quality information can improve routing |
| Task oracle vs uniform | +18.95 | [+17.21, +20.79] | Useful expert signal exists; averaging loses substantial performance |
| Removing CF penalty, semantic router | +0.01 | [−0.02, +0.04] | No practically useful benefit demonstrated |

The preregistered primary required utility routing to exceed the development-selected control by at least 2 pp with a positive interval lower bound. It failed. Utility routing must not be promoted as the overall winner.

| Route (mixed training, CF=0) | Clean | Corrupted | OOD |
|---|---:|---:|---:|
| Uniform | 100.00% | 80.64% | 97.88% |
| Learned fixed | 100.00% | 85.54% | 98.64% |
| Semantic | 100.00% | **95.74%** | 91.99% |
| Utility | 100.00% | 88.93% | 99.55% |
| Corruption proxy | 100.00% | 87.61% | **99.75%** |
| Corruption oracle | 100.00% | 95.47% | 100.00% |
| Task oracle | 100.00% | 99.59% | 100.00% |
| Best single, development-selected | 100.00% | 74.08% | 75.21% |
| Semantic training-mean gates | 100.00% | 84.20% | 79.32% |

Semantic routing leads the tested deployable controls on the corrupted regime but has a substantial OOD regression relative to utility, corruption-proxy, and fixed routing. The OOD interventions were unseen token reversal, image inversion, and state bias. This does not justify selecting a new hybrid using these protected results and calling it confirmed.

The task oracle reaches 99.59%, leaving an observed 3.85 pp gap above semantic routing on corrupted inputs. It selects experts using true-class likelihood per bank and is neither deployable nor a mathematical accuracy upper bound. The corruption oracle is a quality heuristic and is slightly worse than semantic routing; metadata quality alone is not a sufficient routing objective.

## Temporal and shortcut diagnostics

| Mixed/CF=0 regime | Learned future MSE | Persistence MSE | Old formula MSE |
|---|---:|---:|---:|
| Clean | **0.06873** | 0.02977 | 0.02698 |
| Corrupted | **0.13339** | 0.48999 | 0.48695 |
| OOD | **0.11074** | 0.86200 | 0.85905 |

The repaired temporal target reveals useful recovery on corrupted and OOD states, but the learned predictor is worse than both simple baselines on clean states. The target leak is removed; temporal modeling is not universally solved. Domain-only and mask-only classification remain approximately 24–25%, near four-class chance, in the protected data.

## Exact conclusion and next research boundary

1. Retain mixed-corruption training: this is the strongest causal improvement supported by the matched v4 ablation. Preserve the fresh split and leakage repairs as evaluation requirements.
2. Retain semantic routing as the development-selected control for this corrupted regime. Do not replace it with the utility router on the strength of a proxy objective or OOD-only win.
3. Stop treating CF removal as the central repair: the matched effect is negligible here. This tests the analogous v4 probability-fusion penalty, not the original v3 latent-fusion penalty.
4. The remaining measurable problems are routing under distribution shift and clean temporal accuracy. A subsequent, newly frozen study could test stronger corruption-diverse development coverage and a residual temporal predictor against persistence. These are hypotheses, not fixes already validated by this run.
5. SOTA remains **NOT_EVALUATED**: all clean accuracies saturate at 100%, tasks are generated rather than natural data, models are tiny reference implementations, and no public benchmark or independent replication has been run. Cross-version v3/v4 score changes cannot isolate a single cause because the data, objectives, and fusion representation changed.

## Evidence and verification

- **40/40 cells**: five dataset seeds × two training seeds × two coverage conditions × two CF weights; **80 expert-bank fits**, **120/120 protected regime evaluations**, nine routing modes.
- **9/9 qualification tests passed**, no skips/errors/failures, 1.066 seconds on the executing Kaggle host. Local resource-guard rejection receipts are retained.
- Current local scientific sources equal executed frozen sources. All checkpoint bindings and 120 prediction archives pass hash verification. Cross-fit overlap is zero; control selection is development-only; every paired condition uses identical protected data.
- All reported aggregate means, primary cluster deltas, and all 57 comparison cluster deltas/means were independently recomputed from downloaded reports. This is provenance and arithmetic verification, not an independent retraining or an independent scientific replication.
- [Protocol](PROTOCOL.md), [machine-readable summary](SUMMARY.json), [exact generated result table](RESULTS.md), [completed execution receipt](KAGGLE_COMPLETED_VERIFIED.json), [downloaded evidence verification](DOWNLOADED_EVIDENCE_VERIFICATION.json), [test XML](KAGGLE_TESTS.xml).
- Full source, checkpoints, raw prediction tensors, per-cell reports, environment/guard receipts, and logs: `artifacts/oracle-fusion-v4-kaggle-359270139/recovered/` in the repository. The original compressed archive is retained alongside it.
- Recovery SHA-256: `ad2d03f6190d165dd68771224e5cc7f733a0b28919450d543fc6ecbc04ecf552`.
- [Private completed Kaggle notebook](https://www.kaggle.com/code/ryangomez010/olympus-v4).

No production promotion, publication, independent-reproduction claim, or SOTA claim is authorized by these results.
