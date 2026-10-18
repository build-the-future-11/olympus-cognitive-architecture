# Olympus v4 oracle-routing results

Primary: **NOT_SUPPORTED**.

All intervals are descriptive, clustered across five independently generated datasets; two training seeds per dataset.

## Mixed training, no counterfactual penalty

| Route | Clean accuracy | Corrupted accuracy | OOD accuracy |
|---|---:|---:|---:|
| uniform | 1.0000 | 0.8064 | 0.9788 |
| fixed | 1.0000 | 0.8554 | 0.9864 |
| semantic | 1.0000 | 0.9574 | 0.9199 |
| utility | 1.0000 | 0.8893 | 0.9955 |
| corruption | 1.0000 | 0.8761 | 0.9975 |
| oracle_corruption | 1.0000 | 0.9547 | 1.0000 |
| oracle_task | 1.0000 | 0.9959 | 1.0000 |
| best_single | 1.0000 | 0.7408 | 0.7521 |
| semantic_train_mean | 1.0000 | 0.8420 | 0.7932 |

## Paired corrupted accuracy differences

| Contrast | Delta | Descriptive 95% CI |
|---|---:|---|
| corrupted/utility-minus-selected_control | -0.0682 | [-0.0771, -0.0589] |
| corrupted/utility-minus-semantic | -0.0682 | [-0.0771, -0.0589] |
| corrupted/utility-minus-corruption | 0.0132 | [0.0063, 0.0189] |
| corrupted/oracle_corruption-minus-uniform | 0.1482 | [0.1353, 0.1664] |
| corrupted/oracle_task-minus-uniform | 0.1895 | [0.1721, 0.2079] |
| corrupted/semantic-minus-fixed | 0.1021 | [0.0962, 0.1076] |
| corrupted/semantic-minus-semantic_train_mean | 0.1154 | [0.1027, 0.1318] |
| corrupted/uniform/coverage | 0.0297 | [0.0144, 0.0423] |
| corrupted/uniform/counterfactual | 0.0008 | [0.0002, 0.0014] |
| corrupted/semantic/coverage | 0.2742 | [0.2530, 0.2928] |
| corrupted/semantic/counterfactual | 0.0001 | [-0.0002, 0.0004] |
| corrupted/utility/coverage | 0.0895 | [0.0760, 0.1015] |
| corrupted/utility/counterfactual | 0.0001 | [-0.0005, 0.0006] |
| corrupted/corruption/coverage | 0.1300 | [0.1167, 0.1417] |
| corrupted/corruption/counterfactual | -0.0001 | [-0.0006, 0.0005] |
| corrupted/oracle_corruption/coverage | 0.0501 | [0.0436, 0.0569] |
| corrupted/oracle_corruption/counterfactual | 0.0003 | [0.0000, 0.0007] |
| corrupted/oracle_task/coverage | 0.0185 | [0.0142, 0.0225] |
| corrupted/oracle_task/counterfactual | 0.0001 | [0.0000, 0.0003] |

Oracle routes are privileged diagnostics. Neither is deployable.
Matched routing shares expert banks and executes all learned heads. Training time is not equated between privileged oracles and learned heads.
The counterfactual ablation tests the analogous penalty in v4 probability fusion; it is not a causal reanalysis of v3 latent fusion.
No public benchmark, natural-data validation, independent replication, or SOTA result is established.
All old results remain separate; this successor changes data, objective and fusion representation.
