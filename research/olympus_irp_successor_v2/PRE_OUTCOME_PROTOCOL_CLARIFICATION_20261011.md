# Pre-outcome protocol clarification — 2026-10-11

During development-only inspection, the generic Olympus risk-coverage diagnostic was found to calculate coverage only among examples for which Aion had not forced STOP. The registered phrase “selective accuracy at 50% coverage” was therefore ambiguous if the active subset were smaller than the full evaluation split.

Before scientific freeze and before opening any protected clean, corrupted, or OOD metric, the endpoint was made exact: rank all protected examples by effective risk, retain the lowest-risk `ceil(0.5 N)` examples, and calculate accuracy from the raw decision logits on those retained examples. Forced-stop examples remain in the ranking population.

This clarification changes no generator, model, training hyperparameter, split, seed, direction, practical threshold, or protected outcome. It prevents availability-conditioned denominators from making the primary endpoint incomparable across variants.
