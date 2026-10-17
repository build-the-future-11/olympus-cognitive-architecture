# Stable low-temperature reference-model sampling

## Scope

This numerical repair concerns the existing character-bigram Foundry reference
model. It does not implement a new Olympus/Ghost model, run a scientific study, establish
throughput, or bypass model promotion. Base: main at
`88cc23517e758e4fcd48d5bf08e269973a36f125`.

## Reproduced failure

For the artificial text `ab` repeated 20 times, `generate` at temperature `1e-5`
raised `ValueError: Total of weights must be greater than zero`. Directly raising
every probability to `1 / temperature` underflows all weights. The same failure
affects an exactly uniform smoothed row and the smallest positive float.

## Repair and verification

Compute each positive weight as `exp(log(p / max(p)) / temperature)` and retain
zero probabilities as zero weights. Common positive scaling preserves the
categorical distribution, and at least one weight remains exactly one. Dividing
before taking logarithms also preserves distinct near-maximum probabilities that
could otherwise collapse to the same rounded logarithm. The zero-temperature
greedy path is unchanged. Tied probabilities retain their seeded random selection.

Eighteen constructed numerical tests cover small/subnormal positive temperatures,
uniform ties, ordinary-temperature seeded compatibility and zero-temperature
generation. The initial thirteen cases had six failures and seven passes before
repair. Independent review added three valid zero-probability cases from subnormal
smoothing and two cases with distinct near-maximum probabilities in an accepted
16-character checkpoint. The first group preserves zero weights without taking
log(0); the second prevents false ties from rounding logarithms. All eighteen pass.
These ordinary-temperature examples establish their seeded compatibility;
floating-point rescaling is not a universal bitwise-output guarantee for all seeds.

Local numerical verification uses the actual bigram and schema modules with only
the eager application/package initialization isolated; no model/math code is
replaced. The local interpreter is Python 3.12, while the declared full release
environment is Python 3.14. Full API, training, web and release integration remain
subject to the existing hosted release gate. No real research outcome was run.
