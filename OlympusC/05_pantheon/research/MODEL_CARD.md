# System / Model Card

Pantheon is not a learned predictive model in this version. Its core is a deterministic rule adjudicator operating over frozen package hashes, run provenance, canary checks, and numerical result vectors.

## Inputs
- frozen package manifest;
- proposer run directory;
- replicator run directory;
- metrics JSON;
- observed and canonical hashes;
- provenance metadata.

## Outputs
A disagreement label, evidence-backed reasons, and numerical disagreement statistics (`D_z`, relative error, sign agreement, CI overlap).

## Architecture
1. **Package freezer** creates claim/protocol/data/environment artifacts and canonical hashes.
2. **Runner** copies the package into a fresh temporary workspace and executes analysis in a separate process.
3. **Provenance recorder** freezes hashes, runtime, seed, process status, and independence metadata.
4. **Comparator** computes numerical disagreement.
5. **Canonical adjudicator** checks hidden-state canaries and canonical drift before interpreting numerical disagreement.

## Parameter count
Not applicable: no learned Pantheon parameters.

## Intended use
Research infrastructure for diagnosing reproducibility failures in controlled or externally supplied research packages.

## Limitations
The implemented adjudicator is rule-based and the benchmark taxonomy is currently small. It has not been evaluated with multiple LLM research agents, adversarial packages, container-level hardware isolation, or a large corpus of real research repositories.
