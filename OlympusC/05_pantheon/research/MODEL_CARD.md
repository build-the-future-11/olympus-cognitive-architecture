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

## Evaluated external agents

The definitive V4 OOD study ran `qwen3:0.6b` and `llama3.2:1b` through one local Ollama provider on 17 CORE-Bench v1.1 OOD tasks (20 questions). Both models answered 0/20 questions and authored 0/17 valid reports. This verifies that Pantheon can preserve and score failed trajectories; it does not validate the adjudicator on substantive outputs from capable agents.

## Limitations
The implemented adjudicator is rule-based and the controlled benchmark taxonomy is small. It has been exercised with two distinct small local LLM identities, but neither produced a substantive answer. It has not been validated with capable provider-diverse agents, adversarial packages, VM-level or separately administered hardware isolation, blinded expert adjudication, or a representative corpus of research repositories.
