# Project Status

## Current objective

Develop a rigorous method for distinguishing pairwise replication agreement from canonical scientific correctness, then test it without concealing failures.

## Working

- Canonical claim packages, hashing, provenance registries, canary evidence, isolated execution, and rule-based adjudication.
- Controlled 20-seed × 8-condition benchmark: 160 seed-condition units, 200 executions, 480 method-level classifications.
- Public WDBC cross-implementation study: 10 fixed splits, 20 implementation runs.
- Frozen CORE-Bench OOD study: 17 tasks, 20 questions, 34 local-agent trajectories, four fields, Python and R.
- Safe TAR/ZIP extraction, macOS shell sandbox, network denial, credential removal, separate ephemeral workspaces, bounded decoding, resumable persistence, and exact model/source identities.
- Failure-preserving normalization and scoring in which null/null is abstention, not agreement.
- Six-page compiled negative-result preprint.

## Implemented this run

- Downloaded and hash-verified all 17 selected official CORE-Bench v1.1 OOD capsules.
- Added Ollama provider support, nested tool-argument normalization, decoding bounds, safe extraction, and regression tests.
- Froze protocol V4 before the definitive run (source identity `8b4a68f226165995574d04b149fd92340db792f80d76885a8f0c5d52026fe109`).
- Executed all 34 agent trajectories with no provider errors.
- Added question-level Wilson intervals, task-cluster bootstrap intervals, exact paired testing, task diagnostics, and trajectory accounting.
- Rewrote and compiled the paper from measured evidence.

## Tested

- 31 tests pass and 2 macOS-sandbox-dependent tests are skipped, including the
  post-study scientific-readiness separation.
- Controlled evidence validator passes: 480 classifications, 200 executions, 20 public-case runs.
- Frozen V4 machine records remain byte-pinned. Comparing the repaired current
  source to V4 intentionally exits with
  `V4_SOURCE_DRIFT_REQUIRES_VERSIONED_PROTOCOL`; a current-source V5 draft
  manifest is explicitly neither frozen nor executed.
- External matrix complete: 17/17 tasks, 20/20 questions, 34/34 normalized artifacts.
- PDF compiles to six pages and was rendered and visually inspected page by page.

## Metrics / experimental evidence

- Controlled localization: numeric-only 20/160; pairwise artifacts 100/160; full canonical auditing 160/160.
- Public case: direction agreement 10/10; mean absolute effect difference 0.006993; maximum 0.020979.
- OOD study: both models 0/20 correct and 0/20 non-null; 0/34 agent-authored valid reports; 0/20 substantive agreement; 0/20 false consensus; Wilson 95% upper bound 0.161 for each zero rate.
- Qwen: 41 turns, 12 tool calls, 666.3 s. Llama: 184 turns, 284 tool calls, 1112.6 s; seven tasks saturated 20 turns.

## Partially working

- The harness is publication-artifact complete, but the evaluated models are below the capability floor required to study naturally occurring false consensus.
- macOS sandboxing is scoped and tested but weaker than separately administered VMs or hosts.

## Broken

- A source-identical V4 rerun is no longer possible from the repaired current
  tree. Both execution wrappers fail before model execution rather than mixing
  current code with frozen V4 claims or V4 output namespaces.
- The frozen V4 `PUBLICATION-READY` gate is retained for byte-level reproducibility and certifies structural completeness only. The post-study `poststudy/SCIENTIFIC_READINESS.json` gate supersedes that label for interpretation and fails the agent capability floor.

## Blockers

- ICLR-level cross-agent claims require capable frontier agents or equivalent local models, separately hosted/provider-diverse execution, nontrivial completion, and human adjudication. Those resources are not available in the current local environment.

## Highest-value next actions

1. Freeze a new confirmatory model-only protocol using capable agents while retaining the V4 task set and isolation rules.
2. Prespecify a minimum non-null completion threshold before interpreting agreement or false consensus.
3. Add blinded expert adjudication of substantive disagreements and canonical-answer defects.
4. Replicate on a random or exhaustive held-out task sample under VM-level isolation.

## Reproduction commands

```bash
make verify
../../.venv/bin/python analysis/analyze_external_rigor.py
../../.venv/bin/python poststudy/scientific_readiness_gate.py
../../.venv/bin/python scripts/v5_source_manifest.py
```

`scripts/run_local_ood_study.sh` and `scripts/run_final_external.sh` are
deliberately bound to V4 and now stop on source drift. A new confirmation needs
a versioned output namespace and model identities bound before a separate V5
protocol is frozen; the V5 source manifest alone is not execution authority.
