# External Validation Protocol

## Objective
Test the claim that pairwise agreement between independent research agents can contain false consensus, and evaluate Pantheon against canonical published-task answers without exposing those answers to the agent workspaces.

## Unit of analysis
A published computational-reproducibility task. For CORE-Bench, each unit is a Code Ocean capsule derived from a scientific paper.

## Independence requirements
1. Different provider/model identity for Agent A and Agent B.
2. Separate extracted workspaces.
3. No proposer transcript or answers passed to the replicator.
4. Canonical answers stored outside both workspaces.
5. Every command/result recorded in the agent transcript.
6. Model/provider identity recorded per run.

## Primary measurements
- canonical task accuracy for each agent;
- pairwise answer agreement;
- false-consensus count/rate: A and B agree but the agreed answer is outside the canonical accepted set;
- task completion/report rate.

## Canonical stochasticity
CORE-Bench metadata may contain several successful reproduction result sets. Pantheon converts each question to the set of distinct accepted values and scores a prediction correct if it matches any accepted value within the configured numeric tolerance. This avoids selecting a convenient single canonical stochastic realization after seeing model output.

## Prespecified publication gate
A full readiness PASS requires:
- local evidence validator PASS;
- complete test suite PASS;
- >=15 OOD/held-out external tasks;
- >=20 scored questions;
- distinct agent identities on every task;
- >=2 scientific fields;
- complete external summary metrics;
- a valid `report.json` from both agents on every counted task and no provider error;
- at least one `heldout` or `ood` benchmark visibility class.

Public CORE-Bench training tasks are explicitly treated as pilot evidence because their answers are publicly accessible and may be present in model training data or retrieved by an agent.

## Integrity rules
No run is removed because it is wrong. Interrupted runs may be resumed, but existing completed artifacts are preserved. A zero false-consensus result is a valid negative result. A high false-consensus result is not interpreted causally without inspecting trajectories and provenance.
