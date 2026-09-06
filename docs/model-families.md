# Executable Model-Family References

Olympus now contains executable reference implementations for the shared
substrate and all six named family roles. After the bounded local component
run described below, their canonical state is
`EXPERIMENTAL_SMOKE_NOT_PROMOTED`.

That state has a deliberately narrow meaning: the repository contains real
typed contracts, deterministic validation and gating paths, PyTorch modules,
losses, and tests that exercise gradient updates. It does **not** mean that a family has a
rights-cleared training corpus, a revision-pinned external base, a qualifying
checkpoint, a frozen benchmark result, or a promoted serving artifact.

## Shared substrate

Implementation: `olympus/models/substrate.py`.

- `ModelIdentity`, `EvidenceItem`, `Claim`, `PlanStep`, `ToolSpec`,
  `ToolInvocation`, `PermissionSet`, and `WorkspaceState` define the common
  identity, evidence, planning, tool, and authority contracts. Workspace
  validation rejects duplicate identifiers, missing references, cyclic plan
  dependencies, invalid finite-JSON payloads, and discontinuous event hashes.
  This is validation of a process-local snapshot, not a durable event store.
- `AuthorizedWorkspaceView` is the projection intended for learned code. The
  trusted runtime deeply revalidates the source workspace, removes inaccessible
  evidence and unavailable tools before model tokenization or scoring, and does
  not expose permissions, claims, plan state, event history, or a commitment over
  hidden evidence.
- `TextOutput`, `ClaimOutput`, `ActionOutput`, and `AbstainOutput` are
  discriminated, hash-bound output envelopes. `parse_output_envelope` and
  `validate_envelope_against_workspace` validate model/workspace identity,
  evidence ACLs, tool declarations, a deliberately restricted recursive schema
  subset, capabilities, and workspace-supplied action-hash approval bindings.
  This is not full JSON Schema, approval-issuer authentication, or permission to
  execute an action.
- `AdapterCausalDecoder` is a compact decoder-only reference with independently
  selectable bottleneck adapters. `causal_language_model_loss` provides a real
  next-token gradient path.

Boundary: this compact decoder is not Qwen3, a pretrained language model, or a
serving-quality shared backbone. Adapter state dictionaries are component
artifacts, not family checkpoints. Workspace hashes and event chains are
integrity checks over supplied in-memory objects; they are not durable,
append-only, signed, or externally anchored records.

Integration is partial, not one composed runtime. `HermesWorkspaceRuntime`
consumes a filtered shared workspace and returns a validated `TextOutput` or
`AbstainOutput`; `PrometheusSelector` resolves trusted transition receipts from
authorized workspace evidence. Olympus-Atlas, Perseus, Kronos, and Aion still
use family-local service contracts, learned scorers are not wired into every
deterministic service, and Aion does not invoke the other five services as an
end-to-end research loop. The registry proves that declared classes are
importable, not that this dependency graph is integrated.

## Family implementations

| Family | Executable surfaces | What is real | What remains unproven |
| --- | --- | --- | --- |
| Hermes | `HermesGroundedService`, `HermesWorkspaceRuntime`, `HermesGroundingModule` in `olympus/models/hermes.py` | Content-hash checking, authorized-view filtering before scoring, extractive span citations, abstention, shared-workspace-bound text/abstain outputs, memory proposals marked `requires_approval`, and trainable attribution/confidence/memory heads | A memory persistence/approval service, integration of the learned heads with the deterministic runtime, generative answer quality, held-out calibration, a licensed base adapter, and every family promotion gate |
| Olympus-Atlas | `OlympusAtlasService`, `OlympusAtlasRetriever` in `olympus/models/atlas.py` | Opaque keyed corpus-version identifiers inside one service process; ACL-before-ranking and ACL-filtered index inspection; BM25, deterministic hashed dense retrieval, reciprocal-rank fusion, late interaction; and a trainable bi-encoder/late-interaction scorer | Durable or cross-process indexing/replay, shared-workspace/output adaptation, private-corpus quality, latency/scale, learned scorer integration, and comparative retrieval results |
| Prometheus | `TransitionReceipt`, `TrainableBranchProposer`, `TrainableProcessVerifier`, `PrometheusSelector` in `olympus/models/prometheus.py` | Bounded typed branches, canonical transition receipts, host-supplied trusted receipt hashes bound to workspace/model/branch/claim, authorized evidence resolution, typed comparisons, proposer/verifier losses, diversity loss, and deterministic rejection/abstention | Independent receipt issuance or signature verification, integration of learned scorers with the selector, autonomous hypothesis generation, scientific validity, proposer/verifier independence, and advantage over equal-compute baselines |
| Perseus | `DeterministicActionValidator`, `ActionApproval`, `CapabilityKernel`, `TrainableActionPolicy`, `TrainableRecoveryPolicy`, `TransactionManager` in `olympus/models/perseus.py` | Versioned action rules, required preconditions, host-registered action-bound approvals, defensive copies, commit-time reauthorization, sanitized error receipts, differentiable action/recovery policies, and process-local idempotency-key replay around an injected executor | A sandbox implementation, durable/crash-safe transactions, verification of the executor's idempotency claim, rollback of committed or failed material effects, end-to-end recovery quality, and stateful benchmark performance |
| Kronos | `SelectiveStateEncoder`, `KronosTemporalPlanner`, `TemporalAdapterManager` in `olympus/models/kronos.py` | Timestamped event validation/tensorization, an explicit time-decayed state recurrence, probabilistic forecast and action-plan heads, joint loss, local locked/hash/config-bound checkpoint staging, restricted tensor-only loading, and host-allowlisted evidence identifiers bound to one candidate and metric tuple | Parsed or signed evaluation/rollback/deletion reports, actual rollback or deletion replay, causal or calibrated forecasting, useful planning, retention under real drift, and family-level promotion. `TemporalAdapterManager.promote()` is a local experimental checkpoint-acceptance gate and cannot confer Foundry/family promotion |
| Aion | `AionController`, `AionRouter` in `olympus/models/aion.py` | A protocol-bound state machine, controller-owned process-local run/protocol heads, host-registered approvals bound to actor/authority/scope/run/protocol/head, audits bound to run/protocol/head, frozen-protocol evidence IDs, injected trusted time, structurally validated hash-linked receipts, step/tool budgets, rejection of model-authority promotion, legal-transition masking, router loss, and an explicit learned-router kill rule | Durable or signed authority/audit storage, integration with the shared workspace and five services, autonomous research, scientific discovery, external replication, and evidence that learning adds value over the deterministic controller |

`Atlas` is already used by an earlier retrieval-augmented model. Public prose
therefore uses **Olympus-Atlas**; the Python module remains `atlas.py`.

## Verification

Run the focused reference-implementation suite from the repository root:

```bash
.venv/bin/python -m pytest -q \
  tests/test_model_registry.py \
  tests/test_model_substrate.py \
  tests/test_hermes_model.py \
  tests/test_atlas_model.py \
  tests/test_prometheus.py \
  tests/test_perseus.py \
  tests/test_kronos_aion_models.py \
  tests/test_model_smoke.py
```

These tests exercise validation failures, access and authority boundaries,
deterministic behavior, checkpoint staging controls, and actual optimizer
updates on small synthetic tensors. Passing them establishes executable code
and differentiability only. It is not a family training run or an empirical
capability result.

To generate a fresh ignored component-run manifest rather than relying on the
committed release-facing summary:

```bash
.venv/bin/python -m olympus.cli models smoke \
  --output-dir artifacts/family-model-smoke/verification \
  --seed 20260906 --steps 24
```

The bounded 2026-09-06 local run executed optimizer and deterministic checks for
the substrate and each of the six family components. Its release-facing record
is
[`evidence/model_family_smoke_20260906.json`](../evidence/model_family_smoke_20260906.json);
the full generated manifest and non-qualifying component checkpoints remain
under ignored `artifacts/family-model-smoke/`. All component checks passed, and
every emitted checkpoint is explicitly marked `qualifying_checkpoint=false`
and `promoted=false`. The data scope is
`deterministic_synthetic_contract_fixtures`; this is execution evidence, not a
family-level experiment.

Family promotion still requires the immutable dataset, checkpoint, evaluation,
safety, serving, licensing, and human-review gates described in
`research/architectures/` and the Foundry promotion constitution.
