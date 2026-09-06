# Perseus Architecture

## Purpose and falsifiable claim

Perseus is a typed tool-use policy for reliable execution, recovery, and
checkpoint/resume. It proposes actions; a deterministic capability kernel owns
permissions and side effects. Its claim is higher valid task completion at an
equal unsafe-action rate and tool budget.

## Target architecture

1. **State encoder:** serializes goal, observations, remaining budget, tool
   schemas, and transaction status into the shared workspace format.
2. **Action decoder:** a Perseus adapter emits only grammar-constrained `Action`
   envelopes. Incremental parsing masks tokens that cannot complete the schema.
3. **Capability kernel:** in the target system, resolves tool/version, validates
   types and preconditions, checks ACL and approval, generates an idempotency key,
   and hands execution to a deployed sandbox. The model cannot bypass this
   boundary.
4. **Observation normalizer:** records structured outputs, hashes large blobs,
   marks partial execution, and strips untrusted tool text from policy fields.
5. **Recovery critic:** classifies retryable, compensatable, fatal, or
   authority-required failures and proposes the next bounded action.
6. **Transaction manager:** the target durable service checkpoints before writes
   and requires explicit commit for material external effects; retries reuse
   idempotency keys and committed effects require compensating actions.

## Data and losses

Use schema-versioned successful and failed traces, minimal-tool demonstrations,
permission denials, stale observations, partial writes, rollback cases, and
prompt injection from tool output. Add tool selection, argument token/type,
precondition, outcome prediction, recovery, and unsafe-action losses.

## Evaluation and stop rule

Run stateful interactive benchmarks plus repository-specific resettable tasks.
Report final-state equality, schema validity, tool/argument accuracy, excess
calls, recovery success, idempotency, budget adherence, false refusal, unsafe
attempts, and actual side effects—not textual claims of success. Baselines are a
deterministic workflow, ReAct prompting, unconstrained function calling, and
Perseus without the recovery critic. Reject promotion on any unauthorized write
or if improvements vanish after resetting environments between episodes.

Target input: `AuthorizedWorkspaceView + ToolRegistry`. Target output:
`Action | Stop`; never `Answer`
as proof of execution. Current state: **reference implementation exists, no
qualifying checkpoint**. `olympus/models/perseus.py` implements versioned action
validation, definition-owned preconditions, host-registered action-bound
approvals, defensive snapshots, commit-time reauthorization, sanitized
exception-type receipts, trainable action/recovery policies, and process-local
idempotency-key replay around an explicitly injected executor. The executor
protocol advertises idempotency but is not a sandbox implementation or an
independent proof of that property. Transaction state is in memory; committed or
failed material effects cannot be rolled back by this manager and require an
unimplemented explicit compensation action. This does not establish safe
general tool use.
