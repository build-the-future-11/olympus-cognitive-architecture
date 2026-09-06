# Perseus Architecture

## Purpose and falsifiable claim

Perseus is a typed tool-use policy for reliable execution, recovery, and
checkpoint/resume. It proposes actions; a deterministic capability kernel owns
permissions and side effects. Its claim is higher valid task completion at an
equal unsafe-action rate and tool budget.

## Architecture

1. **State encoder:** serializes goal, observations, remaining budget, tool
   schemas, and transaction status into the shared workspace format.
2. **Action decoder:** a Perseus adapter emits only grammar-constrained `Action`
   envelopes. Incremental parsing masks tokens that cannot complete the schema.
3. **Capability kernel:** resolves tool/version, validates types and preconditions,
   checks ACL and user approval, generates an idempotency key, and executes in a
   sandbox. The model cannot bypass this kernel.
4. **Observation normalizer:** records structured outputs, hashes large blobs,
   marks partial execution, and strips untrusted tool text from policy fields.
5. **Recovery critic:** classifies retryable, compensatable, fatal, or
   authority-required failures and proposes the next bounded action.
6. **Transaction manager:** checkpoints before writes and requires explicit
   commit for material external effects; retries reuse idempotency keys.

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

Input: `WorkspaceState + ToolRegistry`. Output: `Action | Stop`; never `Answer`
as proof of execution. Current state: **specified, no qualifying checkpoint**.
