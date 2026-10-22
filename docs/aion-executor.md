# Aion host-governed execution

`olympus.evaluation.aion_executor` converts a requested Aion transition into a
host-enforced execution receipt. It is intentionally independent of Torch and
model loading: a model may request a stage, but cannot create a callback,
override transition legality, spend beyond the supplied budget, or bypass STOP.

The host supplies one immutable `AionExecutionPlan` for every decision. The
plan binds a policy identity, stage names, legal-transition mask, finite
per-stage costs, finite remaining budget, and the zero-cost STOP stage. The
supplied `HostActionRegistry` snapshots the only callbacks the executor can
invoke, so later mutation of the caller's mapping cannot change authority.

The executor stops without invoking a callback when the decision is forced to
STOP, explicitly selects STOP, selects an illegal stage, exceeds budget, or
names an unregistered action. It records a canonical trace with separate plan,
decision, and registered-action-set digests plus the post-action budget.
`verify_aion_trace` rederives the host enforcement from those inputs without
invoking either the original callback or a substitute callback. Result digests
bind JSON-serializable outputs, but they do not independently establish that an
external side effect occurred or was correct.

`execute_aion_sequence` carries the remaining budget across up to 256 requested
decisions and emits an immutable `AionExecutionLedger`. The ledger binds the
complete requested sequence, including requests ignored after a terminal STOP,
and records both requested and consumed counts. `verify_aion_ledger` checks the
continuous budget chain and proves that no trace follows STOP without invoking
any registered action.

Run the artificial CLI fixture:

```bash
.venv/bin/olympus bu1ld aion-executor-demo
.venv/bin/olympus bu1ld aion-executor-demo --selected-stage 1
```

The first command executes the host-registered `inspect_evidence` fixture. The
second requests an illegal `publish` stage and produces a terminal STOP trace.
Both are engineering fixtures, not an LLM, a tool invocation, a promotion, or
a safety evaluation.

The API route `POST /aion/dry-run` and the Forge web-console fixture use only
simulated callbacks. They exercise policy enforcement without exposing a real
filesystem, shell, network, account, or publication capability. A deployed
host must supply a separately reviewed registry and its own authorization,
idempotency, audit retention, and rollback controls before connecting any
external action.
