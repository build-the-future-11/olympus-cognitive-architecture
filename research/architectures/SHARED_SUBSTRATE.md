# Shared Olympus Substrate

## Design decision

Use one revision-pinned decoder-only Transformer and specialize it with
independently versioned adapters and heads. The first experiment remains the
already selected `Qwen/Qwen3-0.6B-Base`; architecture compatibility is more
important than inheriting Qwen3's reported capability. No family name attaches
to a checkpoint until a family-specific promotion suite passes.

This design avoids six costly pretraining programs, makes matched-budget
ablations possible, and lets serving reuse one quantized backbone. A larger base
is justified only after the 0.6B experiment establishes that data, interfaces,
and evaluations work.

## Common learned interface

The backbone consumes a serialized `WorkspaceState` and emits exactly one of
four grammar-constrained envelopes:

```text
Answer   {claims[], citations[], confidence, abstention_reason?}
Proposal {hypotheses[], tests[], assumptions[], confidence}
Action   {tool_id, schema_version, arguments, preconditions[], idempotency_key}
Stop     {reason, unresolved[], required_authority?}
```

The shared token sequence is:

```text
<objective> ... <policy> ... <evidence id=...> ...
<state> ... <budget> ... <request> ... <mode=answer|proposal|action|stop>
```

The base has RoPE and grouped-query attention because those are native to the
selected candidate, not because Olympus independently validated those choices.
Adapters target attention and feed-forward projections. A separate small
bidirectional encoder is permitted for retrieval; family decoders receive
evidence as prefix tokens so the first implementation does not require invasive
cross-attention changes.

## Typed state and evidence contracts

`WorkspaceState` is an event-sourced object with:

- objective, constraints, resource budget, and monotonic step counter;
- evidence references containing content hash, source URI, acquisition time,
  license, ACL label, and quoted span offsets;
- claims with support/refute edges and an explicit `unknown` state;
- a plan DAG with preconditions, postconditions, and retry policy;
- tool schemas, permission scopes, and outstanding approvals;
- immutable observation history and model/checkpoint identity.

Models never write directly to storage or external systems. They emit an
envelope; deterministic validators parse it, apply access controls, execute an
authorized transition, and append the result. Confidence is metadata, never an
authorization signal.

## Shared training objectives

For family (f), optimize a weighted objective whose components must each have
a reported ablation:

```text
L_f = L_next-token + λ_struct L_schema + λ_attr L_attribution
      + λ_cal L_calibration + λ_safe L_policy + family-specific terms.
```

Loss weights are frozen before held-out evaluation. Suggested stages are:

1. supervised envelope and abstention examples;
2. family adapter training on license-cleared, deduplicated records;
3. verifier/retriever heads where labels have objective construction rules;
4. preference optimization only after the existing 5,000-pair admission gate;
5. quantization and fresh-process serving validation.

Hidden rationales are not stored as scientific evidence. Training targets use
short, inspectable derivations, tool receipts, claim/evidence edges, or checkable
intermediate states.

## Promotion and evaluation

Every result reports three levels:

- **component:** retrieval, schema validity, calibration, or verifier quality;
- **end-to-end:** task completion under a fixed tool and token budget;
- **system safety:** access leakage, prompt injection, unsafe side effects,
  rollback, and false promotion.

Required comparisons are the unadapted base, deterministic non-LLM baseline,
single shared adapter, and the proposed specialization. Use immutable train,
validation, public-test, and sequestered-test manifests; at least three seeds;
bootstrap confidence intervals; per-example outputs; latency, memory, tokens,
tool calls, and failure taxonomies. Development stops if the specialized system
does not beat the cheapest valid baseline at matched budget.

## Dependency order

```text
Foundry -> Hermes -> {Prometheus, Perseus}
                    -> Olympus-Atlas -> Kronos -> Aion
Pantheon audit ----------------------------------^
```

Aion does not wait on every capability to begin as a deterministic controller,
but learned autonomous behavior cannot be promoted before upstream interfaces
and rollback controls are real.
