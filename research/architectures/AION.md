# Aion Architecture

## Purpose and key correction

Aion is a governed research-loop **system**, not necessarily another language
model. It coordinates Hermes, Prometheus, Perseus, Olympus-Atlas, Kronos, and
Pantheon while keeping authorization and scientific promotion outside learned
components. A learned router is added only if it beats deterministic routing.

## Target protocol state machine

```text
QUESTION -> EVIDENCE_MAP -> HYPOTHESES -> PREREGISTERED_PROTOCOL
         -> AUTHORIZED_EXECUTION -> AUDIT -> {REVISE, STOP, HUMAN_REVIEW}
         -> PROMOTION_CANDIDATE -> EXTERNAL_REPLICATION
```

In the target durable system, every transition has a typed input, deterministic
admission predicate, budget, authenticated actor, and immutable receipt. Outcome
data cannot alter a frozen protocol without creating a new version visibly
marked post hoc.

## Target architecture

1. **Deterministic controller baseline:** explicit state machine, routing table,
   resource quotas, and admission checks.
2. **Proposal policy:** Prometheus suggests hypotheses and protocols with
   assumptions and falsification criteria.
3. **Evidence service:** Olympus-Atlas resolves claims to authorized sources;
   held-out answers and protected evaluation data are invisible to proposers.
4. **Execution service:** Perseus runs bounded, sandboxed actions after approval.
5. **Temporal service:** Kronos monitors delayed outcomes and drift without
   changing frozen decisions.
6. **Audit service:** Pantheon compares canonical packages, signatures,
   environments, and claims. A model cannot waive a failed audit.
7. **Learned router, optional:** predicts the next permitted service and budget;
   deterministic masks make illegal transitions impossible.
8. **Promotion board:** target rule-based gates plus mandatory human authorization for
   external writes, publication, high-cost runs, or evidence promotion.

## Data, evaluation, and falsification

Train an optional router only on versioned protocol traces with negative
results, interruptions, permission denials, and independent claim/evidence
reviews. Never train on sequestered final answers. Measure valid-experiment
rate, protocol deviation, false-promotion rate, claim precision, evidence
coverage, protected-data leakage, budget, intervention recovery, and external
reproduction.

Compare the learned system with the deterministic controller using identical
tools, data, and budgets. **Reject Aion's learned component if deterministic
control matches its valid-experiment rate with equal or lower false-promotion
rate.** Stop immediately on self-approval, protocol rewriting after outcome
inspection, hidden test access, unbounded recursion, or unauthorized effects.

Current state: **deterministic controller and optional-router reference
implemented; no qualifying Aion checkpoint or autonomous-research result**.
`olympus/models/aion.py` enforces scoped non-model approval, protocol/audit
binding, legal transitions, frozen-protocol evidence membership, and
step/tool-call budgets against controller-owned in-memory run/protocol heads.
Host-registered approvals bind actor, authority, scope, run, protocol, expected
head, and optional expiry; registered audits bind run, protocol, audited head,
and evidence-artifact hashes. Model authority is rejected for promotion even
when an approval object is supplied. These records are defensively copied but
remain process-local and unsigned; “trusted” means accepted by the host, not
cryptographically verified. Receipts are structurally validated and hash-linked
in memory; they are not durable, append-only storage, signed, or externally
anchored. The learned router can rank only controller-permitted transitions and
remains subject to the deterministic-baseline kill rule.
