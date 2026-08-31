# Future Olympus Model-Family Roadmaps

These are planned programs, not instantiated models. Reserved names remain
unavailable until a real checkpoint passes its own family gates.

## Prometheus — reasoning and scientific synthesis

- Capability justification: multi-source scientific synthesis, explicit
  uncertainty, executable mathematics, and hypothesis comparison.
- Data: licensed papers and textbooks with document-level splits, verified
  proofs, lab-method extraction, contradictory-source sets, and citation tasks.
- Compute path: Hermes-quality small model, teacher-assisted data only where
  derivation rights exist, then 1–4B QLoRA; distill only after a measured teacher
  frontier exists.
- Evaluations: exact math, proof checking, source entailment, citation fidelity,
  calibrated abstention, and cross-domain held-out synthesis.
- Bottlenecks: contamination, copyrighted source rights, judge reliability,
  faithful citation, and 16 GB context limits.
- Dependency: a promoted Hermes data/evaluation/serving substrate.

## Perseus — tool-using operational agent

- Capability justification: reliable multi-step execution with recovery and
  explicit state, not free-form chat quality.
- Data: sandboxed tool schemas, successful and failed traces, permission
  boundaries, idempotency examples, rollback cases, and prompt-injection attacks.
- Compute path: adapter specialization on a promoted Hermes or Prometheus base;
  scale parameters only after execution reliability saturates.
- Evaluations: tool choice, typed arguments, result comparison, retries,
  checkpoint/resume, budget adherence, refusal, and side-effect containment.
- Bottlenecks: environment determinism, safe external writes, evaluation reset,
  and distinguishing language success from actual task success.
- Dependency: stable Percy-compatible task harness and model identity binding.

## Atlas — long-context knowledge and retrieval

- Capability justification: grounded analysis over large private corpora with
  citations and memory boundaries.
- Data: licensed long documents, retrieval positives/negatives, temporal
  updates, access-control labels, and multi-document question sets.
- Compute path: retrieval-first architecture; context extension only after
  retrieval baselines; distill retrieval decisions before expanding base size.
- Evaluations: recall, citation correctness, lost-in-the-middle, temporal
  freshness, access leakage, context scaling, latency, and RAM.
- Bottlenecks: KV-cache memory, stale indexes, private-data isolation, and
  context-length claims that do not translate into usable retrieval.
- Dependency: promoted safety and tool-use behavior from Perseus/Hermes.

## Kronos — temporal planning and continual adaptation

- Capability justification: forecast, plan, observe, correct, and learn without
  corrupting previous capabilities.
- Data: timestamped trajectories, intervention outcomes, delayed rewards,
  counterfactuals, drift episodes, and replay buffers with deletion provenance.
- Compute path: offline temporal adapters first; guarded replay and distillation;
  no online weight mutation until rollback and forgetting tests are proven.
- Evaluations: forecast calibration, plan success, recovery, causal attribution,
  catastrophic forgetting, data deletion, and checkpoint rollback.
- Bottlenecks: feedback delay, non-stationarity, hidden confounding, safe online
  learning, and reproducible environment snapshots.
- Dependency: Atlas grounding plus Perseus execution reliability.
