# Olympus Model Architecture Program

**Status: design specification, 2026-09-06.** No document in this directory is
evidence that a named model has been trained, evaluated, or promoted.

The six names are roles on one governed substrate, not six independent
foundation-model pretraining projects:

| Family | System role | Learned core | Non-learned boundary | First decision gate |
|---|---|---|---|---|
| Hermes | grounded user interface | decoder adapter + confidence/evidence heads | retrieval, memory ACL, citation validator | beats the frozen Qwen baseline on held-out grounded response quality |
| Prometheus | scientific synthesis | proposer and verifier adapters | calculators, proof/checking tools, evidence ledger | proposer+verifier beats equal-compute sampling |
| Perseus | tool execution | action and recovery adapters | capability kernel, schema parser, transaction manager | beats deterministic planner without increasing unsafe side effects |
| Olympus-Atlas | private-corpus retrieval | retriever/reranker and attribution adapter | ACL filter, index/version manager | hybrid retrieval beats BM25 and dense-only baselines |
| Kronos | temporal planning | temporal encoder + planner adapter | replay store, rollback, update gate | beats static replanning without measurable forgetting |
| Aion | governed research orchestration | optional proposal/routing policy | protocol state machine, Pantheon audit, human approval | beats deterministic orchestration at equal budget and false-promotion rate |

`Atlas` is already the name of a retrieval-augmented language model in
arXiv:2208.03299. External artifacts therefore use **Olympus-Atlas** and all
checkpoint identifiers must use the `olympus-` prefix. The other family names
are likewise codenames until their promotion gates pass.

Read [SHARED_SUBSTRATE.md](SHARED_SUBSTRATE.md) first, then the family files.
The paper source and the claim/source audit live in `research/paper/`.
