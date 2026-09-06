# Claim–Source Ledger

| ID | Manuscript claim | Support | Qualification |
|---|---|---|---|
| C1 | A decoder-only Transformer is a reasonable shared substrate. | arXiv:1706.03762; repository base decision | Design choice, not an experimental Olympus result. |
| C2 | Retrieval can externalize knowledge and provenance, but generated claims still need validation. | arXiv:2005.11401; 2112.04426 | The citations establish retrieval augmentation, not automatic faithfulness. |
| C3 | Long contexts can underuse evidence based on position. | arXiv:2307.03172 | Applies to studied models/tasks; motivates a test rather than a universal law. |
| C4 | Late interaction is a viable reranking mechanism. | arXiv:2004.12832 | Must be re-evaluated on private Olympus corpora and latency constraints. |
| C5 | Tool-use learning benefits from structured actions; syntactic validity is separable from authorization. | arXiv:2210.03629; 2302.04761; 2109.05093 | The authorization boundary is an Olympus design inference. |
| C6 | Branching and verification are plausible reasoning components. | arXiv:2305.10601; 2203.11171; 2305.20050 | No claim that these techniques improve Olympus before matched-budget tests. |
| C7 | Model confidence can be measured and calibrated, but generalization is limited. | arXiv:2207.05221 | Requires in/out-of-domain tests. |
| C8 | Explicit memory tiers are preferable to treating the context window as memory. | arXiv:2310.08560 | “Preferable” is an engineering judgment, not a universal empirical result. |
| C9 | Temporal sequence models and SSMs are candidates for offline planning/event streams. | arXiv:2106.01345; 2312.00752 | Both are ablations, not selected winners. |
| C10 | Catastrophic forgetting requires direct measurement under continual adaptation. | arXiv:1612.00796 | EWC is only one baseline and not assumed sufficient. |
| C11 | Stateful tool agents require interactive evaluation. | arXiv:2308.03688; 2406.12045; 2408.04682 | Benchmarks do not certify production safety. |
| C12 | Automated research loops exist as a research direction. | arXiv:2408.06292 | Olympus/Aion has no automated-research result. |
| C13 | All six Olympus families are currently specifications, not promoted models. | `TRUTH_MAP.md`, `research/SOURCE_OF_TRUTH.md`, repository artifacts | Direct local fact as of 2026-09-06. |
| C14 | The first planned external-base experiment uses Qwen3-0.6B-Base. | `research/HERMES_BASE_CANDIDATES.md`; arXiv:2505.09388 | Planned; snapshot/download/training not executed here. |
| C15 | `Atlas` collides with an existing retrieval-augmented model name. | arXiv:2208.03299 | Therefore the proposal uses `Olympus-Atlas`. |

## Integrity rule

Any future numerical result must be added with its run manifest, exact code and
data hashes, preregistered analysis, per-example outputs, and uncertainty. No
result row may be inferred from an architectural proposal or cited paper.
