# Canonical Research Report: Olympus Model Architecture Proposal

**Research date:** 2026-09-06
**Scope:** architecture synthesis for Hermes, Prometheus, Perseus,
Olympus-Atlas, Kronos, and Aion.
**Evidence class:** primary arXiv papers, repository-local implementation, and
repository-local status records.
**Not in scope:** claiming qualifying family checkpoints, reporting unrun
experiments, or endorsing cited systems beyond what their papers establish.

## Executive finding

The proposed program is technically coherent only as a modular system on one
shared decoder/evidence/action substrate. Six independent foundation-model efforts
would multiply data, compute, serving, and evaluation risk without a current
empirical basis. Hermes should be the first learned specialization; Prometheus
and Perseus become orthogonal reasoning and action adapters; Olympus-Atlas is a
retrieval service with learned components; Kronos adds offline temporal state
and guarded adaptation; Aion is primarily a deterministic governed controller
whose learned router must earn inclusion against a rule-based baseline.

The repository now accompanies this design with executable reference
components for the shared substrate and each role. Those components establish
typed contracts, an authorized pre-model workspace view, selected process-local
validation/gating paths, and differentiable losses. A separate fixed,
two-phase reference replay composes all six roles along one synthetic path:
authorized Olympus-Atlas retrieval, trusted-receipt Prometheus selection, an
Aion approval pause, declaratively non-material Perseus execution,
non-authoritative Kronos observation, extractive Hermes response, and Aion
audit/STOP. Runtime-secret HMAC identifiers and a private in-memory reservation
capability bind the reviewed action scope, but no sandbox or durable recovery
service exists. This run establishes contract interoperability only; it does
not establish a trained family model, scientific result, or comparative
advantage. The bounded release-facing records are
`evidence/model_family_smoke_20260906.json` and
`evidence/model_composition_smoke_20260906.json`; both use deterministic
synthetic contract fixtures and explicitly authorize no promotion.

## Literature synthesis and design consequences

| Evidence | What the paper supports | Olympus consequence | What it does not establish |
|---|---|---|---|
| Transformer, arXiv:1706.03762 | attention-only sequence architecture | reuse a mature decoder backbone | that Olympus needs new pretraining |
| RoPE, arXiv:2104.09864; GQA, arXiv:2305.13245 | positional and inference-efficiency mechanisms | accept these as inherited Qwen3 properties | Olympus-specific gains |
| Qwen3 report, arXiv:2505.09388 | Qwen3 family and public 0.6B scale | justifies compatibility study with the already selected candidate | a trained Hermes checkpoint |
| RAG, arXiv:2005.11401; RETRO, arXiv:2112.04426 | non-parametric retrieval can augment generation | keep knowledge external and versioned | faithful attribution by default |
| ColBERT, arXiv:2004.12832 | effective late-interaction ranking | rerank an ACL-filtered candidate pool | private-corpus isolation |
| Lost in the Middle, arXiv:2307.03172 | long-context use is position-sensitive | test evidence position; retrieval before context scaling | that nominal context length is useful context |
| ReAct, arXiv:2210.03629; Toolformer, arXiv:2302.04761 | interleaved reasoning/action and learned API use | train tool selection and recovery traces | safe authority or side effects |
| PICARD, arXiv:2109.05093 | incremental parsing can reject invalid formal outputs | constrain Action envelopes during decoding | semantically correct or authorized calls |
| Tree of Thoughts, arXiv:2305.10601; self-consistency, arXiv:2203.11171 | inference-time branching and aggregation | compare bounded branch search at equal compute | scientific validity of a consensus branch |
| Let's Verify Step by Step, arXiv:2305.20050 | process supervision can outperform outcome-only supervision in its studied setting | separate proposer and process verifier | robust verification outside that setting |
| Language Models (Mostly) Know What They Know, arXiv:2207.05221 | self-evaluation/calibration can be trained and measured | add confidence heads and risk-coverage tests | universal out-of-domain calibration |
| MemGPT, arXiv:2310.08560 | explicit memory tiers can extend effective context | make memory an external, permissioned service | trustworthy autonomous memory writes |
| Decision Transformer, arXiv:2106.01345 | conditional sequence modeling for offline decisions | baseline return/constraint-conditioned planning | safe online adaptation |
| Mamba, arXiv:2312.00752 | selective state spaces scale linearly with sequence length | test a temporal encoder for long event streams | superiority for Olympus planning data |
| EWC, arXiv:1612.00796 | regularization approach to catastrophic forgetting | include as one guarded-adaptation baseline | solved continual learning |
| Constitutional AI, arXiv:2212.08073 | explicit principles can supervise behavior | encode policies in training examples | permission enforcement; rules remain external |
| AgentBench, arXiv:2308.03688; tau-bench, arXiv:2406.12045; ToolSandbox, arXiv:2408.04682 | interactive/stateful agent evaluation | evaluate actual state transitions and policy compliance | production safety certification |
| AI Scientist, arXiv:2408.06292 | end-to-end automated research loop is an active system direction | compare Aion's bounded lifecycle with prior automation | that Olympus has automated science |

## Decisions frozen by this report

1. One shared revision-pinned base and tokenizer; family-specific adapters and
   heads; independent services where security or falsifiability requires them.
2. Prefix evidence in version one; no custom cross-attention before a retrieval
   baseline exists.
3. In the target system, deterministic parsing, access control, permissions,
   transactions, protocol freezing, audit, and promotion remain outside neural
   components. Current references implement only the bounded subset recorded in
   the truth ledger.
4. Aion begins as a rule-based controller. A learned router is optional and is
   removed if it cannot beat the deterministic controller at equal budget and
   false-promotion rate.
5. `Olympus-Atlas` is the public name to avoid collision with the prior Atlas
   RAG model.
6. No online Kronos weight mutation before replay/deletion, forgetting, and
   atomic rollback tests pass.
7. Results must distinguish component metrics, end-to-end task success, and
   system safety; text saying “success” is never a completion metric.

## Evidence gaps

- No family-specific dataset satisfying provenance and held-out requirements.
- No revision-pinned external base snapshot in this repository.
- No completed 0.6B QLoRA comparison, qualifying trained adapter, calibrated
  held-out head evaluation, corpus-scale hybrid index study, stateful tool
  benchmark, temporal benchmark, or Aion study. Reference modules and synthetic
  unit-gradient paths do not close these gaps.
- No evidence that specialization beats prompting or deterministic baselines.
- No general, configurable, crash-durable, sandboxed, or autonomous composition
  of the six roles. The fixed in-process synthetic replay exercises one
  predetermined contract path and is not a family or system-quality result.
- No constrained token decoder, deployed sandbox, durable family transaction or
  Aion authority store, cryptographically signed approvals/audits, parsed
  rollback/deletion evidence service, or committed-effect rollback.
- No scale law or compute estimate supports training six separate bases.

## Primary-source links

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer / RoPE](https://arxiv.org/abs/2104.09864)
- [GQA](https://arxiv.org/abs/2305.13245)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- [RETRO](https://arxiv.org/abs/2112.04426)
- [ColBERT](https://arxiv.org/abs/2004.12832)
- [Lost in the Middle](https://arxiv.org/abs/2307.03172)
- [ReAct](https://arxiv.org/abs/2210.03629)
- [Toolformer](https://arxiv.org/abs/2302.04761)
- [PICARD](https://arxiv.org/abs/2109.05093)
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601)
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
- [Self-Consistency](https://arxiv.org/abs/2203.11171)
- [Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221)
- [MemGPT](https://arxiv.org/abs/2310.08560)
- [Decision Transformer](https://arxiv.org/abs/2106.01345)
- [Mamba](https://arxiv.org/abs/2312.00752)
- [Overcoming catastrophic forgetting](https://arxiv.org/abs/1612.00796)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)
- [AgentBench](https://arxiv.org/abs/2308.03688)
- [tau-bench](https://arxiv.org/abs/2406.12045)
- [ToolSandbox](https://arxiv.org/abs/2408.04682)
- [The AI Scientist](https://arxiv.org/abs/2408.06292)
