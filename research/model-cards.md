# Model Cards

## Foundry Verification Bigram

- Identity: content-addressed `foundry-verification-bigram-<hash>` assigned only
  after a passing held-out evaluation.
- Training: real maximum-likelihood character-transition counts over the
  repository-owned Foundry verification corpus.
- Baseline: frozen uniform next-character distribution over the learned
  alphabet.
- Promotion: candidate held-out negative log-likelihood and perplexity must both
  beat the baseline.
- Serving: portable JSON checkpoint through the local OpenAI-compatible API.
- Integrity: SHA-256 verified before health or generation.
- Intended use: proving the complete Foundry lifecycle at negligible cost.
- Limitations: not an assistant, reasoner, coding model, research model, or
  Hermes checkpoint.

## Hermes Nano

- Implementation: `olympus.models.families.HermesNano`.
- Purpose: deterministic interpretation-and-memory integration fixture.
- Inputs: a non-empty text prompt supplied directly by the caller.
- Outputs: one interpretive statement, its heuristic confidence, and its
  interpretation category.
- Memory: the selected statement is persisted to a caller-provided SQLite
  `MemoryStore`.
- Training: none. Hermes Nano is not a pretrained language model and has no
  claimed parameter count.
- Evaluation: the automated suite verifies local persistence and a non-zero
  confidence result. The demo suite verifies executable integration.
- Limitations: responses are chosen from deterministic interpretive heuristics;
  the runtime does not provide general language-model capabilities, factual
  guarantees, or safety certification.
- Intended use: local integration tests and demonstrations of Olympus memory
  and interpretation interfaces.
- Naming caveat: the historical class name does not indicate a trained Hermes
  family model and cannot satisfy any Hermes release gate.
