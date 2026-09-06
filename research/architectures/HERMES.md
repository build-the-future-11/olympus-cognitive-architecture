# Hermes Architecture

## Purpose and falsifiable claim

Hermes is the grounded human-facing interface: interpret a request, retrieve
authorized evidence, answer with span-level citations, remember only approved
facts, and abstain when support is inadequate. The initial claim is narrow:
adapter specialization improves grounded-response utility and calibration over
the same frozen base and retrieval context.

## Architecture

1. **Request interpreter:** a LoRA adapter on the shared decoder classifies
   intent, decomposes compound requests, and produces retrieval queries.
2. **Evidence intake:** Olympus-Atlas or a baseline retriever returns immutable
   evidence IDs. A deterministic composer packs diverse passages and varies
   relevant-passage position during training.
3. **Grounded decoder:** the same adapter generates an `Answer` envelope.
4. **Attribution head:** scores each claim/evidence pair for support, refutation,
   or insufficiency. It may veto unsupported claims but cannot invent sources.
5. **Confidence head:** predicts answer correctness and evidence sufficiency;
   thresholds are calibrated on a held-out domain and evaluated out of domain.
6. **Memory selector:** proposes memory writes. An ACL/provenance service, not
   the model, authorizes persistence and supplies future memory as evidence.

The first version uses prefix evidence rather than architectural
cross-attention. Retrieval, memory, and validation remain separately testable.

## Data and losses

Use licensed question/evidence/answer triples, contradiction pairs, answerable
and unanswerable requests, citation-span labels, memory-consent examples, and
prompt-injection negatives. Split by document and source, not row. In addition
to the shared loss, train claim attribution, abstention, Brier-style confidence,
and memory-write selection losses.

## Evaluation and stop rule

Report answer correctness, citation precision/recall, claim entailment,
expected calibration error, Brier score, risk-coverage, unsupported-claim rate,
memory leakage, and user utility under fixed latency. Compare no retrieval,
BM25, the shared unadapted base, and Hermes. Stop the family if gains disappear
under document-level decontamination or if citation appearance rises without
claim support.

## Interfaces and promotion

Input: `WorkspaceState + UserRequest`. Output: `Answer | Proposal | Stop`.
Hermes cannot execute tools. Promotion additionally requires the repository's
existing dataset-size, quantization, safety, serving, license, and model-card
gates. Current state: **specified, no qualifying checkpoint**.
