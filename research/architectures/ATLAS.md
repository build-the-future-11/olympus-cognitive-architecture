# Olympus-Atlas Architecture

## Naming and purpose

The external name is **Olympus-Atlas** because `Atlas` already identifies the
retrieval-augmented model in arXiv:2208.03299. Olympus-Atlas grounds other
families over large, changing, private corpora with inspectable provenance. It
does not equate nominal context length with usable knowledge access.

## Target architecture

1. **Ingestion:** parse documents into overlapping semantic passages; preserve
   document/version hashes, timestamps, licenses, ACL labels, and span offsets.
2. **Security filter:** remove unauthorized documents before lexical or neural
   scoring so ranks and timing do not leak hidden corpus membership.
3. **Candidate retrieval:** fuse BM25 and a compact bi-encoder using reciprocal
   rank fusion. Both remain independently measurable.
4. **Late-interaction reranker:** ColBERT-style token-level scoring reranks the
   authorized candidate pool; a freshness feature is explicit, not inferred.
5. **Context composer:** selects diverse passages under a token budget, includes
   source IDs beside every span, and randomizes evidence position to measure and
   train against lost-in-the-middle effects.
6. **Attribution adapter:** a shared decoder adapter answers or produces an
   evidence graph; a separate entailment head validates claim/source edges.
7. **Version manager:** the target durable service atomically swaps immutable
   indexes and allows exact replay against prior versions.

## Data and losses

Use rights-cleared query/passage judgments, hard negatives from the same
document, temporal replacements, contradictions, ACL canaries, and
multi-document questions. Train contrastive retrieval, late-interaction ranking,
freshness selection, attribution, and abstention. Never generate ACL labels.

## Evaluation and stop rule

Measure Recall@k, nDCG@10, answer/citation correctness, attribution entailment,
freshness, position sensitivity, index lag, latency/RAM, and zero tolerated ACL
leakage. Compare BM25, dense-only, long-context/no-retrieval, hybrid without
reranking, and the full system. Reject architectural expansion if hybrid search
does not beat BM25 at matched latency, or if longer contexts merely hide poor
retrieval.

Target input: `AuthorizedQuery + CorpusVersion`. Target output: immutable `EvidenceSet` or
`InsufficientEvidence`. Current state: **reference implementation exists, no
qualifying checkpoint or production index**. `olympus/models/atlas.py` provides
process-local indexes identified by opaque keyed versions, ACL-before-ranking
hybrid retrieval, ACL-filtered `get_index` projections, and a trainable
bi-encoder/late-interaction scorer. Version replay and locking apply only within
one service process; no durable or cross-process atomic index store exists. The
learned scorer is not wired into the deterministic service, and no corpus-scale
comparative retrieval result exists.
