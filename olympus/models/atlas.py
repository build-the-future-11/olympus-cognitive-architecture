from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import secrets
import threading
from collections import Counter
from dataclasses import asdict, dataclass, replace
from typing import TypedDict

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from olympus.models.substrate import acl_labels_are_authorized

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True, slots=True)
class AtlasDocument:
    document_id: str
    text: str
    source_uri: str
    source_version: str
    acquired_at: str
    license_id: str
    acl_labels: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AtlasPassage:
    passage_id: str
    document_id: str
    source_uri: str
    source_version: str
    acquired_at: str
    license_id: str
    acl_labels: tuple[str, ...]
    span_start: int
    span_end: int
    text: str
    document_hash: str
    passage_hash: str


@dataclass(frozen=True, slots=True)
class AuthorizedQuery:
    text: str
    granted_acl: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class EvidenceHit:
    passage: AtlasPassage
    lexical_score: float
    dense_score: float
    reciprocal_rank_score: float
    late_interaction_score: float
    final_score: float


@dataclass(frozen=True, slots=True)
class EvidenceSet:
    evidence_set_id: str
    corpus_version: str
    query_hash: str
    hits: tuple[EvidenceHit, ...]


@dataclass(frozen=True, slots=True)
class InsufficientEvidence:
    corpus_version: str
    reason: str


@dataclass(frozen=True, slots=True)
class VersionedAtlasIndex:
    corpus_version: str
    passages: tuple[AtlasPassage, ...]
    dense_dimensions: int
    rrf_constant: int
    passage_words: int
    overlap_words: int


class OlympusAtlasService:
    """Process-local hybrid retrieval with ACL filtering before scoring."""

    def __init__(self, *, dense_dimensions: int = 96, rrf_constant: int = 60) -> None:
        if dense_dimensions < 8 or rrf_constant < 1:
            raise ValueError("dense_dimensions must be >= 8 and rrf_constant must be positive")
        self.dense_dimensions = dense_dimensions
        self.rrf_constant = rrf_constant
        self._indexes: dict[str, VersionedAtlasIndex] = {}
        self._active_version: str | None = None
        self._version_secret = secrets.token_bytes(32)
        self._lock = threading.RLock()

    @property
    def active_version(self) -> str | None:
        with self._lock:
            return self._active_version

    def build_index(
        self,
        documents: tuple[AtlasDocument, ...] | list[AtlasDocument],
        *,
        passage_words: int = 80,
        overlap_words: int = 20,
    ) -> str:
        if passage_words < 1 or not 0 <= overlap_words < passage_words:
            raise ValueError("require 0 <= overlap_words < passage_words")
        identities = [(document.document_id, document.source_version) for document in documents]
        if len(identities) != len(set(identities)):
            raise ValueError("document ID and source version pairs must be unique")
        ordered_documents = sorted(
            documents,
            key=lambda document: (
                document.document_id,
                document.source_version,
                document.source_uri,
                hashlib.sha256(document.text.encode("utf-8")).hexdigest(),
            ),
        )
        passages = tuple(
            passage
            for document in ordered_documents
            for passage in _passages_for_document(
                document, passage_words=passage_words, overlap_words=overlap_words
            )
        )
        passage_ids = [passage.passage_id for passage in passages]
        if len(passage_ids) != len(set(passage_ids)):
            raise ValueError("generated Atlas passage IDs must be unique")
        canonical = json.dumps(
            {
                "dense_dimensions": self.dense_dimensions,
                "rrf_constant": self.rrf_constant,
                "passage_words": passage_words,
                "overlap_words": overlap_words,
                "passages": [asdict(passage) for passage in passages],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        # Expose a keyed, process-local opaque identifier rather than a raw
        # commitment over protected plaintext. This prevents offline guessing of
        # low-entropy private passages from a public version identifier.
        corpus_version = hmac.new(
            self._version_secret, canonical, hashlib.sha256
        ).hexdigest()
        index = VersionedAtlasIndex(
            corpus_version=corpus_version,
            passages=passages,
            dense_dimensions=self.dense_dimensions,
            rrf_constant=self.rrf_constant,
            passage_words=passage_words,
            overlap_words=overlap_words,
        )
        with self._lock:
            self._indexes.setdefault(corpus_version, index)
            self._active_version = corpus_version
        return corpus_version

    def retrieve(
        self,
        query: AuthorizedQuery,
        *,
        corpus_version: str | None = None,
        top_k: int = 5,
        candidate_k: int = 30,
    ) -> EvidenceSet | InsufficientEvidence:
        if top_k < 1 or candidate_k < top_k:
            raise ValueError("require candidate_k >= top_k >= 1")
        with self._lock:
            version = corpus_version or self._active_version
            index = self._indexes.get(version) if version is not None else None
        if version is None or index is None:
            return InsufficientEvidence(version or "unavailable", "corpus_version_not_found")

        # This filter intentionally precedes tokenization, corpus statistics, and scoring.
        authorized = tuple(
            passage
            for passage in index.passages
            if _is_authorized(passage.acl_labels, query.granted_acl)
        )
        if not authorized:
            return InsufficientEvidence(version, "no_authorized_evidence")
        query_tokens = _tokens(query.text)
        if not query_tokens:
            return InsufficientEvidence(version, "query_has_no_retrievable_terms")

        lexical = _bm25_rank(query_tokens, authorized)
        dense = _dense_rank(query_tokens, authorized, index.dense_dimensions)
        lexical_rank = {passage_id: rank for rank, (passage_id, _) in enumerate(lexical, 1)}
        dense_rank = {passage_id: rank for rank, (passage_id, _) in enumerate(dense, 1)}
        lexical_score = dict(lexical)
        dense_score = dict(dense)
        passage_by_id = {passage.passage_id: passage for passage in authorized}

        fused: list[tuple[str, float]] = []
        for passage_id in passage_by_id:
            score = 1.0 / (index.rrf_constant + lexical_rank[passage_id])
            score += 1.0 / (index.rrf_constant + dense_rank[passage_id])
            fused.append((passage_id, score))
        fused.sort(key=lambda row: (-row[1], row[0]))

        reranked: list[EvidenceHit] = []
        for passage_id, rrf_score in fused[:candidate_k]:
            passage = passage_by_id[passage_id]
            late_score = _late_interaction(
                query_tokens, _tokens(passage.text), index.dense_dimensions
            )
            reranked.append(
                EvidenceHit(
                    passage=passage,
                    lexical_score=lexical_score[passage_id],
                    dense_score=dense_score[passage_id],
                    reciprocal_rank_score=rrf_score,
                    late_interaction_score=late_score,
                    final_score=rrf_score + late_score,
                )
            )
        reranked.sort(key=lambda hit: (-hit.final_score, hit.passage.passage_id))
        hits = tuple(reranked[:top_k])
        if not hits or hits[0].late_interaction_score <= 0.0:
            return InsufficientEvidence(version, "insufficient_retrieval_support")

        query_hash = hashlib.sha256(query.text.encode("utf-8")).hexdigest()
        identity = json.dumps(
            {
                "corpus_version": version,
                "query_hash": query_hash,
                "passages": [hit.passage.passage_id for hit in hits],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return EvidenceSet(
            evidence_set_id=hashlib.sha256(identity).hexdigest(),
            corpus_version=version,
            query_hash=query_hash,
            hits=hits,
        )

    def get_index(
        self,
        corpus_version: str,
        *,
        granted_acl: frozenset[str] = frozenset(),
    ) -> VersionedAtlasIndex | None:
        """Return only the caller's authorized projection of a local index."""

        with self._lock:
            index = self._indexes.get(corpus_version)
        if index is None:
            return None
        return replace(
            index,
            passages=tuple(
                passage
                for passage in index.passages
                if _is_authorized(passage.acl_labels, granted_acl)
            ),
        )


class AtlasForwardOutput(TypedDict, total=False):
    dense_logits: Tensor
    late_interaction_logits: Tensor
    scores: Tensor
    loss: Tensor


class OlympusAtlasRetriever(nn.Module):
    """Trainable bi-encoder plus ColBERT-style late-interaction scorer."""

    def __init__(self, vocab_size: int, hidden_size: int = 64, temperature: float = 0.1) -> None:
        super().__init__()
        if (
            vocab_size < 2
            or hidden_size < 2
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError("invalid retriever dimensions or temperature")
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.score_mix = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.temperature = temperature

    def forward(
        self,
        query_ids: Tensor,
        passage_ids: Tensor,
        *,
        query_mask: Tensor | None = None,
        passage_mask: Tensor | None = None,
        targets: Tensor | None = None,
    ) -> AtlasForwardOutput:
        if query_ids.ndim != 2 or passage_ids.ndim != 3:
            raise ValueError("expected query [batch, tokens] and passages [batch, items, tokens]")
        if query_ids.shape[0] != passage_ids.shape[0]:
            raise ValueError("query and passage batch sizes differ")
        if query_ids.shape[0] == 0 or query_ids.shape[1] == 0:
            raise ValueError("query batch and token dimensions must be non-empty")
        if passage_ids.shape[1] == 0 or passage_ids.shape[2] == 0:
            raise ValueError("passage item and token dimensions must be non-empty")
        _validate_token_ids(query_ids, self.embedding.num_embeddings, "query_ids")
        _validate_token_ids(passage_ids, self.embedding.num_embeddings, "passage_ids")
        if query_mask is not None and query_mask.shape != query_ids.shape:
            raise ValueError("query_mask must match query_ids")
        if passage_mask is not None and passage_mask.shape != passage_ids.shape:
            raise ValueError("passage_mask must match passage_ids")
        query_mask = query_ids.ne(0) if query_mask is None else query_mask.bool()
        passage_mask = passage_ids.ne(0) if passage_mask is None else passage_mask.bool()
        if torch.any(query_mask.sum(dim=1) == 0):
            raise ValueError("each query requires at least one unmasked token")
        valid_passages = passage_mask.any(dim=-1)
        if torch.any(valid_passages.sum(dim=1) == 0):
            raise ValueError("each query requires at least one valid passage")

        query_tokens = F.normalize(self.projection(self.embedding(query_ids)), dim=-1)
        passage_tokens = F.normalize(self.projection(self.embedding(passage_ids)), dim=-1)
        query_vector = F.normalize(_masked_mean(query_tokens, query_mask), dim=-1)
        passage_vectors = F.normalize(_masked_mean(passage_tokens, passage_mask), dim=-1)
        dense_logits = torch.einsum("bh,bnh->bn", query_vector, passage_vectors)

        similarities = torch.einsum("bqh,bnph->bnqp", query_tokens, passage_tokens)
        similarities = similarities.masked_fill(
            ~passage_mask.unsqueeze(2), torch.finfo(similarities.dtype).min
        )
        maximum = similarities.max(dim=-1).values
        maximum = maximum.masked_fill(~valid_passages.unsqueeze(-1), 0.0)
        late_logits = (
            maximum * query_mask.unsqueeze(1).to(dtype=maximum.dtype)
        ).sum(dim=-1) / query_mask.sum(dim=-1, keepdim=True).clamp_min(1)

        weights = self.score_mix.softmax(dim=0)
        scores = (weights[0] * dense_logits + weights[1] * late_logits) / self.temperature
        scores = scores.masked_fill(~valid_passages, torch.finfo(scores.dtype).min)
        output: AtlasForwardOutput = {
            "dense_logits": dense_logits,
            "late_interaction_logits": late_logits,
            "scores": scores,
        }
        if targets is not None:
            if targets.shape != (query_ids.shape[0],):
                raise ValueError("retrieval targets must contain one passage index per query")
            if targets.dtype not in {torch.int32, torch.int64}:
                raise ValueError("retrieval targets must use an integer tensor dtype")
            if torch.any(targets < 0) or torch.any(targets >= passage_ids.shape[1]):
                raise ValueError("retrieval target index is outside the passage set")
            selected_validity = valid_passages.gather(1, targets.unsqueeze(1)).squeeze(1)
            if not torch.all(selected_validity):
                raise ValueError("retrieval targets must identify valid passages")
            output["loss"] = F.cross_entropy(scores, targets)
        return output


def _passages_for_document(
    document: AtlasDocument, *, passage_words: int, overlap_words: int
) -> tuple[AtlasPassage, ...]:
    if passage_words < 1 or not 0 <= overlap_words < passage_words:
        raise ValueError("require 0 <= overlap_words < passage_words")
    matches = tuple(re.finditer(r"\S+", document.text))
    if not matches:
        return ()
    document_hash = hashlib.sha256(document.text.encode("utf-8")).hexdigest()
    identity = json.dumps(
        {
            "document_id": document.document_id,
            "source_uri": document.source_uri,
            "source_version": document.source_version,
            "document_hash": document_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    document_identity_hash = hashlib.sha256(identity).hexdigest()
    step = passage_words - overlap_words
    passages: list[AtlasPassage] = []
    for start_word in range(0, len(matches), step):
        selected = matches[start_word : start_word + passage_words]
        if not selected:
            break
        span_start = selected[0].start()
        span_end = selected[-1].end()
        text = document.text[span_start:span_end]
        passage_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        passage_id = f"passage:{document_identity_hash}:{start_word}"
        passages.append(
            AtlasPassage(
                passage_id=passage_id,
                document_id=document.document_id,
                source_uri=document.source_uri,
                source_version=document.source_version,
                acquired_at=document.acquired_at,
                license_id=document.license_id,
                acl_labels=document.acl_labels,
                span_start=span_start,
                span_end=span_end,
                text=text,
                document_hash=document_hash,
                passage_hash=passage_hash,
            )
        )
        if start_word + passage_words >= len(matches):
            break
    return tuple(passages)


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in _TOKEN_RE.findall(text))


def _is_authorized(required: tuple[str, ...], granted: frozenset[str]) -> bool:
    return acl_labels_are_authorized(required, granted)


def _bm25_rank(
    query_tokens: tuple[str, ...], passages: tuple[AtlasPassage, ...]
) -> list[tuple[str, float]]:
    tokenized = {passage.passage_id: _tokens(passage.text) for passage in passages}
    average_length = sum(map(len, tokenized.values())) / max(1, len(tokenized))
    document_frequency = Counter(
        token for tokens in tokenized.values() for token in set(tokens)
    )
    query_counts = Counter(query_tokens)
    ranked: list[tuple[str, float]] = []
    for passage in passages:
        tokens = tokenized[passage.passage_id]
        counts = Counter(tokens)
        score = 0.0
        for token, query_frequency in query_counts.items():
            frequency = counts[token]
            if frequency == 0:
                continue
            idf = math.log(1.0 + (len(passages) - document_frequency[token] + 0.5) / (
                document_frequency[token] + 0.5
            ))
            denominator = frequency + 1.5 * (1.0 - 0.75 + 0.75 * len(tokens) / average_length)
            score += query_frequency * idf * frequency * 2.5 / denominator
        ranked.append((passage.passage_id, score))
    return sorted(ranked, key=lambda row: (-row[1], row[0]))


def _dense_rank(
    query_tokens: tuple[str, ...], passages: tuple[AtlasPassage, ...], dimensions: int
) -> list[tuple[str, float]]:
    query_vector = _mean_hash_vector(query_tokens, dimensions)
    ranked = [
        (
            passage.passage_id,
            _dot(query_vector, _mean_hash_vector(_tokens(passage.text), dimensions)),
        )
        for passage in passages
    ]
    return sorted(ranked, key=lambda row: (-row[1], row[0]))


def _late_interaction(
    query_tokens: tuple[str, ...], passage_tokens: tuple[str, ...], dimensions: int
) -> float:
    if not passage_tokens:
        return 0.0
    passage_vectors = tuple(_hash_vector(token, dimensions) for token in passage_tokens)
    maxima = [
        max(
            _dot(_hash_vector(token, dimensions), passage_vector)
            for passage_vector in passage_vectors
        )
        for token in query_tokens
    ]
    return sum(maxima) / len(maxima)


def _mean_hash_vector(tokens: tuple[str, ...], dimensions: int) -> tuple[float, ...]:
    if not tokens:
        return (0.0,) * dimensions
    vectors = tuple(_hash_vector(token, dimensions) for token in tokens)
    mean = [sum(vector[index] for vector in vectors) / len(vectors) for index in range(dimensions)]
    norm = math.sqrt(sum(value * value for value in mean)) or 1.0
    return tuple(value / norm for value in mean)


def _hash_vector(token: str, dimensions: int) -> tuple[float, ...]:
    vector = [0.0] * dimensions
    padded = f"^{token}$"
    features = {token, *(padded[index : index + 3] for index in range(max(1, len(padded) - 2)))}
    for feature in features:
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        vector[bucket] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return tuple(value / norm for value in vector)


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)


def _validate_token_ids(token_ids: Tensor, vocab_size: int, name: str) -> None:
    if token_ids.dtype not in {torch.int32, torch.int64}:
        raise ValueError(f"{name} must use an integer tensor dtype")
    if torch.any(token_ids < 0) or torch.any(token_ids >= vocab_size):
        raise ValueError(f"{name} contains a token outside the configured vocabulary")
