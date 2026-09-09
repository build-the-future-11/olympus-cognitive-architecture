"""Detached Ed25519 attestations for model-promotion evidence.

The trust store is an operator-controlled policy input. A candidate cannot make
its own reports independent merely by signing them: release automation must pin
the reviewed trust store and keep the corresponding private keys outside the
training process.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Self

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import Field, model_validator

from olympus.core.schemas import StrictModel

AttestationKind = Literal["evaluation", "quantization", "license_review", "serving"]
_REQUIRED_KINDS: frozenset[str] = frozenset(
    {"evaluation", "quantization", "license_review", "serving"}
)
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def read_evidence_bytes(path: Path, *, limit: int = 10_000_000) -> bytes:
    """Take one bounded snapshot; parsing and hashing must share these bytes."""
    if limit < 1:
        raise ValueError("evidence byte limit must be positive")
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("evidence must be a regular file")
        if metadata.st_size > limit:
            raise ValueError("evidence exceeds the configured byte limit")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(limit + 1)
    finally:
        os.close(descriptor)
    if len(payload) > limit:
        raise ValueError("evidence exceeds the configured byte limit")
    return payload


def parse_strict_json(payload: bytes) -> Any:
    return json.loads(
        payload,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_constant,
    )


def load_strict_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object while rejecting duplicate keys and non-finite values."""

    payload = parse_strict_json(read_evidence_bytes(path))
    if not isinstance(payload, dict):
        raise ValueError("JSON document must contain an object")
    return payload


def _decode_base64(value: str, *, label: str, expected_bytes: int | None = None) -> bytes:
    try:
        decoded = base64.b64decode(value.encode("ascii"), validate=True)
    except ValueError as exc:
        raise ValueError(f"{label} must be canonical base64") from exc
    if expected_bytes is not None and len(decoded) != expected_bytes:
        raise ValueError(f"{label} must decode to {expected_bytes} bytes")
    if base64.b64encode(decoded).decode("ascii") != value:
        raise ValueError(f"{label} must use canonical padded base64")
    return decoded


class PromotionEvidenceSubject(StrictModel):
    """Exact candidate and evidence identities covered by every attestation."""

    schema_version: Literal[1] = 1
    requested_model_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,127}$")
    checkpoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    dataset_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    evaluation_sha256: str = Field(pattern=_SHA256_PATTERN)
    quantization_report_sha256: str = Field(pattern=_SHA256_PATTERN)
    model_card_sha256: str = Field(pattern=_SHA256_PATTERN)
    serving_verification_sha256: str = Field(pattern=_SHA256_PATTERN)
    approved_base_license: str = Field(min_length=1, max_length=200)


class AttestationStatement(StrictModel):
    schema_version: Literal[1] = 1
    scope: Literal["olympus-model-promotion-v1"] = "olympus-model-promotion-v1"
    attestation_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    kind: AttestationKind
    issuer_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    runner_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    runner_source_sha256: str = Field(pattern=_SHA256_PATTERN)
    issued_at: datetime
    expires_at: datetime
    outcome: Literal["passed", "approved"]
    evidence_sha256: str = Field(pattern=_SHA256_PATTERN)
    subject: PromotionEvidenceSubject

    @model_validator(mode="after")
    def validate_time_and_outcome(self) -> Self:
        if self.issued_at.utcoffset() is None or self.expires_at.utcoffset() is None:
            raise ValueError("attestation timestamps must be timezone-aware")
        if self.expires_at <= self.issued_at:
            raise ValueError("attestation expiry must be after issue time")
        expected = "approved" if self.kind == "license_review" else "passed"
        if self.outcome != expected:
            raise ValueError(f"{self.kind} attestation outcome must be {expected}")
        return self

    def canonical_bytes(self) -> bytes:
        return _canonical_json(self.model_dump(mode="json"))


class SignedAttestation(StrictModel):
    statement: AttestationStatement
    key_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    signature_base64: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_signature_encoding(self) -> Self:
        _decode_base64(self.signature_base64, label="signature", expected_bytes=64)
        return self


class AttestationBundle(StrictModel):
    schema_version: Literal[1] = 1
    attestations: list[SignedAttestation] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def reject_duplicate_records(self) -> Self:
        ids = [item.statement.attestation_id for item in self.attestations]
        if len(ids) != len(set(ids)):
            raise ValueError("attestation IDs must be unique")
        kinds = [item.statement.kind for item in self.attestations]
        if len(kinds) != len(set(kinds)):
            raise ValueError("attestation kinds must be unique")
        return self


class TrustedAttestationKey(StrictModel):
    key_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    issuer_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    runner_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    runner_source_sha256: str = Field(pattern=_SHA256_PATTERN)
    public_key_base64: str = Field(min_length=1, max_length=128)
    allowed_kinds: set[AttestationKind] = Field(min_length=1)
    authority_class: Literal["independent_verifier"] = "independent_verifier"
    revoked: bool = False

    @model_validator(mode="after")
    def validate_public_key(self) -> Self:
        _decode_base64(self.public_key_base64, label="public key", expected_bytes=32)
        return self


class AttestationTrustStore(StrictModel):
    schema_version: Literal[1] = 1
    minimum_distinct_issuers: int = Field(default=2, ge=2, le=4)
    keys: list[TrustedAttestationKey] = Field(min_length=2, max_length=64)

    @model_validator(mode="after")
    def reject_duplicate_keys(self) -> Self:
        ids = [item.key_id for item in self.keys]
        if len(ids) != len(set(ids)):
            raise ValueError("trusted key IDs must be unique")
        public_keys = [item.public_key_base64 for item in self.keys]
        if len(public_keys) != len(set(public_keys)):
            raise ValueError("each trusted issuer must use a distinct public key")
        return self


class AttestationVerification(StrictModel):
    passed: bool
    verified_kinds: list[str]
    verified_issuers: list[str]
    bundle_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    trust_store_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    issues: list[str]


def sign_attestation(
    statement: AttestationStatement,
    *,
    key_id: str,
    private_key: Ed25519PrivateKey,
) -> SignedAttestation:
    """Create a detached statement for an authorized runner process."""

    signature = private_key.sign(statement.canonical_bytes())
    return SignedAttestation(
        statement=statement,
        key_id=key_id,
        signature_base64=base64.b64encode(signature).decode("ascii"),
    )


def evidence_sha256_for_kind(subject: PromotionEvidenceSubject, kind: str) -> str:
    return {
        "evaluation": subject.evaluation_sha256,
        "quantization": subject.quantization_report_sha256,
        "license_review": subject.model_card_sha256,
        "serving": subject.serving_verification_sha256,
    }[kind]


def verify_promotion_attestations(
    *,
    bundle_path: Path | None,
    trust_store_path: Path | None,
    expected_subject: PromotionEvidenceSubject,
    now: datetime | None = None,
    maximum_lifetime: timedelta = timedelta(days=7),
) -> AttestationVerification:
    """Verify exact evidence identities against operator-selected trust roots."""

    if bundle_path is None or trust_store_path is None:
        return AttestationVerification(
            passed=False,
            verified_kinds=[],
            verified_issuers=[],
            issues=["signed attestation bundle and trust store are required"],
        )
    try:
        bundle_bytes = read_evidence_bytes(bundle_path, limit=1_000_000)
        trust_bytes = read_evidence_bytes(trust_store_path, limit=1_000_000)
        bundle = AttestationBundle.model_validate(parse_strict_json(bundle_bytes))
        trust = AttestationTrustStore.model_validate(parse_strict_json(trust_bytes))
    except (OSError, ValueError) as exc:
        return AttestationVerification(
            passed=False,
            verified_kinds=[],
            verified_issuers=[],
            issues=[f"invalid attestation input: {exc}"],
        )

    current = now or datetime.now(UTC)
    if current.utcoffset() is None:
        raise ValueError("verification time must be timezone-aware")
    trusted = {item.key_id: item for item in trust.keys}
    verified_kinds: set[str] = set()
    verified_issuers: set[str] = set()
    issues: list[str] = []
    for signed in bundle.attestations:
        statement = signed.statement
        key = trusted.get(signed.key_id)
        if key is None:
            issues.append(f"{statement.kind}: signing key is not trusted")
            continue
        if key.revoked:
            issues.append(f"{statement.kind}: signing key is revoked")
            continue
        if statement.issuer_id != key.issuer_id:
            issues.append(f"{statement.kind}: issuer does not match trusted key")
            continue
        if statement.kind not in key.allowed_kinds:
            issues.append(f"{statement.kind}: signing key is not allowed for this evidence kind")
            continue
        if statement.runner_id != key.runner_id:
            issues.append(f"{statement.kind}: runner identity does not match trusted policy")
            continue
        if statement.runner_source_sha256 != key.runner_source_sha256:
            issues.append(f"{statement.kind}: runner source does not match trusted policy")
            continue
        if statement.subject != expected_subject:
            issues.append(f"{statement.kind}: attested subject does not match the candidate")
            continue
        expected_evidence = evidence_sha256_for_kind(expected_subject, statement.kind)
        if statement.evidence_sha256 != expected_evidence:
            issues.append(f"{statement.kind}: attested evidence hash is incorrect")
            continue
        if statement.issued_at > current + timedelta(minutes=5):
            issues.append(f"{statement.kind}: attestation issue time is in the future")
            continue
        if statement.expires_at <= current:
            issues.append(f"{statement.kind}: attestation is expired")
            continue
        if statement.expires_at - statement.issued_at > maximum_lifetime:
            issues.append(f"{statement.kind}: attestation lifetime exceeds policy")
            continue
        try:
            public_key = Ed25519PublicKey.from_public_bytes(
                _decode_base64(
                    key.public_key_base64,
                    label="public key",
                    expected_bytes=32,
                )
            )
            public_key.verify(
                _decode_base64(
                    signed.signature_base64,
                    label="signature",
                    expected_bytes=64,
                ),
                statement.canonical_bytes(),
            )
        except (InvalidSignature, ValueError):
            issues.append(f"{statement.kind}: signature verification failed")
            continue
        verified_kinds.add(statement.kind)
        verified_issuers.add(statement.issuer_id)

    missing = sorted(_REQUIRED_KINDS - verified_kinds)
    if missing:
        issues.append(f"missing verified evidence kinds: {', '.join(missing)}")
    if len(verified_issuers) < trust.minimum_distinct_issuers:
        issues.append(
            "verified evidence does not meet the minimum distinct-issuer policy"
        )
    return AttestationVerification(
        passed=not issues,
        verified_kinds=sorted(verified_kinds),
        verified_issuers=sorted(verified_issuers),
        bundle_sha256=hashlib.sha256(bundle_bytes).hexdigest(),
        trust_store_sha256=hashlib.sha256(trust_bytes).hexdigest(),
        issues=issues,
    )
