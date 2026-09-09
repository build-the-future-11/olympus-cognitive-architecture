from __future__ import annotations

import base64
import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from olympus.foundry import attestation
from olympus.foundry.attestation import (
    AttestationBundle,
    AttestationKind,
    AttestationStatement,
    AttestationTrustStore,
    PromotionEvidenceSubject,
    SignedAttestation,
    TrustedAttestationKey,
    sign_attestation,
    verify_promotion_attestations,
)


def test_evidence_snapshot_rejects_oversized_and_special_files(tmp_path: Path) -> None:
    path = tmp_path / "evidence"
    path.write_bytes(b"1234")
    with pytest.raises(ValueError, match="byte limit"):
        attestation.read_evidence_bytes(path, limit=3)
    with pytest.raises(ValueError, match="positive"):
        attestation.read_evidence_bytes(path, limit=0)
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular file"):
        attestation.read_evidence_bytes(fifo)


def test_verifier_hashes_the_exact_parsed_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_path, trust_path, _, _ = _signed_inputs(tmp_path)
    originals = {path: path.read_bytes() for path in (bundle_path, trust_path)}
    reader = attestation.read_evidence_bytes

    def replacing_reader(path: Path, *, limit: int = 10_000_000) -> bytes:
        payload = reader(path, limit=limit)
        path.write_bytes(b"replaced after read")
        return payload

    monkeypatch.setattr(attestation, "read_evidence_bytes", replacing_reader)
    result = verify_promotion_attestations(
        bundle_path=bundle_path, trust_store_path=trust_path, expected_subject=_subject()
    )
    assert result.passed
    assert result.bundle_sha256 == hashlib.sha256(originals[bundle_path]).hexdigest()
    assert result.trust_store_sha256 == hashlib.sha256(originals[trust_path]).hexdigest()


def _subject() -> PromotionEvidenceSubject:
    return PromotionEvidenceSubject(
        requested_model_id="hermes-fixture",
        checkpoint_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        evaluation_sha256="c" * 64,
        quantization_report_sha256="d" * 64,
        model_card_sha256="e" * 64,
        serving_verification_sha256="f" * 64,
        approved_base_license="Apache-2.0",
    )


def _public_key_base64(private_key: Ed25519PrivateKey) -> str:
    return base64.b64encode(
        private_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
    ).decode("ascii")


def _signed_inputs(
    tmp_path: Path,
    *,
    issued_at: datetime | None = None,
    expires_at: datetime | None = None,
) -> tuple[Path, Path, AttestationBundle, AttestationTrustStore]:
    subject = _subject()
    evaluation_key = Ed25519PrivateKey.generate()
    release_key = Ed25519PrivateKey.generate()
    trust = AttestationTrustStore(
        keys=[
            TrustedAttestationKey(
                key_id="evaluation-key",
                issuer_id="evaluation-lab",
                runner_id="evaluation-runner",
                runner_source_sha256="1" * 64,
                public_key_base64=_public_key_base64(evaluation_key),
                allowed_kinds={"evaluation", "quantization"},
            ),
            TrustedAttestationKey(
                key_id="release-key",
                issuer_id="release-lab",
                runner_id="release-runner",
                runner_source_sha256="1" * 64,
                public_key_base64=_public_key_base64(release_key),
                allowed_kinds={"license_review", "serving"},
            ),
        ]
    )
    issued = issued_at or datetime.now(UTC) - timedelta(minutes=1)
    expires = expires_at or issued + timedelta(days=1)
    specs: tuple[tuple[AttestationKind, str, str, Ed25519PrivateKey], ...] = (
        ("evaluation", "evaluation-lab", "evaluation-key", evaluation_key),
        ("quantization", "evaluation-lab", "evaluation-key", evaluation_key),
        ("license_review", "release-lab", "release-key", release_key),
        ("serving", "release-lab", "release-key", release_key),
    )
    evidence = {
        "evaluation": subject.evaluation_sha256,
        "quantization": subject.quantization_report_sha256,
        "license_review": subject.model_card_sha256,
        "serving": subject.serving_verification_sha256,
    }
    signed: list[SignedAttestation] = []
    for kind, issuer_id, key_id, private_key in specs:
        outcome: Literal["approved", "passed"] = (
            "approved" if kind == "license_review" else "passed"
        )
        statement = AttestationStatement(
            attestation_id=f"attestation-{kind}",
            kind=kind,
            issuer_id=issuer_id,
            runner_id=(
                "evaluation-runner"
                if kind in {"evaluation", "quantization"}
                else "release-runner"
            ),
            runner_source_sha256="1" * 64,
            issued_at=issued,
            expires_at=expires,
            outcome=outcome,
            evidence_sha256=evidence[kind],
            subject=subject,
        )
        signed.append(
            sign_attestation(statement, key_id=key_id, private_key=private_key)
        )
    bundle = AttestationBundle(attestations=signed)
    bundle_path = tmp_path / "bundle.json"
    trust_path = tmp_path / "trust.json"
    bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    trust_path.write_text(trust.model_dump_json(indent=2), encoding="utf-8")
    return bundle_path, trust_path, bundle, trust


def test_attestation_verification_accepts_fresh_exact_signed_evidence(
    tmp_path: Path,
) -> None:
    bundle_path, trust_path, _, _ = _signed_inputs(tmp_path)

    result = verify_promotion_attestations(
        bundle_path=bundle_path,
        trust_store_path=trust_path,
        expected_subject=_subject(),
    )

    assert result.passed is True
    assert result.verified_kinds == [
        "evaluation",
        "license_review",
        "quantization",
        "serving",
    ]
    assert result.verified_issuers == ["evaluation-lab", "release-lab"]
    assert result.bundle_sha256 is not None
    assert result.trust_store_sha256 is not None


def test_attestation_verification_rejects_signature_and_subject_tampering(
    tmp_path: Path,
) -> None:
    bundle_path, trust_path, bundle, _ = _signed_inputs(tmp_path)
    original = base64.b64decode(bundle.attestations[0].signature_base64)
    changed = bytes([original[0] ^ 1]) + original[1:]
    tampered = bundle.attestations[0].model_copy(
        update={"signature_base64": base64.b64encode(changed).decode("ascii")}
    )
    bundle_path.write_text(
        bundle.model_copy(
            update={"attestations": [tampered, *bundle.attestations[1:]]}
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    signature_result = verify_promotion_attestations(
        bundle_path=bundle_path,
        trust_store_path=trust_path,
        expected_subject=_subject(),
    )
    different_subject = _subject().model_copy(update={"checkpoint_sha256": "0" * 64})
    subject_result = verify_promotion_attestations(
        bundle_path=bundle_path,
        trust_store_path=trust_path,
        expected_subject=different_subject,
    )

    assert signature_result.passed is False
    assert "evaluation: signature verification failed" in signature_result.issues
    assert subject_result.passed is False
    assert any("subject does not match" in issue for issue in subject_result.issues)


def test_attestation_verification_rejects_expired_revoked_and_duplicate_json(
    tmp_path: Path,
) -> None:
    issued = datetime.now(UTC) - timedelta(days=2)
    bundle_path, trust_path, bundle, trust = _signed_inputs(
        tmp_path,
        issued_at=issued,
        expires_at=issued + timedelta(hours=1),
    )
    expired = verify_promotion_attestations(
        bundle_path=bundle_path,
        trust_store_path=trust_path,
        expected_subject=_subject(),
    )
    assert expired.passed is False
    assert sum("expired" in issue for issue in expired.issues) == 4

    trust.keys[0].revoked = True
    trust_path.write_text(trust.model_dump_json(indent=2), encoding="utf-8")
    revoked = verify_promotion_attestations(
        bundle_path=bundle_path,
        trust_store_path=trust_path,
        expected_subject=_subject(),
    )
    assert revoked.passed is False
    assert sum("signing key is revoked" in issue for issue in revoked.issues) == 2

    attestations_json = json.dumps(bundle.model_dump(mode="json")["attestations"])
    bundle_path.write_text(
        '{"schema_version":1,"schema_version":1,"attestations":'
        + attestations_json
        + "}",
        encoding="utf-8",
    )
    duplicate = verify_promotion_attestations(
        bundle_path=bundle_path,
        trust_store_path=trust_path,
        expected_subject=_subject(),
    )
    assert duplicate.passed is False
    assert duplicate.issues[0].startswith("invalid attestation input: duplicate JSON key")


def test_attestation_verification_fails_closed_without_policy_inputs() -> None:
    result = verify_promotion_attestations(
        bundle_path=None,
        trust_store_path=None,
        expected_subject=_subject(),
    )

    assert result.passed is False
    assert result.verified_kinds == []
    assert result.bundle_sha256 is None


def test_attestation_verification_rejects_incomplete_bundle_and_unpinned_runner(
    tmp_path: Path,
) -> None:
    bundle_path, trust_path, bundle, trust = _signed_inputs(tmp_path)
    trust.keys[0].runner_source_sha256 = "2" * 64
    trust_path.write_text(trust.model_dump_json(indent=2), encoding="utf-8")

    wrong_runner = verify_promotion_attestations(
        bundle_path=bundle_path,
        trust_store_path=trust_path,
        expected_subject=_subject(),
    )
    assert wrong_runner.passed is False
    assert sum("runner source does not match" in issue for issue in wrong_runner.issues) == 2

    bundle_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "attestations": bundle.model_dump(mode="json")["attestations"][:3],
            }
        ),
        encoding="utf-8",
    )
    incomplete = verify_promotion_attestations(
        bundle_path=bundle_path,
        trust_store_path=trust_path,
        expected_subject=_subject(),
    )
    assert incomplete.passed is False
    assert incomplete.issues[0].startswith("invalid attestation input:")


def test_trust_store_rejects_one_key_masquerading_as_two_issuers(tmp_path: Path) -> None:
    bundle_path, trust_path, _, trust = _signed_inputs(tmp_path)
    trust_payload = trust.model_dump(mode="json")
    trust_payload["keys"][1]["public_key_base64"] = trust_payload["keys"][0][
        "public_key_base64"
    ]
    trust_path.write_text(json.dumps(trust_payload), encoding="utf-8")

    result = verify_promotion_attestations(
        bundle_path=bundle_path,
        trust_store_path=trust_path,
        expected_subject=_subject(),
    )

    assert result.passed is False
    assert "distinct public key" in result.issues[0]
