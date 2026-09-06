import pytest
import torch

from olympus.models.atlas import (
    AtlasDocument,
    AuthorizedQuery,
    EvidenceSet,
    OlympusAtlasRetriever,
    OlympusAtlasService,
)


def _document(
    identifier: str,
    text: str,
    *,
    version: str = "1",
    acl: tuple[str, ...] = (),
) -> AtlasDocument:
    return AtlasDocument(
        document_id=identifier,
        text=text,
        source_uri=f"file:///{identifier}.txt",
        source_version=version,
        acquired_at="2026-09-06T00:00:00Z",
        license_id="test-only",
        acl_labels=acl,
    )


def test_atlas_hybrid_retrieval_returns_relevant_passage_with_provenance() -> None:
    service = OlympusAtlasService()
    version = service.build_index(
        [
            _document("orchards", "Apple trees need winter chill before spring blossoms."),
            _document("oceans", "Deep ocean vents support unusual ecosystems."),
        ],
        passage_words=20,
        overlap_words=4,
    )

    result = service.retrieve(
        AuthorizedQuery("apple trees winter chill"), corpus_version=version, top_k=1
    )

    assert isinstance(result, EvidenceSet)
    assert result.corpus_version == version
    assert result.hits[0].passage.document_id == "orchards"
    assert result.hits[0].lexical_score > 0
    assert result.hits[0].late_interaction_score > 0
    passage = result.hits[0].passage
    index = service.get_index(version)
    assert index is not None
    assert passage.text == next(
        item.text for item in index.passages if item.document_id == "orchards"
    )


def test_atlas_filters_acl_before_ranking_and_preserves_old_versions() -> None:
    service = OlympusAtlasService()
    first_version = service.build_index(
        [
            _document("public", "Public gardening notes mention roses."),
            _document(
                "secret",
                "Zephyr launch telemetry uses cobalt calibration.",
                acl=("flight",),
            ),
        ]
    )

    public_result = service.retrieve(
        AuthorizedQuery("zephyr cobalt telemetry"), corpus_version=first_version, top_k=1
    )
    assert not isinstance(public_result, EvidenceSet) or all(
        hit.passage.document_id != "secret" for hit in public_result.hits
    )

    private_result = service.retrieve(
        AuthorizedQuery("zephyr cobalt telemetry", frozenset({"flight"})),
        corpus_version=first_version,
        top_k=1,
    )
    assert isinstance(private_result, EvidenceSet)
    assert private_result.hits[0].passage.document_id == "secret"

    public_index = service.get_index(first_version)
    assert public_index is not None
    assert all(passage.document_id != "secret" for passage in public_index.passages)
    private_index = service.get_index(first_version, granted_acl=frozenset({"flight"}))
    assert private_index is not None
    assert any(passage.document_id == "secret" for passage in private_index.passages)

    second_version = service.build_index([_document("new", "A replacement corpus about glaciers.")])
    assert second_version != first_version
    replay = service.retrieve(
        AuthorizedQuery("zephyr cobalt telemetry", frozenset({"flight"})),
        corpus_version=first_version,
        top_k=1,
    )
    assert replay == private_result


def test_atlas_trainable_retriever_has_real_loss_and_parameter_update() -> None:
    torch.manual_seed(7)
    model = OlympusAtlasRetriever(vocab_size=48, hidden_size=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    before = model.projection.weight.detach().clone()

    output = model(
        torch.tensor([[1, 2, 3, 0], [7, 8, 0, 0]]),
        torch.tensor(
            [
                [[1, 2, 4, 0], [9, 10, 11, 0]],
                [[12, 13, 0, 0], [7, 8, 14, 0]],
            ]
        ),
        targets=torch.tensor([0, 1]),
    )
    loss = output["loss"]
    assert torch.isfinite(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert output["scores"].shape == (2, 2)
    assert not torch.equal(before, model.projection.weight.detach())


def test_atlas_rejects_duplicate_identity_and_canonicalizes_document_order() -> None:
    first = _document("a", "Alpha evidence.")
    second = _document("b", "Beta evidence.")
    service = OlympusAtlasService()
    forward = service.build_index([first, second])
    reverse = service.build_index([second, first])
    assert forward == reverse
    assert OlympusAtlasService().build_index([first, second]) != forward

    with pytest.raises(ValueError, match="must be unique"):
        service.build_index([first, first])

    delimiter_collision = service.build_index(
        [
            _document("a:b", "Same content.", version="c"),
            _document("a", "Same content.", version="b:c"),
        ]
    )
    index = service.get_index(delimiter_collision)
    assert index is not None
    assert len({passage.passage_id for passage in index.passages}) == 2


def test_atlas_rejects_all_padding_and_invalid_target_passages() -> None:
    model = OlympusAtlasRetriever(vocab_size=16, hidden_size=8)
    query = torch.tensor([[1, 2]])
    passages = torch.zeros((1, 2, 3), dtype=torch.long)
    with pytest.raises(ValueError, match="valid passage"):
        model(query, passages, targets=torch.tensor([0]))

    passages[0, 0, 0] = 3
    with pytest.raises(ValueError, match="identify valid"):
        model(query, passages, targets=torch.tensor([1]))
    with pytest.raises(ValueError, match="integer tensor"):
        model(query, passages, targets=torch.tensor([0.9]))
    with pytest.raises(ValueError, match="temperature"):
        OlympusAtlasRetriever(vocab_size=16, hidden_size=8, temperature=float("nan"))
