from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from qdrant_client import QdrantClient, models

from ingestion import pipeline
from ingestion.config import Settings
from ingestion.data import Book, load_books
from ingestion.search import SearchFilters, format_hit, hybrid_search
from ingestion.store import DENSE_VECTOR, ensure_collection, wait_for_qdrant
from tests.conftest import FIXTURE_CSV, HashingSparse

COLLECTION = "test_books"


def _books() -> list[Book]:
    return [
        Book(
            isbn13="1",
            title="Starship Captain",
            authors=["Jane Doe"],
            categories=["Fiction"],
            description="A space opera with a strong female lead commanding a starship.",
            published_year=2010,
            average_rating=4.3,
            num_pages=400,
        ),
        Book(
            isbn13="2",
            title="Old Space Tales",
            authors=["John Roe"],
            categories=["Fiction"],
            description="Classic space opera adventures.",
            published_year=1975,
            average_rating=4.5,
            num_pages=200,
        ),
        Book(
            isbn13="3",
            title="Gardening Basics",
            authors=["Jane Doe", "Sam Poe"],
            categories=["Gardening"],
            description="How to grow tomatoes.",
            published_year=2015,
            average_rating=3.2,
            num_pages=120,
        ),
    ]


@pytest.fixture
def loaded(
    memory_client: QdrantClient, dense: DeterministicFakeEmbedding, sparse: HashingSparse
) -> QdrantClient:
    pipeline.ingest_books(_books(), memory_client, COLLECTION, dense, sparse, batch_size=2)
    return memory_client


def test_ingest_creates_hybrid_collection(loaded: QdrantClient) -> None:
    info = loaded.get_collection(COLLECTION)
    vectors = info.config.params.vectors
    assert isinstance(vectors, dict)
    assert vectors[DENSE_VECTOR].size == 32
    assert info.config.params.sparse_vectors is not None
    assert "sparse" in info.config.params.sparse_vectors
    assert loaded.count(COLLECTION).count == 3


def test_reingest_is_idempotent(
    loaded: QdrantClient, dense: DeterministicFakeEmbedding, sparse: HashingSparse
) -> None:
    pipeline.ingest_books(_books(), loaded, COLLECTION, dense, sparse)
    assert loaded.count(COLLECTION).count == 3


def test_payload_is_stored_under_metadata(loaded: QdrantClient) -> None:
    points, _ = loaded.scroll(COLLECTION, limit=10, with_payload=True)
    payload = next(
        p.payload for p in points if p.payload and p.payload["metadata"]["isbn13"] == "3"
    )
    assert payload is not None
    assert payload["metadata"]["authors"] == ["Jane Doe", "Sam Poe"]
    assert payload["metadata"]["num_pages"] == 120
    assert payload["page_content"].startswith("Title: Gardening Basics")


def test_dimension_mismatch_is_rejected(loaded: QdrantClient) -> None:
    with pytest.raises(ValueError, match="dense dim 32, model produces 64"):
        ensure_collection(loaded, COLLECTION, 64)


def test_recreate_drops_existing(loaded: QdrantClient) -> None:
    ensure_collection(loaded, COLLECTION, 64, recreate=True)
    assert loaded.count(COLLECTION).count == 0


def test_hybrid_search_with_filters(
    loaded: QdrantClient, dense: DeterministicFakeEmbedding, sparse: HashingSparse
) -> None:
    hits = hybrid_search(
        loaded,
        COLLECTION,
        "space opera with strong female lead",
        dense,
        sparse,
        filters=SearchFilters(min_year=2001, min_rating=4.0),
    )
    assert [h.book["isbn13"] for h in hits] == ["1"]
    assert "Starship Captain" in format_hit(1, hits[0])


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        (SearchFilters(), {"1", "2", "3"}),
        (SearchFilters(authors=["Jane Doe"]), {"1", "3"}),
        (SearchFilters(categories=["Gardening"]), {"3"}),
        (SearchFilters(max_year=2000), {"2"}),
        (SearchFilters(max_pages=250), {"2", "3"}),
        (SearchFilters(authors=["Nobody"]), set()),
    ],
)
def test_filters(
    loaded: QdrantClient,
    dense: DeterministicFakeEmbedding,
    sparse: HashingSparse,
    filters: SearchFilters,
    expected: set[str],
) -> None:
    hits = hybrid_search(loaded, COLLECTION, "book", dense, sparse, filters=filters, limit=10)
    assert {h.book["isbn13"] for h in hits} == expected


def test_sparse_branch_ranks_keyword_match_first(
    loaded: QdrantClient, sparse: HashingSparse
) -> None:
    # Fake dense vectors are random, so a lexical query should be won by BM25-style matching.
    hits = hybrid_search(
        loaded, COLLECTION, "tomatoes", DeterministicFakeEmbedding(size=32), sparse, limit=1
    )
    assert hits[0].book["title"] == "Gardening Basics"


def test_empty_filter_is_none() -> None:
    assert SearchFilters().to_qdrant() is None
    condition = SearchFilters(min_rating=4).to_qdrant()
    assert isinstance(condition, models.Filter)


def test_format_hit_handles_missing_metadata() -> None:
    from ingestion.search import SearchHit

    line = format_hit(3, SearchHit(score=0.5, book={"title": "X"}, text=""))
    assert "unknown author" in line
    assert "uncategorised" in line


def test_run_end_to_end_on_fixture(
    settings: Settings, memory_client: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline, "connect", lambda s: memory_client)
    monkeypatch.setattr(pipeline, "build_sparse_embeddings", lambda s: HashingSparse())
    count = pipeline.run(settings, csv_path=FIXTURE_CSV)
    assert count == len(load_books(FIXTURE_CSV))
    assert memory_client.count(settings.qdrant_collection).count == count


def test_run_rejects_empty_csv(settings: Settings, tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("isbn13,title\n")
    with pytest.raises(ValueError, match="No valid books"):
        pipeline.run(settings, csv_path=empty)


def test_wait_for_qdrant_times_out() -> None:
    class Down:
        def get_collections(self) -> None:
            raise ConnectionError("refused")

    with pytest.raises(TimeoutError):
        wait_for_qdrant(Down(), timeout_seconds=0, interval=0)  # type: ignore[arg-type]


def test_wait_for_qdrant_retries_until_up() -> None:
    class Flaky:
        calls = 0

        def get_collections(self) -> None:
            self.calls += 1
            if self.calls < 3:
                raise ConnectionError("refused")

    client = Flaky()
    wait_for_qdrant(client, timeout_seconds=10, interval=0)  # type: ignore[arg-type]
    assert client.calls == 3
