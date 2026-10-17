"""Runs against a real Qdrant server (CI uses a service container) with real BM25."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from qdrant_client import QdrantClient

from ingestion.config import EmbeddingProvider, Settings
from ingestion.data import load_books
from ingestion.embeddings import build_sparse_embeddings
from ingestion.pipeline import ingest_books
from ingestion.search import SearchFilters, hybrid_search
from ingestion.snapshot import create_snapshot
from ingestion.store import payload_field
from tests.conftest import FIXTURE_CSV

pytestmark = pytest.mark.integration

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


@pytest.fixture(scope="module")
def server() -> Iterator[QdrantClient]:
    client = QdrantClient(url=QDRANT_URL, timeout=10)
    try:
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"no Qdrant server at {QDRANT_URL}: {exc}")
    yield client
    client.close()


@pytest.fixture(scope="module")
def collection(server: QdrantClient) -> Iterator[str]:
    name = f"it_{uuid.uuid4().hex[:8]}"
    settings = Settings(embedding_provider=EmbeddingProvider.FAKE)
    ingest_books(
        load_books(FIXTURE_CSV),
        server,
        name,
        DeterministicFakeEmbedding(size=32),
        build_sparse_embeddings(settings),
        batch_size=16,
    )
    yield name
    server.delete_collection(name)


def test_payload_indexes_exist(server: QdrantClient, collection: str) -> None:
    schema = server.get_collection(collection).payload_schema
    for field in ("authors", "categories"):
        assert schema[payload_field(field)].data_type.value == "keyword"
    for field in ("published_year", "num_pages"):
        assert schema[payload_field(field)].data_type.value == "integer"
    assert schema[payload_field("average_rating")].data_type.value == "float"


def test_demo_query_returns_filtered_results(server: QdrantClient, collection: str) -> None:
    sparse = build_sparse_embeddings(Settings(embedding_provider=EmbeddingProvider.FAKE))
    hits = hybrid_search(
        server,
        collection,
        "space opera with strong female lead",
        DeterministicFakeEmbedding(size=32),
        sparse,
        filters=SearchFilters(min_year=2001, min_rating=4.0),
    )
    assert hits
    for hit in hits:
        assert hit.book["published_year"] >= 2001
        assert hit.book["average_rating"] >= 4.0


def test_bm25_finds_exact_title(server: QdrantClient, collection: str) -> None:
    sparse = build_sparse_embeddings(Settings(embedding_provider=EmbeddingProvider.FAKE))
    hits = hybrid_search(
        server, collection, "The Forever War Haldeman", DeterministicFakeEmbedding(size=32), sparse
    )
    assert any(h.book["title"] == "The Forever War" for h in hits[:3])


def test_snapshot_round_trip(server: QdrantClient, collection: str, tmp_path: Path) -> None:
    settings = Settings(
        embedding_provider=EmbeddingProvider.FAKE,
        qdrant_url=QDRANT_URL,
        qdrant_collection=collection,
    )
    out = create_snapshot(settings, server, tmp_path / "c.snapshot")
    assert out.stat().st_size > 0
