from __future__ import annotations

import re
import zlib
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_qdrant.sparse_embeddings import SparseEmbeddings, SparseVector
from qdrant_client import QdrantClient

from ingestion.config import EmbeddingProvider, Settings

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "books_sample.csv"


class HashingSparse(SparseEmbeddings):
    """Offline stand-in for BM25: lower-cased word counts hashed into indices."""

    def _embed(self, text: str) -> SparseVector:
        counts = Counter(re.findall(r"[a-z0-9]+", text.lower()))
        merged: dict[int, float] = {}
        for token, count in counts.items():
            index = zlib.crc32(token.encode()) & 0x7FFFFFFF
            merged[index] = merged.get(index, 0.0) + float(count)
        return SparseVector(indices=list(merged), values=list(merged.values()))

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> SparseVector:
        return self._embed(text)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Never pick up a developer's real .env or credentials in unit tests.
    monkeypatch.chdir(tmp_path)
    for var in ("NEBIUS_API_KEY", "KAGGLE_USERNAME", "KAGGLE_KEY", "QDRANT_API_KEY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        embedding_provider=EmbeddingProvider.FAKE,
        fake_embedding_dim=32,
        data_dir=tmp_path / "data",
        qdrant_collection="test_books",
        qdrant_wait_seconds=0,
    )


@pytest.fixture
def dense() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=32)


@pytest.fixture
def sparse() -> HashingSparse:
    return HashingSparse()


@pytest.fixture
def memory_client() -> Iterator[QdrantClient]:
    client = QdrantClient(location=":memory:")
    yield client
    client.close()
