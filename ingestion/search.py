"""Hybrid search: dense + BM25 prefetch fused with RRF, under optional payload filters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_qdrant.sparse_embeddings import SparseEmbeddings
from qdrant_client import QdrantClient, models

from ingestion.store import CONTENT_KEY, DENSE_VECTOR, METADATA_KEY, SPARSE_VECTOR, payload_field


@dataclass(frozen=True)
class SearchFilters:
    authors: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    min_year: int | None = None
    max_year: int | None = None
    min_rating: float | None = None
    max_pages: int | None = None

    def to_qdrant(self) -> models.Filter | None:
        must: list[models.Condition] = []
        if self.authors:
            must.append(_match_any("authors", self.authors))
        if self.categories:
            must.append(_match_any("categories", self.categories))
        if self.min_year is not None or self.max_year is not None:
            must.append(_range("published_year", gte=self.min_year, lte=self.max_year))
        if self.min_rating is not None:
            must.append(_range("average_rating", gte=self.min_rating))
        if self.max_pages is not None:
            must.append(_range("num_pages", lte=self.max_pages))
        return models.Filter(must=must) if must else None


def _match_any(name: str, values: list[str]) -> models.FieldCondition:
    return models.FieldCondition(key=payload_field(name), match=models.MatchAny(any=values))


def _range(name: str, gte: float | None = None, lte: float | None = None) -> models.FieldCondition:
    return models.FieldCondition(key=payload_field(name), range=models.Range(gte=gte, lte=lte))


@dataclass(frozen=True)
class SearchHit:
    score: float
    book: dict[str, Any]
    text: str


def hybrid_search(
    client: QdrantClient,
    collection: str,
    query: str,
    dense: Embeddings,
    sparse: SparseEmbeddings,
    *,
    filters: SearchFilters | None = None,
    limit: int = 5,
    prefetch_limit: int | None = None,
) -> list[SearchHit]:
    """Run dense and sparse retrieval under the same filter and fuse with RRF.

    Each branch over-fetches (`prefetch_limit`, default 4x `limit`) so RRF has
    enough candidates from both rankings to reorder.
    """
    query_filter = (filters or SearchFilters()).to_qdrant()
    candidates = prefetch_limit or max(limit * 4, 20)
    sparse_vector = sparse.embed_query(query)
    response = client.query_points(
        collection_name=collection,
        prefetch=[
            models.Prefetch(
                query=dense.embed_query(query),
                using=DENSE_VECTOR,
                filter=query_filter,
                limit=candidates,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vector.indices, values=sparse_vector.values
                ),
                using=SPARSE_VECTOR,
                filter=query_filter,
                limit=candidates,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )
    return [
        SearchHit(
            score=point.score,
            book=(point.payload or {}).get(METADATA_KEY, {}),
            text=(point.payload or {}).get(CONTENT_KEY, ""),
        )
        for point in response.points
    ]


def format_hit(rank: int, hit: SearchHit) -> str:
    book = hit.book
    authors = ", ".join(book.get("authors") or []) or "unknown author"
    meta = [
        str(book.get("published_year") or "?"),
        f"rating {book.get('average_rating') or '?'}",
        f"{book.get('num_pages') or '?'} pages",
        ", ".join(book.get("categories") or []) or "uncategorised",
    ]
    return f"{rank:>2}. [{hit.score:.4f}] {book.get('title')} - {authors} ({' | '.join(meta)})"
