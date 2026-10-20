"""MCP server exposing the book hybrid search as tools for LLM agents (`bookembed-mcp`).

Tools are thin wrappers over `ingestion.search.hybrid_search`. The Qdrant client and the
dense/sparse embedders are built from `Settings` on first use (not at import or startup), so
the server can start and list its tools even while Qdrant or Nebius is unreachable.
"""

# No `from __future__ import annotations`: FastMCP resolves tool signatures at runtime, and
# `search_books` closes over `max_limit` in its annotations, which string annotations would hide.

import logging
import sys
import threading
from collections.abc import Callable
from typing import Annotated, Any, Self, TypeVar

import anyio
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from langchain_core.embeddings import Embeddings
from langchain_qdrant.sparse_embeddings import SparseEmbeddings
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from ingestion.config import McpTransport, Settings, get_settings
from ingestion.data import point_id_for
from ingestion.embeddings import build_dense_embeddings, build_sparse_embeddings
from ingestion.search import SearchFilters, SearchHit, hybrid_search
from ingestion.store import METADATA_KEY, connect

logger = logging.getLogger(__name__)

T = TypeVar("T")

SERVER_NAME = "bookembed"
INSTRUCTIONS = (
    "Search a catalogue of ~7k books (title, authors, categories, description, year, rating, "
    "page count). Use `search_books` with a natural-language query plus optional exact-match "
    "filters; use `get_book` to fetch one book by ISBN-13; use `collection_info` to check the "
    "index is populated."
)


class Book(BaseModel):
    """A book as stored in the index."""

    isbn13: str | None = None
    isbn10: str | None = None
    title: str | None = None
    subtitle: str | None = None
    authors: list[str] = []
    categories: list[str] = []
    published_year: int | None = None
    average_rating: float | None = None
    num_pages: int | None = None
    ratings_count: int | None = None
    description: str | None = None
    thumbnail: str | None = None

    @classmethod
    def from_payload(cls, metadata: dict[str, Any]) -> Self:
        # `authors`/`categories` may be null in older payloads; normalise to lists.
        data = {k: v for k, v in metadata.items() if k in cls.model_fields and v is not None}
        return cls.model_validate(data)


class BookMatch(Book):
    """A search hit: the book plus its fused relevance score (higher is better)."""

    score: float

    @classmethod
    def from_hit(cls, hit: SearchHit) -> Self:
        return cls(score=hit.score, **Book.from_payload(hit.book).model_dump())


class CollectionInfo(BaseModel):
    """Status of the Qdrant collection backing the search tools."""

    collection: str
    exists: bool
    points_count: int | None = None


class _Backend:
    """Holds the Qdrant client and embedders, building any that weren't injected exactly once."""

    def __init__(
        self,
        settings: Settings,
        client: QdrantClient | None,
        dense: Embeddings | None,
        sparse: SparseEmbeddings | None,
    ) -> None:
        self.settings = settings
        self._client = client
        self._dense = dense
        self._sparse = sparse
        # Tools run in worker threads; without the lock two first calls could both build.
        self._lock = threading.Lock()

    @property
    def collection(self) -> str:
        return self.settings.qdrant_collection

    def client(self) -> QdrantClient:
        with self._lock:
            if self._client is None:
                self._client = connect(self.settings)
            return self._client

    def dense(self) -> Embeddings:
        with self._lock:
            if self._dense is None:
                self._dense = build_dense_embeddings(self.settings)
            return self._dense

    def sparse(self) -> SparseEmbeddings:
        with self._lock:
            if self._sparse is None:
                self._sparse = build_sparse_embeddings(self.settings)
            return self._sparse

    def collection_exists(self) -> bool:
        try:
            return self.client().collection_exists(self.collection)
        except Exception as exc:
            logger.warning("Qdrant check failed: %r", exc)
            raise ToolError(
                f"Cannot reach Qdrant at {self.settings.qdrant_url}; is it running?"
            ) from exc

    def require_collection(self) -> QdrantClient:
        if not self.collection_exists():
            raise ToolError(
                f"Collection {self.collection!r} does not exist yet. "
                "Ingest the dataset first (`bookembed ingest`)."
            )
        return self.client()


async def _in_thread(func: Callable[[], T]) -> T:
    # Qdrant and embedding calls are blocking; keep them off the event loop.
    return await anyio.to_thread.run_sync(func)


def create_server(
    settings: Settings,
    client: QdrantClient | None = None,
    dense: Embeddings | None = None,
    sparse: SparseEmbeddings | None = None,
) -> FastMCP[None]:
    """Build the MCP server. Dependencies not passed in are built from `settings` lazily."""
    backend = _Backend(settings, client, dense, sparse)
    max_limit = settings.mcp_max_limit
    mcp: FastMCP[None] = FastMCP(SERVER_NAME, instructions=INSTRUCTIONS)

    @mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
    async def search_books(
        query: Annotated[
            str,
            Field(
                min_length=1,
                description="What the reader wants, in natural language "
                "(themes, plot, mood, title or author words).",
            ),
        ],
        authors: Annotated[
            list[str] | None,
            Field(description="Only books by at least one of these authors (exact full names)."),
        ] = None,
        categories: Annotated[
            list[str] | None,
            Field(
                description="Only books in at least one of these categories "
                "(exact, case-sensitive, e.g. 'Fiction', 'Science fiction', 'History')."
            ),
        ] = None,
        min_year: Annotated[
            int | None, Field(description="Earliest publication year (inclusive).")
        ] = None,
        max_year: Annotated[
            int | None, Field(description="Latest publication year (inclusive).")
        ] = None,
        min_rating: Annotated[
            float | None,
            Field(ge=0, le=5, description="Minimum average reader rating, 0 to 5 (inclusive)."),
        ] = None,
        max_pages: Annotated[
            int | None, Field(gt=0, description="Maximum page count (inclusive).")
        ] = None,
        limit: Annotated[
            int, Field(ge=1, le=max_limit, description="Number of results to return.")
        ] = 5,
    ) -> list[BookMatch]:
        """Find books by meaning and keywords, optionally filtered by metadata.

        Combines semantic (dense) and keyword (BM25) retrieval with rank fusion, so both
        descriptive queries ("space opera with a strong female lead") and exact title or
        author words work. All filters are optional and combine with AND; values inside
        `authors` or `categories` combine with OR. Books with an unknown year, rating or
        page count are excluded by the corresponding filter.

        Returns up to `limit` books, best match first, each with a relevance `score`.
        """
        filters = SearchFilters(
            authors=authors or [],
            categories=categories or [],
            min_year=min_year,
            max_year=max_year,
            min_rating=min_rating,
            max_pages=max_pages,
        )

        def run() -> list[BookMatch]:
            qdrant = backend.require_collection()
            hits = hybrid_search(
                qdrant,
                backend.collection,
                query,
                backend.dense(),
                backend.sparse(),
                filters=filters,
                limit=limit,
            )
            return [BookMatch.from_hit(hit) for hit in hits]

        return await _in_thread(run)

    @mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
    async def get_book(
        isbn13: Annotated[
            str,
            Field(
                pattern=r"^[0-9]{13}$",
                description="The book's 13-digit ISBN, digits only (as returned by search_books).",
            ),
        ],
    ) -> Book:
        """Fetch one book's full record by its ISBN-13."""

        def run() -> Book:
            qdrant = backend.require_collection()
            points = qdrant.retrieve(backend.collection, [point_id_for(isbn13)], with_payload=True)
            if not points:
                raise ToolError(f"No book with ISBN-13 {isbn13} in {backend.collection!r}.")
            return Book.from_payload((points[0].payload or {}).get(METADATA_KEY, {}))

        return await _in_thread(run)

    @mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
    async def collection_info() -> CollectionInfo:
        """Report the searched collection's name, whether it exists, and how many books it holds.

        Use this to check the index is ready before searching; `points_count` is null when the
        collection does not exist.
        """

        def run() -> CollectionInfo:
            name = backend.collection
            if not backend.collection_exists():
                return CollectionInfo(collection=name, exists=False)
            count = backend.client().count(name, exact=True).count
            return CollectionInfo(collection=name, exists=True, points_count=count)

        return await _in_thread(run)

    return mcp


def main() -> None:
    """Entry point for `bookembed-mcp`: serve over stdio (default) or HTTP, per `Settings`."""
    # stdout carries the MCP protocol on stdio, so logs must go to stderr.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = get_settings()
    server = create_server(settings)
    if settings.mcp_transport is McpTransport.HTTP:
        server.run(
            transport="http", host=settings.mcp_host, port=settings.mcp_port, show_banner=False
        )
    else:
        server.run(transport="stdio", show_banner=False)
