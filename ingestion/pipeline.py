"""End-to-end ingestion: CSV -> clean records -> dense + sparse vectors -> Qdrant upsert."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from itertools import islice
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_qdrant.sparse_embeddings import SparseEmbeddings
from qdrant_client import QdrantClient

from ingestion.config import Settings
from ingestion.data import Book, load_books
from ingestion.download import resolve_csv
from ingestion.embeddings import build_dense_embeddings, build_sparse_embeddings
from ingestion.store import (
    CONTENT_KEY,
    DENSE_VECTOR,
    METADATA_KEY,
    SPARSE_VECTOR,
    connect,
    ensure_collection,
    wait_for_qdrant,
)

logger = logging.getLogger(__name__)


def batched(items: Sequence[Book], size: int) -> Iterator[list[Book]]:
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


def build_vector_store(
    client: QdrantClient,
    collection: str,
    dense: Embeddings,
    sparse: SparseEmbeddings,
) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=dense,
        sparse_embedding=sparse,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=DENSE_VECTOR,
        sparse_vector_name=SPARSE_VECTOR,
        content_payload_key=CONTENT_KEY,
        metadata_payload_key=METADATA_KEY,
    )


def ingest_books(
    books: Sequence[Book],
    client: QdrantClient,
    collection: str,
    dense: Embeddings,
    sparse: SparseEmbeddings,
    *,
    batch_size: int = 64,
    recreate: bool = False,
) -> int:
    """Upsert `books` into `collection`. Re-running is idempotent: IDs derive from ISBN13."""
    dense_dim = len(dense.embed_query("dimension probe"))
    ensure_collection(client, collection, dense_dim, recreate=recreate)
    store = build_vector_store(client, collection, dense, sparse)

    done = 0
    for batch in batched(books, batch_size):
        store.add_texts(
            texts=[book.embedding_text() for book in batch],
            metadatas=[book.payload() for book in batch],
            ids=[book.point_id for book in batch],
            batch_size=batch_size,
        )
        done += len(batch)
        logger.info("Upserted %d/%d books", done, len(books))
    return done


def run(settings: Settings, *, csv_path: Path | None = None, recreate: bool = False) -> int:
    path = csv_path or resolve_csv(settings)
    books = load_books(path)
    if not books:
        raise ValueError(f"No valid books found in {path}")

    client = connect(settings)
    wait_for_qdrant(client, settings.qdrant_wait_seconds)
    count = ingest_books(
        books,
        client,
        settings.qdrant_collection,
        build_dense_embeddings(settings),
        build_sparse_embeddings(settings),
        batch_size=settings.embedding_batch_size,
        recreate=recreate,
    )
    total = client.count(settings.qdrant_collection, exact=True).count
    logger.info(
        "Ingested %d books; collection %r holds %d points", count, settings.qdrant_collection, total
    )
    return count
