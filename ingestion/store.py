"""Qdrant collection schema: named dense + sparse vectors and payload indexes."""

from __future__ import annotations

import logging
import time

from qdrant_client import QdrantClient, models

from ingestion.config import Settings

logger = logging.getLogger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
METADATA_KEY = "metadata"
CONTENT_KEY = "page_content"

KEYWORD_FIELDS = ("authors", "categories")
INTEGER_FIELDS = ("published_year", "num_pages", "ratings_count")
FLOAT_FIELDS = ("average_rating",)


def payload_field(name: str) -> str:
    """LangChain nests document metadata under `metadata`; filters must use the full path."""
    return f"{METADATA_KEY}.{name}"


def connect(settings: Settings) -> QdrantClient:
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    return QdrantClient(url=settings.qdrant_url, api_key=api_key, timeout=settings.qdrant_timeout)


def wait_for_qdrant(client: QdrantClient, timeout_seconds: int, interval: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            client.get_collections()
            return
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Qdrant not reachable after {timeout_seconds}s") from exc
            logger.info("Waiting for Qdrant (%s)", exc.__class__.__name__)
            time.sleep(interval)


def ensure_collection(
    client: QdrantClient, name: str, dense_dim: int, *, recreate: bool = False
) -> None:
    """Create the hybrid collection and its payload indexes if missing.

    An existing collection is reused (ingestion upserts into it), unless its dense
    dimension no longer matches the embedding model, which would corrupt search.
    """
    if recreate and client.collection_exists(name):
        logger.info("Dropping existing collection %s", name)
        client.delete_collection(name)

    if client.collection_exists(name):
        vectors = client.get_collection(name).config.params.vectors
        existing = vectors.get(DENSE_VECTOR) if isinstance(vectors, dict) else None
        if existing is None or existing.size != dense_dim:
            found = existing.size if existing else "none"
            raise ValueError(
                f"Collection {name!r} has dense dim {found}, model produces {dense_dim}; "
                "re-run with --recreate"
            )
        logger.info("Reusing collection %s", name)
    else:
        logger.info("Creating collection %s (dense dim %d)", name, dense_dim)
        client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)
            },
            # BM25 vectors carry term frequencies only; Qdrant applies IDF at query time.
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
    create_payload_indexes(client, name)


def create_payload_indexes(client: QdrantClient, name: str) -> None:
    schema: dict[str, models.PayloadSchemaType] = {}
    schema |= dict.fromkeys(KEYWORD_FIELDS, models.PayloadSchemaType.KEYWORD)
    schema |= dict.fromkeys(INTEGER_FIELDS, models.PayloadSchemaType.INTEGER)
    schema |= dict.fromkeys(FLOAT_FIELDS, models.PayloadSchemaType.FLOAT)
    for field, field_schema in schema.items():
        # Idempotent: Qdrant treats re-creating an identical index as a no-op.
        client.create_payload_index(name, payload_field(field), field_schema=field_schema)
