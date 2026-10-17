"""Create a Qdrant snapshot of the collection and download it to disk."""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
from qdrant_client import QdrantClient

from ingestion.config import Settings

logger = logging.getLogger(__name__)


def create_snapshot(
    settings: Settings,
    client: QdrantClient,
    out_path: Path,
    http: httpx.Client | None = None,
) -> Path:
    """Snapshot `settings.qdrant_collection` server-side, then stream it to `out_path`.

    The file can be baked into an image and restored with `qdrant --snapshot`,
    skipping re-embedding entirely (see Dockerfile.qdrant).
    """
    collection = settings.qdrant_collection
    description = client.create_snapshot(collection_name=collection, wait=True)
    if description is None:
        raise RuntimeError(f"Qdrant did not return a snapshot for {collection!r}")
    url = f"{settings.qdrant_url.rstrip('/')}/collections/{collection}/snapshots/{description.name}"
    headers = {}
    if settings.qdrant_api_key:
        headers["api-key"] = settings.qdrant_api_key.get_secret_value()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    owns_client = http is None
    session = http or httpx.Client(timeout=settings.qdrant_timeout)
    try:
        with session.stream("GET", url, headers=headers) as response:
            response.raise_for_status()
            with out_path.open("wb") as fh:
                for chunk in response.iter_bytes():
                    fh.write(chunk)
    finally:
        if owns_client:
            session.close()
    logger.info("Wrote snapshot %s (%d bytes)", out_path, out_path.stat().st_size)
    return out_path
