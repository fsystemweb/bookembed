"""Command-line entry point: `bookembed {download,ingest,search,snapshot}`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from ingestion import pipeline
from ingestion.config import Settings, get_settings
from ingestion.download import download_dataset
from ingestion.embeddings import build_dense_embeddings, build_sparse_embeddings
from ingestion.search import SearchFilters, format_hit, hybrid_search
from ingestion.snapshot import create_snapshot
from ingestion.store import connect, wait_for_qdrant

DEMO_QUERY = "space opera with strong female lead"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bookembed", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="download the Kaggle dataset into DATA_DIR")

    ingest = sub.add_parser("ingest", help="load the CSV into Qdrant (idempotent upsert)")
    ingest.add_argument("--csv", type=Path, help="CSV to ingest (default: BOOKS_CSV or download)")
    ingest.add_argument("--recreate", action="store_true", help="drop the collection first")

    search = sub.add_parser("search", help="hybrid search with optional metadata filters")
    search.add_argument("query", nargs="?", help=f"free text (default demo: {DEMO_QUERY!r})")
    search.add_argument("--author", action="append", default=[], help="exact author; repeatable")
    search.add_argument(
        "--category", action="append", default=[], help="exact category; repeatable"
    )
    search.add_argument("--min-year", type=int)
    search.add_argument("--max-year", type=int)
    search.add_argument("--min-rating", type=float)
    search.add_argument("--max-pages", type=int)
    search.add_argument("-k", "--limit", type=int, default=5)
    search.add_argument("--json", action="store_true", help="print results as JSON")
    search.add_argument(
        "--fail-if-empty", action="store_true", help="exit 1 when nothing matches (smoke tests)"
    )

    snap = sub.add_parser("snapshot", help="snapshot the collection to a local file")
    snap.add_argument("--out", type=Path, help="default: SNAPSHOT_DIR/<collection>.snapshot")
    return parser


def _search(settings: Settings, args: argparse.Namespace) -> int:
    if args.query is None:
        # The README demo: "space opera with strong female lead, after 2000, rating >= 4".
        query = DEMO_QUERY
        filters = SearchFilters(min_year=2001, min_rating=4.0)
    else:
        query = args.query
        filters = SearchFilters(
            authors=args.author,
            categories=args.category,
            min_year=args.min_year,
            max_year=args.max_year,
            min_rating=args.min_rating,
            max_pages=args.max_pages,
        )
    client = connect(settings)
    hits = hybrid_search(
        client,
        settings.qdrant_collection,
        query,
        build_dense_embeddings(settings),
        build_sparse_embeddings(settings),
        filters=filters,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps([asdict(hit) for hit in hits], indent=2, ensure_ascii=False))
    else:
        print(f"Query: {query!r}  filters: {filters}")
        for rank, hit in enumerate(hits, start=1):
            print(format_hit(rank, hit))
        if not hits:
            print("No results.")
    return 1 if args.fail_if_empty and not hits else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # Quiet per-request HTTP logs unless debugging.
    logging.getLogger("httpx").setLevel(logging.DEBUG if args.verbose else logging.WARNING)
    settings = get_settings()

    if args.command == "download":
        print(download_dataset(settings))
        return 0
    if args.command == "ingest":
        pipeline.run(settings, csv_path=args.csv, recreate=args.recreate)
        return 0
    if args.command == "search":
        return _search(settings, args)
    if args.command == "snapshot":
        client = connect(settings)
        wait_for_qdrant(client, settings.qdrant_wait_seconds)
        out = args.out or settings.snapshot_dir / f"{settings.qdrant_collection}.snapshot"
        print(create_snapshot(settings, client, out))
        return 0
    raise AssertionError(f"unhandled command {args.command}")  # pragma: no cover
