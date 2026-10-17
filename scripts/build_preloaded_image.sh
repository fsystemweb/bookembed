#!/usr/bin/env bash
# Snapshot the populated `books` collection and bake it into a Qdrant image.
# Prereq: `docker compose up` has finished ingesting.
set -euo pipefail
cd "$(dirname "$0")/.."

tag="${1:-bookembed-qdrant:preloaded}"
mkdir -p snapshots
docker compose run --rm --user "$(id -u):$(id -g)" ingestion snapshot --out /app/snapshots/books.snapshot
docker build -f Dockerfile.qdrant -t "$tag" snapshots/
echo "Built $tag. Run it with: docker run -p 6333:6333 $tag"
