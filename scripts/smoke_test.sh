#!/usr/bin/env bash
# End-to-end Docker check on the fixture: build, ingest, assert a filtered hybrid search hits.
set -euo pipefail
cd "$(dirname "$0")/.."

compose=(docker compose -p bookembed-smoke -f docker-compose.yml -f docker-compose.smoke.yml)
cleanup() { "${compose[@]}" down -v --remove-orphans >/dev/null 2>&1 || true; }
trap cleanup EXIT

"${compose[@]}" build ingestion
# `up` waits for Qdrant to be healthy, runs ingestion, and returns its exit code.
"${compose[@]}" up --exit-code-from ingestion --abort-on-container-failure ingestion
# Demo query: "space opera with strong female lead", year >= 2001, rating >= 4.
"${compose[@]}" run --rm ingestion search --fail-if-empty
"${compose[@]}" run --rm ingestion search "war" --min-rating 4 --json --fail-if-empty >/dev/null
echo "smoke test passed"
