#!/usr/bin/env bash
# End-to-end Docker check on the fixture: build, ingest, then assert that
#  - every fixture row landed in Qdrant,
#  - the filtered demo search returns hits that all satisfy the filters,
#  - BM25 (the sparse branch) finds an exact title, which fake dense vectors cannot.
set -euo pipefail
cd "$(dirname "$0")/.."

compose=(docker compose -p bookembed-smoke -f docker-compose.yml -f docker-compose.smoke.yml)
cleanup() { "${compose[@]}" down -v --remove-orphans >/dev/null 2>&1 || true; }
trap cleanup EXIT
fail() { echo "smoke test FAILED: $*" >&2; exit 1; }

expected=$(python3 -c "import csv,sys; print(len({r['isbn13'] for r in csv.DictReader(open(sys.argv[1]))}))" \
  tests/fixtures/books_sample.csv)

"${compose[@]}" build ingestion
# `up` waits for Qdrant to be healthy, runs ingestion, and returns its exit code.
"${compose[@]}" up --exit-code-from ingestion --abort-on-container-failure ingestion

count=$("${compose[@]}" exec -T qdrant bash -c \
  "exec 3<>/dev/tcp/127.0.0.1/6333 && printf 'GET /collections/books HTTP/1.0\r\n\r\n' >&3 && cat <&3" \
  | grep -o '"points_count":[0-9]*' | cut -d: -f2)
[[ "$count" == "$expected" ]] || fail "collection has ${count:-?} points, expected $expected"
echo "collection holds $count points"

# Demo query: "space opera with strong female lead", year >= 2001, rating >= 4.
"${compose[@]}" run --rm ingestion search --fail-if-empty
"${compose[@]}" run --rm -T ingestion search "space opera with strong female lead" \
  --min-year 2001 --min-rating 4 --json --fail-if-empty \
  | python3 -c "
import json, sys
hits = json.load(sys.stdin)
bad = [h['book']['title'] for h in hits
       if h['book']['published_year'] < 2001 or h['book']['average_rating'] < 4]
sys.exit(f'filter violated by: {bad}' if bad else 0)
" || fail "demo search returned books outside the filters"

"${compose[@]}" run --rm -T ingestion search "The Forever War Haldeman" -k 3 --json \
  | python3 -c "
import json, sys
titles = [h['book']['title'] for h in json.load(sys.stdin)]
sys.exit(0 if 'The Forever War' in titles else f'BM25 miss, got {titles}')
" || fail "keyword search did not find 'The Forever War'"

echo "smoke test passed"
