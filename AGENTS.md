# AGENTS.md

`bookembed` ingests a books CSV into Qdrant with Nebius dense and FastEmbed BM25 sparse vectors for hybrid search, and exposes it through a CLI and an MCP server.

## Commands
- Install: `uv sync` (creates `.venv` from `uv.lock`)
- Hooks: `uv run pre-commit install` (mirror CI)
- CLI: `uv run bookembed --help` (subcommands include `download`, `ingest`, `search`, `snapshot`)
- MCP server: `uv run bookembed-mcp`
- Lint: `uv run ruff check .` and `uv run ruff format --check .`
- Types: `uv run mypy ingestion`
- Unit tests: `uv run pytest -m "not integration" --cov --cov-fail-under=70`
- Integration tests: need Qdrant at `QDRANT_URL`: `QDRANT_URL=http://localhost:6333 uv run pytest`
- Keyless smoke stack: `./scripts/smoke_test.sh`
- Full stack: `docker compose up`

## Conventions
- Python 3.11+ (`.python-version`), uv with `uv_build`, package `ingestion/`. Ruff line length 100, mypy strict.
- `ingestion/`: `cli.py`, `config.py` (pydantic-settings), `data.py`, `download.py`, `embeddings.py`, `pipeline.py`, `search.py`, `snapshot.py`, `store.py`, `mcp_server.py`.
- `tests/`: one `test_*.py` per module, `tests/integration/` for the real Qdrant, `tests/fixtures/books_sample.csv` as the small dataset.
- Config comes from env vars or `.env`, defined in `ingestion/config.py` and listed in `.env.example` (e.g. `NEBIUS_API_KEY`, `QDRANT_URL`, `EMBEDDING_PROVIDER`).
- Use `EMBEDDING_PROVIDER=fake` for offline tests; unit tests mock Nebius and must not call the real API.
- CI (`.github/workflows/ci.yml`) runs ruff, mypy and pytest with coverage of at least 70%.
- `.claude/agents/` holds `builder.md` and `qa-validator.md` subagent definitions.

## Boundaries
- Always: add or update tests under `tests/` with every behavior change, then run `uv run ruff check .`, `uv run mypy ingestion` and the unit tests above.
- Ask first: adding a dependency, changing the Qdrant collection schema or payload indexes, changing the embedding model default (needs `--recreate`).
- Never: commit secrets or read `.env` files.
- Never: call the real Nebius API from tests; mock it or set `EMBEDDING_PROVIDER=fake`.
