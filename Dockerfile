# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Must match SPARSE_MODEL at runtime, or FastEmbed downloads the other model on first use.
ARG SPARSE_MODEL=Qdrant/bm25

COPY --from=ghcr.io/astral-sh/uv:0.12.24 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH \
    FASTEMBED_CACHE_DIR=/opt/fastembed \
    DATA_DIR=/app/data \
    SNAPSHOT_DIR=/app/snapshots \
    SPARSE_MODEL=${SPARSE_MODEL}

WORKDIR /app

# Dependencies first so code changes don't invalidate this layer.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY ingestion ./ingestion
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Bake the BM25 model into the image: no Hugging Face download at runtime.
RUN python -c "import os; from fastembed import SparseTextEmbedding; SparseTextEmbedding(os.environ['SPARSE_MODEL'], cache_dir='/opt/fastembed')"

# Fixture sample for keyless smoke runs (see docker-compose.smoke.yml).
COPY tests/fixtures/books_sample.csv /app/fixtures/books_sample.csv

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/data /app/snapshots \
    && chown -R app /app/data /app/snapshots /opt/fastembed
USER app

# Used by the optional MCP server (`bookembed-mcp` with MCP_TRANSPORT=http).
EXPOSE 8000

ENTRYPOINT ["bookembed"]
CMD ["ingest"]
