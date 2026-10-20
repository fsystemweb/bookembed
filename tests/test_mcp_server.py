from __future__ import annotations

from typing import Any

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from langchain_core.embeddings import DeterministicFakeEmbedding
from qdrant_client import QdrantClient

from ingestion import mcp_server, pipeline
from ingestion.config import McpTransport, Settings
from ingestion.data import load_books
from tests.conftest import FIXTURE_CSV, HashingSparse

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def loaded(
    settings: Settings,
    memory_client: QdrantClient,
    dense: DeterministicFakeEmbedding,
    sparse: HashingSparse,
) -> QdrantClient:
    books = load_books(FIXTURE_CSV)
    pipeline.ingest_books(
        books, memory_client, settings.qdrant_collection, dense, sparse, batch_size=64
    )
    return memory_client


@pytest.fixture
def server(
    settings: Settings,
    loaded: QdrantClient,
    dense: DeterministicFakeEmbedding,
    sparse: HashingSparse,
) -> FastMCP[None]:
    return mcp_server.create_server(settings, client=loaded, dense=dense, sparse=sparse)


async def _call(server: FastMCP[None], tool: str, args: dict[str, Any] | None = None) -> Any:
    async with Client(server) as client:
        result = await client.call_tool(tool, args or {})
    return result.structured_content


async def _search(server: FastMCP[None], **args: Any) -> list[dict[str, Any]]:
    content = await _call(server, "search_books", args)
    hits: list[dict[str, Any]] = content["result"]
    return hits


async def test_lists_tools_with_descriptions(server: FastMCP[None]) -> None:
    async with Client(server) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
    assert set(tools) == {"search_books", "get_book", "collection_info"}
    for tool in tools.values():
        assert tool.description
    props = tools["search_books"].inputSchema["properties"]
    assert set(props) == {
        "query",
        "authors",
        "categories",
        "min_year",
        "max_year",
        "min_rating",
        "max_pages",
        "limit",
    }
    assert props["limit"]["maximum"] == 50
    assert props["query"]["description"]
    assert tools["search_books"].inputSchema["required"] == ["query"]


async def test_search_without_filters(server: FastMCP[None]) -> None:
    hits = await _search(server, query="The Forever War Haldeman", limit=3)
    assert len(hits) == 3
    # Fake dense vectors are random, so only BM25 guarantees the match makes the top 3.
    match = next(h for h in hits if h["title"] == "The Forever War")
    assert match["isbn13"] == "9780060510862"
    assert match["authors"] == ["Joe Haldeman"]
    assert match["description"].startswith("Drafted into the ranks")
    assert {"categories", "published_year", "average_rating", "num_pages"} <= set(match)
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


async def test_search_default_limit_is_five(server: FastMCP[None]) -> None:
    assert len(await _search(server, query="novel")) == 5


async def test_search_with_filters(server: FastMCP[None]) -> None:
    hits = await _search(
        server,
        query="space opera with strong female lead",
        min_year=2001,
        min_rating=4.0,
        max_pages=600,
        limit=50,
    )
    assert hits
    for hit in hits:
        assert hit["published_year"] >= 2001
        assert hit["average_rating"] >= 4.0
        assert hit["num_pages"] <= 600


async def test_search_with_author_and_category_filters(server: FastMCP[None]) -> None:
    hits = await _search(
        server, query="book", authors=["Victoria Alexander"], categories=["Fiction"], limit=50
    )
    assert {h["title"] for h in hits} == {"What A Lady Wants", "A Little Bit Wicked"}
    assert await _search(server, query="book", authors=["Nobody"], max_year=1900) == []


@pytest.mark.parametrize("limit", [0, 51])
async def test_limit_out_of_bounds_is_rejected(server: FastMCP[None], limit: int) -> None:
    with pytest.raises(ToolError, match="limit"):
        await _search(server, query="war", limit=limit)


async def test_max_limit_comes_from_settings(
    settings: Settings,
    loaded: QdrantClient,
    dense: DeterministicFakeEmbedding,
    sparse: HashingSparse,
) -> None:
    settings.mcp_max_limit = 2
    server = mcp_server.create_server(settings, client=loaded, dense=dense, sparse=sparse)
    assert len(await _search(server, query="war", limit=2)) == 2
    with pytest.raises(ToolError):
        await _search(server, query="war", limit=3)


async def test_get_book(server: FastMCP[None]) -> None:
    book = await _call(server, "get_book", {"isbn13": "9780060510862"})
    assert book["title"] == "The Forever War"
    assert book["isbn10"] == "0060510862"
    assert "score" not in book


async def test_get_book_unknown_and_malformed_isbn(server: FastMCP[None]) -> None:
    with pytest.raises(ToolError, match="No book with ISBN-13 9999999999999"):
        await _call(server, "get_book", {"isbn13": "9999999999999"})
    with pytest.raises(ToolError):
        await _call(server, "get_book", {"isbn13": "978-0060510862"})


async def test_collection_info(server: FastMCP[None], settings: Settings) -> None:
    info = await _call(server, "collection_info")
    assert info == {
        "collection": settings.qdrant_collection,
        "exists": True,
        "points_count": len(load_books(FIXTURE_CSV)),
    }


@pytest.fixture
def empty_server(
    settings: Settings,
    memory_client: QdrantClient,
    dense: DeterministicFakeEmbedding,
    sparse: HashingSparse,
) -> FastMCP[None]:
    return mcp_server.create_server(settings, client=memory_client, dense=dense, sparse=sparse)


async def test_missing_collection_is_a_friendly_error(empty_server: FastMCP[None]) -> None:
    with pytest.raises(ToolError, match=r"'test_books' does not exist yet.*bookembed ingest"):
        await _search(empty_server, query="war")
    with pytest.raises(ToolError, match="does not exist yet"):
        await _call(empty_server, "get_book", {"isbn13": "9780060510862"})
    info = await _call(empty_server, "collection_info")
    assert info == {"collection": "test_books", "exists": False, "points_count": None}


async def test_unreachable_qdrant_is_a_friendly_error(
    settings: Settings, dense: DeterministicFakeEmbedding, sparse: HashingSparse
) -> None:
    class Down:
        def collection_exists(self, name: str) -> bool:
            raise ConnectionError("refused")

    server = mcp_server.create_server(
        settings,
        client=Down(),  # type: ignore[arg-type]
        dense=dense,
        sparse=sparse,
    )
    with pytest.raises(ToolError, match="Cannot reach Qdrant"):
        await _call(server, "collection_info")


async def test_dependencies_are_built_lazily_and_once(
    settings: Settings, loaded: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, int] = {"connect": 0, "sparse": 0}

    def fake_connect(s: Settings) -> QdrantClient:
        calls["connect"] += 1
        return loaded

    def fake_sparse(s: Settings) -> HashingSparse:
        calls["sparse"] += 1
        return HashingSparse()

    monkeypatch.setattr(mcp_server, "connect", fake_connect)
    monkeypatch.setattr(mcp_server, "build_sparse_embeddings", fake_sparse)
    # Dense comes from the real factory: settings use the offline `fake` provider (dim 32).
    server = mcp_server.create_server(settings)
    assert calls == {"connect": 0, "sparse": 0}

    for _ in range(2):
        assert len(await _search(server, query="war", limit=2)) == 2
    assert calls == {"connect": 1, "sparse": 1}


@pytest.mark.parametrize(
    ("transport", "expected"),
    [
        (McpTransport.STDIO, {"transport": "stdio", "show_banner": False}),
        (
            McpTransport.HTTP,
            {"transport": "http", "host": "0.0.0.0", "port": 9001, "show_banner": False},
        ),
    ],
)
def test_main_runs_configured_transport(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    transport: McpTransport,
    expected: dict[str, Any],
) -> None:
    settings.mcp_transport = transport
    settings.mcp_host = "0.0.0.0"
    settings.mcp_port = 9001
    captured: dict[str, Any] = {}
    monkeypatch.setattr(mcp_server, "get_settings", lambda: settings)
    monkeypatch.setattr(FastMCP, "run", lambda self, **kwargs: captured.update(kwargs))
    mcp_server.main()
    assert captured == expected


def test_mcp_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("MCP_TRANSPORT", "http")
    monkeypatch.setenv("MCP_PORT", "8123")
    monkeypatch.setenv("MCP_MAX_LIMIT", "10")
    s = Settings()
    assert s.mcp_transport is McpTransport.HTTP
    assert (s.mcp_host, s.mcp_port, s.mcp_max_limit) == ("127.0.0.1", 8123, 10)
