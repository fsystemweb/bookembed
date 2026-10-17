from __future__ import annotations

from typing import Any

import httpx
import openai
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_openai import OpenAIEmbeddings

from ingestion import embeddings as emb
from ingestion.config import EmbeddingProvider, Settings


def _rate_limit() -> openai.RateLimitError:
    request = httpx.Request("POST", "https://nebius.test/v1/embeddings")
    return openai.RateLimitError(
        "slow down", response=httpx.Response(429, request=request), body=None
    )


class Flaky(Embeddings):
    def __init__(self, failures: int, error: Exception | None = None) -> None:
        self.failures = failures
        self.error = error or _rate_limit()
        self.calls = 0

    def _maybe_fail(self) -> None:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._maybe_fail()
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self._maybe_fail()
        return [1.0, 0.0]


def test_retries_rate_limits_then_succeeds() -> None:
    inner = Flaky(failures=2)
    wrapped = emb.RetryingEmbeddings(inner, max_retries=3, min_wait=0, max_wait=0)
    assert wrapped.embed_documents(["a", "b"]) == [[1.0, 0.0], [1.0, 0.0]]
    assert inner.calls == 3


def test_gives_up_after_max_retries() -> None:
    inner = Flaky(failures=10)
    wrapped = emb.RetryingEmbeddings(inner, max_retries=2, min_wait=0, max_wait=0)
    with pytest.raises(openai.RateLimitError):
        wrapped.embed_query("q")
    assert inner.calls == 3


def test_does_not_retry_non_transient_errors() -> None:
    inner = Flaky(failures=1, error=ValueError("bad input"))
    wrapped = emb.RetryingEmbeddings(inner, max_retries=5, min_wait=0, max_wait=0)
    with pytest.raises(ValueError, match="bad input"):
        wrapped.embed_query("q")
    assert inner.calls == 1


def test_fake_provider(settings: Settings) -> None:
    dense = emb.build_dense_embeddings(settings)
    assert isinstance(dense, DeterministicFakeEmbedding)
    assert len(dense.embed_query("x")) == settings.fake_embedding_dim


def test_nebius_provider_is_configured_without_network() -> None:
    settings = Settings(
        embedding_provider=EmbeddingProvider.NEBIUS,
        nebius_api_key="k",
        nebius_embedding_model="some/model",
        nebius_base_url="https://nebius.test/v1/",
        embedding_batch_size=16,
    )
    dense = emb.build_dense_embeddings(settings)
    assert isinstance(dense, emb.RetryingEmbeddings)
    inner = dense.inner
    assert isinstance(inner, OpenAIEmbeddings)
    assert inner.model == "some/model"
    assert inner.openai_api_base == "https://nebius.test/v1/"
    assert inner.check_embedding_ctx_length is False
    assert inner.chunk_size == 16


def test_nebius_provider_calls_openai_compatible_endpoint() -> None:
    """End-to-end through the OpenAI client against a mocked Nebius HTTP endpoint."""
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content)
        seen.append({"url": str(request.url), "auth": request.headers["authorization"], **body})
        data = [
            {"object": "embedding", "index": i, "embedding": [0.1, 0.2]}
            for i in range(len(body["input"]))
        ]
        return httpx.Response(200, json={"object": "list", "data": data, "model": body["model"]})

    inner = OpenAIEmbeddings(
        model="some/model",
        base_url="https://nebius.test/v1/",
        api_key="k",
        check_embedding_ctx_length=False,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert inner.embed_documents(["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]
    assert seen[0]["url"] == "https://nebius.test/v1/embeddings"
    assert seen[0]["auth"] == "Bearer k"
    assert seen[0]["input"] == ["a", "b"]


def test_sparse_factory_passes_model_and_cache(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    captured: dict[str, Any] = {}

    class StubSparse:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(emb, "FastEmbedSparse", StubSparse)
    settings.fastembed_cache_dir = settings.data_dir / "cache"
    emb.build_sparse_embeddings(settings)
    assert captured == {"model_name": "Qdrant/bm25", "cache_dir": str(settings.data_dir / "cache")}
