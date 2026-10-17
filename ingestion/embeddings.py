"""Dense (Nebius) and sparse (BM25) embedding factories, with rate-limit retries."""

from __future__ import annotations

import logging

import openai
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse
from langchain_qdrant.sparse_embeddings import SparseEmbeddings
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from ingestion.config import EmbeddingProvider, Settings

logger = logging.getLogger(__name__)

RETRYABLE_ERRORS: tuple[type[BaseException], ...] = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


class RetryingEmbeddings(Embeddings):
    """Wraps an Embeddings model with exponential backoff on transient API errors."""

    def __init__(
        self,
        inner: Embeddings,
        max_retries: int,
        min_wait: float = 1.0,
        max_wait: float = 60.0,
    ) -> None:
        self.inner = inner
        self._retrying = Retrying(
            retry=retry_if_exception_type(RETRYABLE_ERRORS),
            stop=stop_after_attempt(max_retries + 1),
            wait=wait_random_exponential(multiplier=min_wait, max=max_wait),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._retrying(self.inner.embed_documents, texts)

    def embed_query(self, text: str) -> list[float]:
        return self._retrying(self.inner.embed_query, text)


def build_dense_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_provider is EmbeddingProvider.FAKE:
        return DeterministicFakeEmbedding(size=settings.fake_embedding_dim)

    assert settings.nebius_api_key is not None  # enforced by Settings validation
    nebius = OpenAIEmbeddings(
        model=settings.nebius_embedding_model,
        base_url=settings.nebius_base_url,
        api_key=settings.nebius_api_key,
        # Nebius models are not OpenAI models: send raw strings, not tiktoken ids.
        check_embedding_ctx_length=False,
        chunk_size=settings.embedding_batch_size,
        # Retries are handled by RetryingEmbeddings so backoff is logged and tunable.
        max_retries=0,
    )
    return RetryingEmbeddings(nebius, max_retries=settings.embedding_max_retries)


def build_sparse_embeddings(settings: Settings) -> SparseEmbeddings:
    cache_dir = str(settings.fastembed_cache_dir) if settings.fastembed_cache_dir else None
    return FastEmbedSparse(model_name=settings.sparse_model, cache_dir=cache_dir)
