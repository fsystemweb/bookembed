"""Runtime configuration, read from environment variables (and an optional `.env`)."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingProvider(StrEnum):
    NEBIUS = "nebius"
    # Deterministic hash-based vectors: no network, no key. For tests, CI and smoke runs only.
    FAKE = "fake"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Dataset -----------------------------------------------------------
    kaggle_username: str | None = None
    kaggle_key: SecretStr | None = None
    kaggle_dataset: str = "dylanjcastillo/7k-books-with-metadata"
    kaggle_file: str = "books.csv"
    data_dir: Path = Path("data")
    # When set and the file exists, it is used as-is and nothing is downloaded.
    books_csv: Path | None = None

    # --- Embeddings --------------------------------------------------------
    embedding_provider: EmbeddingProvider = EmbeddingProvider.NEBIUS
    nebius_api_key: SecretStr | None = None
    nebius_base_url: str = "https://api.tokenfactory.nebius.com/v1/"
    nebius_embedding_model: str = "Qwen/Qwen3-Embedding-8B"
    fake_embedding_dim: int = Field(default=384, gt=0)
    embedding_batch_size: int = Field(default=64, gt=0)
    embedding_max_retries: int = Field(default=6, ge=0)
    sparse_model: str = "Qdrant/bm25"
    fastembed_cache_dir: Path | None = None

    # --- Qdrant ------------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "books"
    qdrant_timeout: int = Field(default=60, gt=0)
    qdrant_wait_seconds: int = Field(default=60, ge=0)

    @model_validator(mode="after")
    def _require_nebius_key(self) -> Settings:
        if self.embedding_provider is EmbeddingProvider.NEBIUS and not self.nebius_api_key:
            raise ValueError("NEBIUS_API_KEY is required when EMBEDDING_PROVIDER=nebius")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
