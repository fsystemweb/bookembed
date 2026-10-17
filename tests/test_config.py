from __future__ import annotations

import pytest
from pydantic import ValidationError

from ingestion.config import EmbeddingProvider, Settings


def test_nebius_requires_api_key() -> None:
    with pytest.raises(ValidationError, match="NEBIUS_API_KEY"):
        Settings(embedding_provider=EmbeddingProvider.NEBIUS)


def test_settings_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEBIUS_API_KEY", "secret")
    monkeypatch.setenv("NEBIUS_EMBEDDING_MODEL", "BAAI/bge-en-icl")
    monkeypatch.setenv("QDRANT_COLLECTION", "other")
    settings = Settings()
    assert settings.nebius_api_key is not None
    assert settings.nebius_api_key.get_secret_value() == "secret"
    assert settings.nebius_embedding_model == "BAAI/bge-en-icl"
    assert settings.qdrant_collection == "other"
    assert "secret" not in repr(settings)


def test_fake_provider_needs_no_key() -> None:
    settings = Settings(embedding_provider="fake")
    assert settings.embedding_provider is EmbeddingProvider.FAKE
