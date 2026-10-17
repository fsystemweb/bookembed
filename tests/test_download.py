from __future__ import annotations

import io
import zipfile
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from ingestion.config import Settings
from ingestion.download import download_dataset, resolve_csv


def _zip(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def _client(payload: bytes, seen: list[httpx.Request]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_download_with_credentials(settings: Settings) -> None:
    settings.kaggle_username = "user"
    settings.kaggle_key = SecretStr("key")
    seen: list[httpx.Request] = []
    path = download_dataset(settings, _client(_zip({"books.csv": "isbn13,title\n"}), seen))
    assert path.read_text() == "isbn13,title\n"
    assert seen[0].url.path == "/api/v1/datasets/download/dylanjcastillo/7k-books-with-metadata"
    assert seen[0].headers["authorization"].startswith("Basic ")


def test_download_anonymous(settings: Settings) -> None:
    seen: list[httpx.Request] = []
    download_dataset(settings, _client(_zip({"books.csv": "x"}), seen))
    assert "authorization" not in seen[0].headers


def test_download_skips_existing_file(settings: Settings) -> None:
    settings.data_dir.mkdir(parents=True)
    (settings.data_dir / "books.csv").write_text("cached")
    seen: list[httpx.Request] = []
    assert download_dataset(settings, _client(b"", seen)).read_text() == "cached"
    assert seen == []


def test_download_missing_file_in_archive(settings: Settings) -> None:
    with pytest.raises(FileNotFoundError, match=r"books\.csv not in archive"):
        download_dataset(settings, _client(_zip({"other.csv": "x"}), []))


def test_download_uses_configured_url(settings: Settings) -> None:
    settings.kaggle_download_url = "https://mirror.test/{dataset}.zip"
    seen: list[httpx.Request] = []
    download_dataset(settings, _client(_zip({"books.csv": "x"}), seen))
    assert str(seen[0].url) == "https://mirror.test/dylanjcastillo/7k-books-with-metadata.zip"


def test_unit_tests_cannot_reach_network(settings: Settings) -> None:
    with pytest.raises(RuntimeError, match="tried to reach the network"):
        download_dataset(settings)


def test_download_http_error(settings: Settings) -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(403)))
    with pytest.raises(httpx.HTTPStatusError):
        download_dataset(settings, client)


def test_resolve_csv_prefers_books_csv(settings: Settings, tmp_path: Path) -> None:
    local = tmp_path / "local.csv"
    local.write_text("x")
    settings.books_csv = local
    assert resolve_csv(settings) == local


def test_resolve_csv_missing_books_csv(settings: Settings, tmp_path: Path) -> None:
    settings.books_csv = tmp_path / "nope.csv"
    with pytest.raises(FileNotFoundError):
        resolve_csv(settings)


def test_resolve_csv_downloads_when_unset(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("ingestion.download.download_dataset", lambda s: tmp_path / "dl.csv")
    assert resolve_csv(settings) == tmp_path / "dl.csv"
