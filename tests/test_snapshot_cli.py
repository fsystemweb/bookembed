from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pydantic import SecretStr
from qdrant_client import QdrantClient

from ingestion import cli
from ingestion.config import Settings
from ingestion.snapshot import create_snapshot
from tests.conftest import FIXTURE_CSV, HashingSparse


class SnapshotClient:
    def __init__(self, name: str | None = "books-1.snapshot") -> None:
        self.name = name

    def create_snapshot(self, collection_name: str, wait: bool) -> Any:
        return SimpleNamespace(name=self.name) if self.name else None


def test_create_snapshot_downloads_file(settings: Settings, tmp_path: Path) -> None:
    settings.qdrant_api_key = SecretStr("qkey")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"SNAPSHOT")

    out = tmp_path / "snaps" / "books.snapshot"
    http = httpx.Client(transport=httpx.MockTransport(handler))
    create_snapshot(settings, SnapshotClient(), out, http)  # type: ignore[arg-type]
    assert out.read_bytes() == b"SNAPSHOT"
    assert seen[0].url.path == "/collections/test_books/snapshots/books-1.snapshot"
    assert seen[0].headers["api-key"] == "qkey"


def test_create_snapshot_requires_description(settings: Settings, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        create_snapshot(settings, SnapshotClient(name=None), tmp_path / "x")  # type: ignore[arg-type]


@pytest.fixture
def cli_env(
    settings: Settings, memory_client: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> QdrantClient:
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "connect", lambda s: memory_client)
    monkeypatch.setattr(cli, "build_sparse_embeddings", lambda s: HashingSparse())
    monkeypatch.setattr("ingestion.pipeline.connect", lambda s: memory_client)
    monkeypatch.setattr("ingestion.pipeline.build_sparse_embeddings", lambda s: HashingSparse())
    return memory_client


def test_cli_ingest_then_demo_search(
    cli_env: QdrantClient, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["ingest", "--csv", str(FIXTURE_CSV)]) == 0
    assert cli.main(["search", "--fail-if-empty"]) == 0
    out = capsys.readouterr().out
    assert "space opera with strong female lead" in out
    assert " 1. [" in out


def test_cli_search_json_with_filters(
    cli_env: QdrantClient, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main(["ingest", "--csv", str(FIXTURE_CSV)])
    capsys.readouterr()
    assert cli.main(["search", "war", "--min-year", "2001", "--min-rating", "4", "--json"]) == 0
    hits = json.loads(capsys.readouterr().out)
    assert hits
    assert all(h["book"]["published_year"] >= 2001 for h in hits)
    assert all(h["book"]["average_rating"] >= 4 for h in hits)


def test_cli_search_fail_if_empty(
    cli_env: QdrantClient, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main(["ingest", "--csv", str(FIXTURE_CSV)])
    code = cli.main(["-v", "search", "x", "--author", "Nobody At All", "--fail-if-empty"])
    assert code == 1
    assert "No results." in capsys.readouterr().out


def test_cli_download(settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "download_dataset", lambda s: tmp_path / "books.csv")
    assert cli.main(["download"]) == 0


def test_cli_snapshot(
    cli_env: QdrantClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(cli, "create_snapshot", lambda s, c, out: calls.append(out) or out)
    assert cli.main(["snapshot", "--out", str(tmp_path / "s.snapshot")]) == 0
    assert calls == [tmp_path / "s.snapshot"]
