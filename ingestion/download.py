"""Fetch the dataset archive from the Kaggle API."""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import httpx

from ingestion.config import Settings

logger = logging.getLogger(__name__)


def download_dataset(settings: Settings, client: httpx.Client | None = None) -> Path:
    """Download `settings.kaggle_file` into `settings.data_dir` and return its path.

    Uses HTTP basic auth with KAGGLE_USERNAME / KAGGLE_KEY when both are set (the same
    credentials the official `kaggle` CLI uses). Kaggle also serves public datasets
    anonymously, so the download still works without them.
    """
    target = settings.data_dir / settings.kaggle_file
    if target.exists():
        logger.info("Dataset already present at %s, skipping download", target)
        return target

    auth: httpx.BasicAuth | None = None
    if settings.kaggle_username and settings.kaggle_key:
        auth = httpx.BasicAuth(settings.kaggle_username, settings.kaggle_key.get_secret_value())
    else:
        logger.warning("KAGGLE_USERNAME/KAGGLE_KEY not set; trying anonymous download")

    url = settings.kaggle_download_url.format(dataset=settings.kaggle_dataset)
    owns_client = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=settings.download_timeout)
    try:
        logger.info("Downloading %s", url)
        response = http.get(url, auth=auth)
        response.raise_for_status()
    finally:
        if owns_client:
            http.close()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = archive.namelist()
        if settings.kaggle_file not in names:
            raise FileNotFoundError(f"{settings.kaggle_file} not in archive (found {names})")
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(settings.kaggle_file))
    logger.info("Saved dataset to %s", target)
    return target


def resolve_csv(settings: Settings) -> Path:
    """Use BOOKS_CSV when it points at an existing file, otherwise download."""
    if settings.books_csv is not None:
        if settings.books_csv.exists():
            return settings.books_csv
        raise FileNotFoundError(f"BOOKS_CSV={settings.books_csv} does not exist")
    return download_dataset(settings)
