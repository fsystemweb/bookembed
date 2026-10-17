"""Load and clean the books CSV into typed records."""

from __future__ import annotations

import logging
import math
import re
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = (
    "isbn13",
    "isbn10",
    "title",
    "subtitle",
    "authors",
    "categories",
    "thumbnail",
    "description",
    "published_year",
    "average_rating",
    "num_pages",
    "ratings_count",
)

# Fixed namespace so point IDs are stable across runs and machines.
POINT_ID_NAMESPACE = uuid.UUID("6f1c2a5e-8b1d-4c3e-9a7f-2d4b6e8f0a1c")

_LIST_SPLIT = re.compile(r"\s*;\s*")


class Book(BaseModel):
    isbn13: str
    isbn10: str | None = None
    title: str
    subtitle: str | None = None
    authors: list[str] = []
    categories: list[str] = []
    thumbnail: str | None = None
    description: str | None = None
    published_year: int | None = None
    average_rating: float | None = None
    num_pages: int | None = None
    ratings_count: int | None = None

    @property
    def point_id(self) -> str:
        return point_id_for(self.isbn13)

    def embedding_text(self) -> str:
        """Text that gets embedded: every descriptive field, labelled, skipping blanks."""
        parts = [f"Title: {self.title}"]
        if self.subtitle:
            parts.append(f"Subtitle: {self.subtitle}")
        if self.authors:
            parts.append(f"Authors: {', '.join(self.authors)}")
        if self.categories:
            parts.append(f"Categories: {', '.join(self.categories)}")
        if self.description:
            parts.append(f"Description: {self.description}")
        return "\n".join(parts)

    def payload(self) -> dict[str, Any]:
        return self.model_dump()


def point_id_for(isbn13: str) -> str:
    return str(uuid.uuid5(POINT_ID_NAMESPACE, isbn13))


def _clean_str(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = " ".join(str(value).split())
    return text or None


def _split_list(value: object) -> list[str]:
    text = _clean_str(value)
    if not text:
        return []
    seen: dict[str, None] = {}
    for item in _LIST_SPLIT.split(text):
        if item:
            seen.setdefault(item, None)
    return list(seen)


def _to_float(value: object) -> float | None:
    text = _clean_str(value)
    if text is None:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return None if math.isnan(number) or math.isinf(number) else number


def _to_int(value: object) -> int | None:
    number = _to_float(value)
    return None if number is None else int(number)


def _normalize_isbn13(value: object) -> str | None:
    text = _clean_str(value)
    if text is None:
        return None
    # Spreadsheet round-trips turn ISBNs into floats ("9.78e12", "9780002005883.0").
    if re.fullmatch(r"[0-9.eE+]+", text) and not text.isdigit():
        number = _to_float(text)
        text = str(int(number)) if number is not None else text
    digits = re.sub(r"[^0-9Xx]", "", text)
    return digits or None


def parse_row(row: dict[str, object]) -> Book | None:
    """Turn one raw CSV row into a Book, or None if it lacks an ISBN13 or title."""
    isbn13 = _normalize_isbn13(row.get("isbn13"))
    title = _clean_str(row.get("title"))
    if not isbn13 or not title:
        return None
    return Book(
        isbn13=isbn13,
        isbn10=_clean_str(row.get("isbn10")),
        title=title,
        subtitle=_clean_str(row.get("subtitle")),
        authors=_split_list(row.get("authors")),
        categories=_split_list(row.get("categories")),
        thumbnail=_clean_str(row.get("thumbnail")),
        description=_clean_str(row.get("description")),
        published_year=_to_int(row.get("published_year")),
        average_rating=_to_float(row.get("average_rating")),
        num_pages=_to_int(row.get("num_pages")),
        ratings_count=_to_int(row.get("ratings_count")),
    )


def load_books(path: Path) -> list[Book]:
    """Read the CSV, drop unusable rows and de-duplicate on ISBN13 (last row wins)."""
    # Read everything as strings so ISBNs keep leading zeros; coercion happens per field.
    frame = pd.read_csv(path, dtype=str, keep_default_na=True)
    missing = [col for col in ("isbn13", "title") if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    absent = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
    if absent:
        logger.warning("CSV is missing optional columns %s; they will be empty", absent)

    books: dict[str, Book] = {}
    skipped = 0
    for row in frame.to_dict(orient="records"):
        book = parse_row({str(k): v for k, v in row.items()})
        if book is None:
            skipped += 1
            continue
        books[book.isbn13] = book
    duplicates = len(frame) - skipped - len(books)
    logger.info(
        "Loaded %d books from %s (skipped %d invalid rows, %d duplicate ISBNs)",
        len(books),
        path,
        skipped,
        duplicates,
    )
    return list(books.values())
