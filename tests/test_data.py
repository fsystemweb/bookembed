from __future__ import annotations

import math
from pathlib import Path

import pytest

from ingestion.data import Book, load_books, parse_row, point_id_for
from tests.conftest import FIXTURE_CSV


def test_parse_row_cleans_and_coerces() -> None:
    book = parse_row(
        {
            "isbn13": " 9780002005883 ",
            "isbn10": "0002005883",
            "title": "  Gilead\n",
            "subtitle": float("nan"),
            "authors": "Ann Smith; Bob Jones ;Ann Smith",
            "categories": "Fiction",
            "description": "",
            "published_year": "2004.0",
            "average_rating": "3.85",
            "num_pages": "247",
            "ratings_count": "not a number",
        }
    )
    assert book is not None
    assert book.isbn13 == "9780002005883"
    assert book.title == "Gilead"
    assert book.subtitle is None
    assert book.authors == ["Ann Smith", "Bob Jones"]
    assert book.categories == ["Fiction"]
    assert book.description is None
    assert book.published_year == 2004
    assert book.average_rating == pytest.approx(3.85)
    assert book.num_pages == 247
    assert book.ratings_count is None
    assert book.thumbnail is None


@pytest.mark.parametrize(
    "row",
    [
        {"isbn13": "", "title": "No ISBN"},
        {"isbn13": float("nan"), "title": "No ISBN"},
        {"isbn13": "9780002005883", "title": float("nan")},
        {"isbn13": "9780002005883"},
    ],
)
def test_parse_row_rejects_rows_without_isbn_or_title(row: dict[str, object]) -> None:
    assert parse_row(row) is None


@pytest.mark.parametrize("raw", ["9.780002005883e12", "9780002005883.0", "978-0-00-200588-3"])
def test_isbn13_normalised_from_spreadsheet_formats(raw: str) -> None:
    book = parse_row({"isbn13": raw, "title": "T"})
    assert book is not None
    assert book.isbn13 == "9780002005883"


@pytest.mark.parametrize("raw", ["inf", "nan", "  "])
def test_non_finite_numbers_become_none(raw: str) -> None:
    book = parse_row({"isbn13": "1", "title": "T", "average_rating": raw})
    assert book is not None
    assert book.average_rating is None


def test_embedding_text_skips_blank_fields() -> None:
    book = Book(isbn13="1", title="Dune", authors=["Frank Herbert"], categories=[])
    assert book.embedding_text() == "Title: Dune\nAuthors: Frank Herbert"


def test_embedding_text_includes_all_fields() -> None:
    book = Book(
        isbn13="1",
        title="T",
        subtitle="S",
        authors=["A", "B"],
        categories=["C"],
        description="D",
    )
    assert book.embedding_text() == (
        "Title: T\nSubtitle: S\nAuthors: A, B\nCategories: C\nDescription: D"
    )


def test_point_id_is_deterministic_uuid() -> None:
    assert point_id_for("9780002005883") == point_id_for("9780002005883")
    assert point_id_for("9780002005883") != point_id_for("9780002005884")
    assert Book(isbn13="9780002005883", title="x").point_id == point_id_for("9780002005883")


def test_payload_is_json_friendly() -> None:
    book = Book(isbn13="1", title="T", authors=["A"], average_rating=4.0)
    payload = book.payload()
    assert payload["authors"] == ["A"]
    assert payload["average_rating"] == 4.0
    assert payload["published_year"] is None


def test_load_fixture() -> None:
    books = load_books(FIXTURE_CSV)
    assert 45 <= len(books) <= 55
    assert len({b.isbn13 for b in books}) == len(books)
    assert any(len(b.authors) > 1 for b in books), "fixture should cover multi-author rows"
    assert any(b.description is None for b in books), "fixture should cover missing descriptions"
    assert any(b.average_rating is None for b in books), "fixture should cover missing ratings"
    for book in books:
        assert book.title
        assert all(";" not in author for author in book.authors)
        if book.average_rating is not None:
            assert not math.isnan(book.average_rating)


def test_load_books_deduplicates_and_skips(tmp_path: Path) -> None:
    csv = tmp_path / "books.csv"
    csv.write_text("isbn13,title,authors\n1,First,A\n1,First again,B\n,No isbn,C\n2,Second,\n")
    books = load_books(csv)
    assert [b.title for b in books] == ["First again", "Second"]
    assert books[1].authors == []


def test_load_books_requires_core_columns(tmp_path: Path) -> None:
    csv = tmp_path / "books.csv"
    csv.write_text("isbn10,name\n1,x\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_books(csv)
