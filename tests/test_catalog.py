"""Tests for catalog loading and filtering."""

import json
import pytest
from pathlib import Path

from agentkv.catalog import load_catalog, filter_catalog


def test_load_builtin_catalog():
    books = load_catalog()
    assert len(books) == 100
    assert all("id" in b and "title" in b for b in books)


def test_load_custom_catalog(tmp_path):
    p = tmp_path / "cat.jsonl"
    p.write_text(
        json.dumps({"id": 1, "title": "A", "questions": []}) + "\n"
        + json.dumps({"id": 2, "title": "B", "questions": []}) + "\n"
    )
    books = load_catalog(p)
    assert len(books) == 2
    assert books[0]["id"] == 1


def test_filter_by_ids():
    books = [{"id": i} for i in range(5)]
    out = filter_catalog(books, book_ids=[1, 3])
    assert [b["id"] for b in out] == [1, 3]


def test_filter_num_books():
    books = [{"id": i} for i in range(10)]
    out = filter_catalog(books, num_books=3)
    assert len(out) == 3


def test_filter_empty_ids_means_all():
    books = [{"id": i} for i in range(5)]
    assert filter_catalog(books) == books
