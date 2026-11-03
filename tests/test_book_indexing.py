"""
Unit tests for book indexing: excluding dependency manifests and directory fallback.
Translates GIVEN/WHEN/THEN requirements from specs/book-indexing/spec.md.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import tkinter_gui.main as tk_main
import web_gui_pageoracle.main as web_main


class TestBookIndexing(unittest.TestCase):
    """Test suite verifying exclusion of dependency manifests and directory fallback."""

    def test_backends_are_bit_for_bit_identical(self) -> None:
        """Verify web_gui and tkinter_gui main.py files have identical SHA256 hashes."""
        web_file = Path("web_gui_pageoracle/main.py")
        tk_file = Path("tkinter_gui/main.py")
        h1 = hashlib.sha256(web_file.read_bytes()).hexdigest()
        h2 = hashlib.sha256(tk_file.read_bytes()).hexdigest()
        self.assertEqual(h1, h2, "web_gui_pageoracle/main.py and tkinter_gui/main.py must be bit-for-bit identical")

    def test_excluded_book_files_constant(self) -> None:
        """Verify EXCLUDED_BOOK_FILES exists and contains requirements.txt in both backends."""
        self.assertTrue(hasattr(web_main, "EXCLUDED_BOOK_FILES"))
        self.assertIn("requirements.txt", web_main.EXCLUDED_BOOK_FILES)
        self.assertTrue(hasattr(tk_main, "EXCLUDED_BOOK_FILES"))
        self.assertIn("requirements.txt", tk_main.EXCLUDED_BOOK_FILES)

    def test_requirements_txt_is_not_indexed(self) -> None:
        """
        GIVEN a directory containing requirements.txt and no literary book files
        WHEN automatic book loading is executed during initialization
        THEN the system SHALL NOT annotate, chunk, or index requirements.txt as a book
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            req_file = Path(tmp_dir) / "requirements.txt"
            req_file.write_text("flask>=3.0.0\nlangchain>=0.3.0\n", encoding="utf-8")

            backend = web_main.PageOracleBackend(books_dir=tmp_dir)
            backend._load_all_books()

            self.assertNotIn("requirements.txt", backend.loaded_books)
            for split in backend.splits:
                self.assertNotIn("flask>=3.0.0", split.page_content)
                self.assertNotEqual(Path(split.metadata.get("source", "")).name, "requirements.txt")

    def test_requirements_txt_ignored_when_book_present(self) -> None:
        """
        GIVEN a directory containing both requirements.txt and a valid literary book file
        WHEN automatic book loading is executed
        THEN the book is loaded, and requirements.txt is ignored
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            req_file = Path(tmp_dir) / "requirements.txt"
            req_file.write_text("flask>=3.0.0\n", encoding="utf-8")
            book_file = Path(tmp_dir) / "sample_story.txt"
            book_file.write_text(
                "Часть 1. Глава 1.\nЖила-была зеленая лампа в окне на третьем этаже.", encoding="utf-8"
            )

            backend = web_main.PageOracleBackend(books_dir=tmp_dir)
            backend._load_all_books()

            self.assertIn("sample_story.txt", backend.loaded_books)
            self.assertNotIn("requirements.txt", backend.loaded_books)
            self.assertGreater(len(backend.splits), 0)

    def test_fallback_to_adjacent_books_folder(self) -> None:
        """
        GIVEN the configured books directory contains no eligible text books
        GIVEN a books directory exists in the same folder as the backend script containing .txt files
        WHEN automatic book loading is executed during initialization
        THEN the system SHALL load, annotate, and index the .txt files from that books directory
        THEN the system SHALL update its active books directory to point to that books directory
        """
        with tempfile.TemporaryDirectory() as empty_dir:
            # empty_dir has no .txt files
            backend = web_main.PageOracleBackend(books_dir=empty_dir)
            backend._load_all_books()

            adjacent_books = Path(web_main.__file__).resolve().parent / "books"
            if adjacent_books.is_dir() and list(adjacent_books.glob("*.txt")):
                self.assertEqual(Path(backend.books_dir).resolve(), adjacent_books.resolve())
                self.assertGreater(len(backend.loaded_books), 0)
                self.assertGreater(len(backend.splits), 0)


if __name__ == "__main__":
    unittest.main()
