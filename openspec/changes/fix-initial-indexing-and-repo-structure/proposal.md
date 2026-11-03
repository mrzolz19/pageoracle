# Proposal

## Why

When cloning the repository and launching `gui.py` for the first time in either GUI directory, `_load_all_books` globs `*.txt` in `self.books_dir` (defaulting to `.`). Because `requirements.txt` ends with `.txt` and lives in the launch directory, the application falsely treats it as a book, chunks it, and creates a Chroma vector store for dependency requirements. Meanwhile, the actual sample books are stored in a `books/` subfolder and are completely ignored. Furthermore, the repository contains a deleted root `Book_Example/` directory in git status, while the `books/` folders in `web_gui_pageoracle/` and `tkinter_gui/` are untracked, causing spellcheck gates (`uvx typos`) to fail on Tolstoy's multilingual dialogues.

## What Changes

- Exclude `requirements.txt` from automatic book discovery in `_load_all_books()`.
- Add a fallback/alternative condition to search and index books from the `books/` directory adjacent to the Python application files when no valid book files are found in `self.books_dir`.
- Leave `add_document(filepath)` unchanged (per user decision).
- Remove the legacy `Book_Example/` folder from git tracking.
- Track `web_gui_pageoracle/books/` and `tkinter_gui/books/` in git.
- Update tooling ignore configurations (`typos.toml`, `.rumdl.toml`, `pyproject.toml`, `.gitignore`) so that quality gates scan cleanly.
- Keep `web_gui_pageoracle/main.py` and `tkinter_gui/main.py` bit-for-bit synchronized.
- Update `README.md` to accurately document the launch instructions and the `books/` directory structure.

## Capabilities

### New Capabilities

- `book-indexing`: Automatic book discovery with non-book file exclusion and directory fallback to the adjacent `books/` folder.

### Modified Capabilities

None.

## Impact

- Affected backend modules: `web_gui_pageoracle/main.py`, `tkinter_gui/main.py`.
- Affected configuration files: `typos.toml`, `.rumdl.toml`, `pyproject.toml`, `.gitignore`.
- Affected documentation: `README.md`.
- Affected git tracking: git index removal of `Book_Example/` and tracking of `books/` subdirectories.
