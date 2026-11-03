# Design

## Context

PageOracle has two entrypoints (`web_gui_pageoracle` and `tkinter_gui`) with separate physical copies of `PageOracleBackend` (`main.py`) that must remain bit-for-bit synchronized. When launched with default parameters, `PageOracleBackend(books_dir=".")` searches for all `*.txt` files in `.`. Because `requirements.txt` resides in each entrypoint directory, it is incorrectly picked up as a book and indexed into Chroma. Meanwhile, the actual sample books reside in a `books/` subfolder in each GUI directory and are left untracked in git, while the old root directory `Book_Example/` is marked deleted in git status.

## Goals / Non-Goals

**Goals:**

- Filter `requirements.txt` during automatic book discovery in `_load_all_books()`.
- If no valid text documents are found in `self.books_dir`, fall back to the `books/` folder in the same directory as `main.py` (`Path(__file__).resolve().parent / "books"`).
- Update `self.books_dir` to the resolved `books/` folder when fallback succeeds so subsequent operations and logging reflect the actual library location.
- Maintain bit-for-bit backend synchronization between `web_gui_pageoracle/main.py` and `tkinter_gui/main.py`.
- Remove `Book_Example/` from git index and track `web_gui_pageoracle/books/` and `tkinter_gui/books/`.
- Update quality gate configs (`typos.toml`, `.rumdl.toml`, `pyproject.toml`, `.gitignore`) so all checks pass with status 0.
- Update `README.md` to reflect the current folder structure and launch instructions.

**Non-Goals:**

- Modifying `add_document(filepath)`: per explicit user instruction, `add_document` logic remains unchanged and does not perform exclusion checks.
- Modifying chunking, reranking, or vectorstore persistence mechanisms.

## Decisions

### 1. Centralized exclusion constant in backend

- **Decision:** Define `EXCLUDED_BOOK_FILES = {"requirements.txt"}` in both `main.py` files.
- **Why:** Centralizes the list of non-book text files and makes future exclusions straightforward.
- **Alternatives considered:** Hardcoding `f.name == "requirements.txt"` inline (less maintainable).

### 2. Fallback to adjacent `books/` directory

- **Decision:** In `_load_all_books()`, first scan `Path(self.books_dir)`. Filter out files in `EXCLUDED_BOOK_FILES`. If no eligible `.txt` files are found, check `Path(__file__).resolve().parent / "books"`. If that directory exists and contains eligible `.txt` files, point `self.books_dir` to it and load its documents.
- **Why:** Supports both running from the GUI directory where books live in `./books/` and custom directories passed by callers, without false positive indexing on `requirements.txt`.
- **Alternatives considered:** Always merging files from both `.` and `books/` (could lead to accidental double ingestion if duplicate books exist).

### 3. Updating toolchain exclusions for books

- **Decision:** Update `typos.toml`, `.rumdl.toml`, `pyproject.toml`, and `.gitignore` to replace `Book_Example` with `books` (e.g. `**/books/**`).
- **Why:** Russian classical literature (e.g., "Война и мир") contains French and German dialogues that trigger hundreds of false positive typo warnings in `uvx typos`.
- **Alternatives considered:** Ignoring typos globally (violates quality gate standards).

## Risks / Trade-offs

- **[Risk]** Backend drift between `web_gui_pageoracle/main.py` and `tkinter_gui/main.py`.
  - **Mitigation:** Strict diff inspection to ensure exact identical code in both copies.
- **[Risk]** Git tracking large text files in `books/`.
  - **Mitigation:** The 6 sample books total ~7MB of plaintext, which is well within standard Git repository limits (and replaces the identical 7MB previously stored in `Book_Example/`).
