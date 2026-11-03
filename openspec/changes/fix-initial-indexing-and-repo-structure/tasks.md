# Tasks

## 1. Toolchain & Repository Structure Setup

- [x] 1.1 Update `typos.toml`, `.rumdl.toml`, `pyproject.toml`, and `.gitignore` to exclude `**/books/**` (replacing legacy `Book_Example`) and verify `uvx typos` passes
- [x] 1.2 Remove `Book_Example/` from git tracking (`git rm -r Book_Example`) and stage `web_gui_pageoracle/books/` and `tkinter_gui/books/`

## 2. Core Book Indexing Implementation (TDD)

- [x] 2.1 Write unit tests in `tests/test_book_indexing.py` translating spec scenarios (ignoring `requirements.txt`, falling back to `books/` folder) and verify tests fail
- [x] 2.2 Implement `EXCLUDED_BOOK_FILES` and fallback logic in `web_gui_pageoracle/main.py` and verify unit tests pass
- [x] 2.3 Synchronize `web_gui_pageoracle/main.py` to `tkinter_gui/main.py` bit-for-bit and verify identical checksums and test passes

## 3. Documentation & Quality Gates

- [x] 3.1 Update `README.md` step 4 launch instructions and book directory references, verifying markdown linting passes
- [x] 3.2 Execute aggregate quality gate runner `python scripts/check.py` and verify all 8 checks exit with status 0
- [x] 3.3 Record Decision D-004 in `docs/DECISIONS.md`, commit all changes, and push to remote repository `origin/main`
