# Decision Log

This document records architectural, technical, and process decisions for PageOracle.
Before proposing plans or changes, consult the Table of Contents and relevant domain records.

**Defined Domains:**

- `architecture`: High-level system structure, services, entrypoints.
- `rag-pipeline`: Retrieval, chunking, embeddings, reranking, LangGraph state machine.
- `data`: Book ingestion, metadata schemas, persistence (settings, history).
- `gui`: Web GUI (Flask) and Desktop GUI (Tkinter) interfaces and synchronization.
- `workflow`: Agent interaction, development cycles, testing, task tracking.

---

## Table of Contents

| ID | Date | Domain | Title | Status |
|---|---|---|---|---|
| D-001 | 2026-11-03 | workflow | Dev Quality Gates adoption with mise, ruff, ty, deptry, osv-scanner | Active |
| D-002 | 2026-11-03 | workflow | Configure ty allowed-unresolved-imports for project dependencies | Active |
| D-003 | 2026-11-03 | rag-pipeline | Fix PyTorch CUDA 12.8 wheel ABI and pin reproducible dependencies | Active |
| D-004 | 2026-11-03 | data | Exclude dependency manifests and fallback to adjacent books directory | Active |
| D-005 | 2026-11-04 | architecture | Custom OpenAI-compatible providers, free-form models, and decoupled embeddings | Active |
| D-006 | 2026-11-04 | data | Security gitignore for runtime settings and resilient book encoding fallback | Active |

---

## Decisions

## D-001. Dev Quality Gates Adoption (2026-11-03)

**Question:** How to enforce automated quality gates and handle existing code violations and toolchain execution across platforms.
**Options:** Baseline legacy violations vs direct fix/format (Option A); standard mise only vs mise with native uv fallback runner.
**Chosen:** Option A: fix mechanical findings directly across web_gui and tkinter_gui with sync, standardize requirements.txt, use mise for CI and uv runner locally.
**Why:** Resolves root-cause code smells without accumulating hidden baseline debt and ensures checks run smoothly on Windows without extra dependencies.
**Why it should work:** Direct fixes eliminate lint and type errors permanently; dual-entrypoint synchronization is verified; uvx runs all tools directly.
**Affects:** Toolchain configuration, pyproject.toml, requirements.txt, AGENTS.md, CI workflows, and both GUI modules.
**Status:** Active

## D-002. Configure ty allowed-unresolved-imports for Project Dependencies (2026-11-03)

**Question:** How to prevent `uvx ty check .` failing on unresolved third-party imports in clean/CI environments without pre-installing gigabytes of ML dependencies.
**Options:** Disable `unresolved-import` globally; install all wheels (~4GB) in CI; configure `allowed-unresolved-imports` for declared dependencies.
**Chosen:** Configure `[tool.ty.analysis] allowed-unresolved-imports` explicitly for declared third-party dependencies in `pyproject.toml`.
**Why:** Avoids downloading massive ML runtime packages during fast static checks while keeping `unresolved-import` enforced for undeclared or misspelled modules.
**Why it should work:** `deptry` already strictly validates all imports against `dependencies`, while `ty` validates first-party code and standard library types.
**Affects:** `pyproject.toml`, CI quality gates.
**Status:** Active

## D-003. PyTorch CUDA 12.8 Wheel ABI and Pinned Dependencies (2026-11-03)

**Question:** How to fix `RuntimeError: operator torchvision::nms does not exist` and ensure reproducible environments across GUI versions.
**Options:** Purge torchvision/torchaudio (CPU-only without GPU acceleration); upgrade PyTorch to CUDA 12.8 wheel matching torchvision and pin dependencies.
**Chosen:** User selected Option 1: Install `torch==2.11.0+cu128` to match installed CUDA 12.8 wheels and pin explicit version specifications in requirements.
**Why:** User requested GPU support for fast CrossEncoder reranking and strict dependency pinning for reproducible deployment.
**Why it should work:** Eliminates C++ extension ABI mismatch between PyTorch and torchvision/torchaudio, restores hardware GPU acceleration, and pins tested dependency matrix.
**Affects:** `web_gui_pageoracle/requirements.txt`, `tkinter_gui/requirements.txt`, `web_gui_pageoracle/main.py`, `tkinter_gui/main.py`.
**Status:** Active

## D-004. Exclude Dependency Manifests and Adjacent Books Fallback (2026-11-03)

**Question:** How to prevent initial startup from indexing `requirements.txt` into Chroma and discover sample books reliably across entrypoints.
**Options:** Inline filtering in both `main.py` vs centralized `EXCLUDED_BOOK_FILES` constant and fallback to adjacent `books/` directory; leave `Book_Example/` vs migrate git tracking to entrypoint `books/`.
**Chosen:** Centralized `EXCLUDED_BOOK_FILES = {"requirements.txt"}` with fallback to `books/` folder in `_load_all_books()`, update `self.books_dir`, track `books/` in git, and exclude `**/books/**` in toolchain linters.
**Why:** User requested fixing false positive indexing on `requirements.txt` and ensuring books in `books/` are recognized and indexed on first launch.
**Why it should work:** Eliminates false indexing of dependency manifests, resolves proper book directories automatically, fixes multilingual typo gate failures, and preserves bit-for-bit sync across both GUIs.
**Affects:** `web_gui_pageoracle/main.py`, `tkinter_gui/main.py`, `typos.toml`, `.rumdl.toml`, `pyproject.toml`, `.gitignore`, `README.md`.
**Status:** Active

## D-005. Custom Providers, Free-Form Models, and Decoupled Embeddings (2026-11-04)

**Question:** How to enable unlimited custom AI providers, free-form LLM and embedding model entry, and flexible embedding routing across Web and Tkinter GUIs.
**Options:** Strict hardcoded dictionaries vs extensible provider registry in settings.json with OpenAI-compatible Chat/Embeddings fallback; coupled LLM/Embedding endpoints vs decoupled configuration with an optional sync toggle.
**Chosen:** Extensible `custom_providers` list in `settings.json`, OpenAI-compatible endpoints for custom LLM (`ChatOpenAI`) and embeddings, free-form model entry via editable comboboxes and HTML datalists, with decoupled embedding settings and a "same as LLM" sync toggle.
**Why:** User requested unlimited custom providers (name, endpoint, api_key, model, embedding_model), arbitrary model entry for built-in providers, separate embedding settings, and a synchronization toggle when LLM and embeddings share the same provider/endpoint.
**Why it should work:** OpenAI API format is the universal standard across self-hosted and cloud LLM/embedding servers (Ollama, vLLM, LM Studio, LiteLLM); datalists and normal comboboxes preserve autocomplete suggestions while permitting arbitrary user input; decoupled settings allow mixing local embeddings with remote LLMs.
**Affects:** `settings.json`, `web_gui_pageoracle/main.py`, `tkinter_gui/main.py`, `web_gui_pageoracle/gui.py`, `tkinter_gui/gui.py`, `web_gui_pageoracle/templates/index.html`, `web_gui_pageoracle/static/app.js`, `web_gui_pageoracle/static/style.css`.
**Status:** Active

## D-006. Security Gitignore for Settings and Resilient Book Ingestion (2026-11-04)

**Question:** How to prevent accidental commit of live API keys in `settings.json` and protect book loading against unhandled decoding crashes.
**Options:** Rely on manual omission during staging vs automate git and linter exclusion; fail-fast on malformed book files vs multi-tier encoding fallback with graceful error recovery.
**Chosen:** Add `settings.json`, `chat_history.json`, and `graphify-out` to `.gitignore` and `.rumdl.toml`; wrap tier-3 chardet encoding in `_load_and_annotate` with exception logging returning `[]`; defer heavy `CrossEncoder` import to `SimpleReranker.__init__`.
**Why:** Prevents secret leakage to source control, ensures quality gates pass cleanly without unmonitored tool artifacts, eliminates app startup crashes on corrupted book files, and avoids high CUDA commit spikes during tests.
**Why it should work:** Gitignore blocks tracking of credential files; graceful exception handling prevents an unreadable file from breaking full library ingestion; lazy import decouples testing from PyTorch CUDA initialization.
**Affects:** `.gitignore`, `.rumdl.toml`, `web_gui_pageoracle/main.py`, `tkinter_gui/main.py`, `tests/test_book_indexing.py`.
**Status:** Active

---

## Archive

<!-- Superseded decisions moved here unchanged except for status: Superseded by D-<number> -->
