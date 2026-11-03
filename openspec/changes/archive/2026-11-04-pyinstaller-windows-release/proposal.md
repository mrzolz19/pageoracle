# Proposal

## Why

Currently, running PageOracle requires cloning the repository, provisioning Python 3.11, and manually installing heavy ML dependencies into a virtual environment. This limits distribution strictly to developers. Packaging the Tkinter desktop GUI into a portable standalone Windows executable (`.exe`) and automating release builds via GitHub Actions allows end users to download, unzip, and run PageOracle immediately without any Python toolchain setup.

## What Changes

- Add root-level `pageoracle.spec` configuring PyInstaller to package `tkinter_gui/gui.py` into a portable directory bundle (`--onedir`) with all required hidden imports and data assets (PyTorch, ChromaDB, HuggingFace transformers, LangChain, certificates).
- Adapt path resolution in `tkinter_gui/gui.py` and `tkinter_gui/main.py` to recognize frozen execution (`sys.frozen`) and anchor `settings.json`, `books/`, `chat_history.json`, and `chroma_intro_db/` to the application directory rather than temporary extract folders.
- Add GitHub Actions CI/CD workflow `.github/workflows/release.yml` triggered on release tags (`v*`) and manual dispatch (`workflow_dispatch`).
- Configure a matrix build in CI producing two distinct release distributions:
  - `PageOracle-windows-x64-cpu.zip`: Lightweight universal build (~250 MB) running on any Windows machine.
  - `PageOracle-windows-x64-cu128.zip`: GPU-accelerated build with CUDA 12.8 support for users with NVIDIA RTX GPUs.
- Automate bundling of starter books and default settings template, and attach the zipped archives to GitHub Releases.

## Capabilities

### New Capabilities

- `desktop-distribution`: Covers portable standalone desktop packaging, frozen runtime path resolution for state persistence, and automated CI/CD matrix build and GitHub Release publishing for Windows.

### Modified Capabilities

*(None. Existing `book-indexing` requirements remain unchanged.)*

## Impact

- **Build Specification**: Adds `pageoracle.spec` in project root.
- **CI/CD Pipeline**: Adds `.github/workflows/release.yml` with GitHub Actions matrix runner validated by `jactionlint`.
- **Runtime & Persistence**: Updates `tkinter_gui/gui.py` and `tkinter_gui/main.py` to resolve paths relative to `sys.executable` when running in frozen mode, keeping `web_gui_pageoracle/main.py` bit-for-bit synchronized.
- **Tooling & Quality Gates**: Adds `pyinstaller` to build tooling and verifies all gates (`scripts/check.py`) pass cleanly.
