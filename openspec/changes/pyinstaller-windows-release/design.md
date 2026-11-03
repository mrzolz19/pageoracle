# Design

## Context

PageOracle contains a desktop Tkinter GUI (`tkinter_gui/gui.py`) backed by `PageOracleBackend` (`tkinter_gui/main.py`), utilizing LangChain, LangGraph, ChromaDB, SentenceTransformers, and PyTorch. The application currently expects local execution from source, resolving relative paths against the current working directory or script directory. GitHub Actions hosted runners (`windows-latest`) have strict disk space constraints (~14 GB) and do not provide NVIDIA GPUs.

## Goals / Non-Goals

**Goals:**

- Provide a robust, reproducible PyInstaller build specification (`pageoracle.spec`) in the repository root.
- Implement runtime path resolution in `tkinter_gui` that keeps user configuration, book library, vector store, and chat history portable and persistent adjacent to `PageOracle.exe`.
- Implement a GitHub Actions workflow (`.github/workflows/release.yml`) with a dual-target matrix producing both `PageOracle-windows-x64-cpu.zip` and `PageOracle-windows-x64-cu128.zip`.
- Support execution triggered by git tag pushes (`v*`) and manual triggers (`workflow_dispatch`).
- Ensure all quality gates, including `jactionlint`, pass cleanly.

**Non-Goals:**

- Packaging the Flask web GUI (`web_gui_pageoracle`) into the desktop bundle.
- Building macOS or Linux standalone binaries.
- Windows Authenticode EV code signing (binaries will be unsigned open-source releases).

## Decisions

### 1. Packaging Format: `--onedir` with Root Archive

- **Chosen:** Build a directory distribution (`COLLECT` in PyInstaller) and compress it to `.zip` for release distribution.
- **Alternatives Considered:** `--onefile` standalone `.exe`.
- **Rationale:** `--onefile` decompresses gigabytes of PyTorch and CUDA binaries to `%TEMP%` on every single run, leading to 20–40s startup delays, risk of temp directory filling up, and loss of local mutable state. `--onedir` starts in 1–2 seconds and allows users to manage their `books/` folder directly in Windows Explorer.

### 2. Runtime Path Anchoring (`sys.frozen`)

- **Chosen:** Adapt `tkinter_gui/gui.py` and `tkinter_gui/main.py` to evaluate:

  ```python
  def get_base_dir() -> Path:
      if getattr(sys, "frozen", False):
          return Path(sys.executable).resolve().parent
      return Path(__file__).resolve().parent
  ```

- **Rationale:** Prevents application files (`settings.json`, `chroma_intro_db/`, `books/`, `chat_history.json`) from being routed into temporary internal directories (`_internal/`) or unrelated shell directories when launched from desktop shortcuts. Keeps `web_gui_pageoracle/main.py` in exact synchronization.

### 3. Dual Matrix Build in GitHub Actions

- **Chosen:** Matrix strategy in `.github/workflows/release.yml`:
  - `variant: cpu`: installs PyTorch CPU wheels (`--index-url https://download.pytorch.org/whl/cpu`), producing `PageOracle-windows-x64-cpu.zip` (~250 MB).
  - `variant: cu128`: installs PyTorch CUDA 12.8 wheels per `tkinter_gui/requirements.txt`, producing `PageOracle-windows-x64-cu128.zip` (~1.3 GB).
- **Rationale:** Solves both the bandwidth/compatibility problem for standard PC users and provides GPU acceleration for users with NVIDIA hardware. Matrix separation gives each build job a clean runner with full disk capacity, preventing disk overflow.

### 4. Root `pageoracle.spec` with Comprehensive Hook Collection

- **Chosen:** Root `pageoracle.spec` using PyInstaller utility functions:
  - `collect_data_files('chromadb')` and `collect_submodules('chromadb')` for migrations and segment storage.
  - `collect_submodules` for `langchain`, `langchain_community`, `langgraph`, and `sentence_transformers`.
  - `collect_data_files('certifi')` to ensure HTTPS certificates are present for cloud LLM APIs.
  - `copy_metadata('tqdm')`, `copy_metadata('regex')`, `copy_metadata('requests')`, `copy_metadata('torch')`.
  - Exclude unused packages: `flask`, `jinja2`, `werkzeug`, `pytest`, `IPython`, `notebook`, `matplotlib`.

## Risks / Trade-offs

- **[Disk space limits on GitHub Actions runner]** -> Use separate matrix jobs so CPU and CUDA don't share disk; use `pip install --no-cache-dir`.
- **[HuggingFace CrossEncoder first-run download]** -> Model weights are downloaded to user cache on first rerank call. The application handles download progress in the UI and logs status gracefully.
- **[Antivirus false positives]** -> PyInstaller generated binaries are sometimes flagged by heuristic Windows Defender scanners. Documented in release notes with instructions for exclusion if encountered.
