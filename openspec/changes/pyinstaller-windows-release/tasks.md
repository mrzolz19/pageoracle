# Tasks

## 1. Runtime Path Anchoring & Persistence

- [ ] 1.1 Add test in `tests/test_desktop_distribution.py` verifying runtime path resolution under normal vs frozen (`sys.frozen`) execution modes
- [ ] 1.2 Implement frozen path resolution in `tkinter_gui/gui.py` and `tkinter_gui/main.py` anchoring settings, books, ChromaDB, and chat history adjacent to executable
- [ ] 1.3 Mirror necessary path resolution adjustments into `web_gui_pageoracle/main.py` maintaining bit-for-bit backend synchronization and verify tests pass

## 2. PyInstaller Specification

- [ ] 2.1 Create root `pageoracle.spec` configuring `--onedir` collection for `tkinter_gui/gui.py`, hidden imports, metadata collection, and exclusions
- [ ] 2.2 Add unit test validating `pageoracle.spec` configuration, analysis scripts, data files, and hidden import completeness

## 3. GitHub Actions CI/CD Release Workflow

- [ ] 3.1 Create `.github/workflows/release.yml` with dual-variant matrix (`cpu` and `cu128`) triggered by tags `v*` and `workflow_dispatch`
- [ ] 3.2 Verify `.github/workflows/release.yml` with `jactionlint` and run aggregate quality gates via `python scripts/check.py`

## 4. Documentation and Decision Log

- [ ] 4.1 Record decision in `docs/DECISIONS.md` documenting the standalone packaging architecture and dual CPU/CUDA release matrix
- [ ] 4.2 Update `README.md` with download, extraction, and usage instructions for the standalone Windows release packages
