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

---

## Archive

<!-- Superseded decisions moved here unchanged except for status: Superseded by D-<number> -->
