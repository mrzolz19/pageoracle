# desktop-distribution Specification

## Purpose

Enables packaging PageOracle into standalone Windows executables with portable runtime persistence and automated CI/CD release publishing.

## Requirements

### Requirement: Standalone Application Packaging

The system SHALL provide a build specification to package the Tkinter desktop GUI into a portable Windows directory distribution containing the main executable and all runtime dependencies.

#### Scenario: Launch packaged executable without external Python

- **WHEN** user executes `PageOracle.exe` on a clean Windows machine without Python installed
- **THEN** the Tkinter desktop GUI opens and operates without missing dependency errors

### Requirement: Portable Runtime Path Resolution

The system SHALL resolve settings, book storage, vector persistence, and history adjacent to the executable when running in a frozen environment.

#### Scenario: Persistent settings across restarts

- **WHEN** user updates configuration in the packaged application and restarts `PageOracle.exe`
- **THEN** settings are loaded from the adjacent `settings.json` file rather than temporary extract directories

#### Scenario: Books directory resolution

- **WHEN** the packaged application initializes document loading
- **THEN** it searches the adjacent `books/` directory for `.txt` files to index

### Requirement: Dual Variant Release Packaging

The build system SHALL produce both CPU-only and CUDA-enabled Windows distribution archives.

#### Scenario: CPU release archive generation

- **WHEN** CI workflow executes the CPU build matrix job
- **THEN** it packages `PageOracle-windows-x64-cpu.zip` without NVIDIA CUDA dependencies

#### Scenario: CUDA release archive generation

- **WHEN** CI workflow executes the CUDA build matrix job
- **THEN** it packages `PageOracle-windows-x64-cu128.zip` with CUDA 12.8 GPU support

### Requirement: Automated Release Workflow

The system SHALL provide a GitHub Actions workflow that builds distribution packages and publishes them to GitHub Releases.

#### Scenario: Release triggered by tag push

- **WHEN** a git tag matching `v*` is pushed to the repository
- **THEN** the release workflow builds both distribution archives and attaches them to the GitHub Release

#### Scenario: Release triggered by manual workflow dispatch

- **WHEN** an operator triggers the workflow manually via `workflow_dispatch`
- **THEN** the release workflow builds the distribution packages and attaches them to draft or designated release assets
