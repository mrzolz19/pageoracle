# book-indexing Specification

## Purpose

Provides automatic discovery and indexing of book files while excluding project dependency manifests and falling back to the adjacent library directory.

## Requirements

### Requirement: Exclusion of dependency manifests from book indexing

The backend SHALL ignore dependency manifest files such as requirements.txt when scanning for text documents to index into the vector store.

#### Scenario: requirements.txt is present in the scan directory

- **GIVEN** a directory containing `requirements.txt` and no literary book files
- **WHEN** automatic book loading is executed during initialization
- **THEN** the system SHALL NOT annotate, chunk, or index `requirements.txt` as a book

### Requirement: Fallback to adjacent books directory

When no valid text documents are found in the configured books directory, the backend SHALL search for and index text documents from the `books` folder located in the same directory as the backend module.

#### Scenario: books folder exists alongside module files

- **GIVEN** the configured books directory contains no eligible text books
- **GIVEN** a `books` directory exists in the same folder as the backend script containing `.txt` files
- **WHEN** automatic book loading is executed during initialization
- **THEN** the system SHALL load, annotate, and index the `.txt` files from that `books` directory
- **THEN** the system SHALL update its active books directory to point to that `books` directory
