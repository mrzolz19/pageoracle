# Design

## Context

PageOracle currently employs hardcoded mappings in `web_gui_pageoracle/main.py` and `tkinter_gui/main.py` (`PROVIDERS` and `EMBEDDING_MODELS`). The settings schema stores a single provider string, model name, and embedding model name chosen from rigid comboboxes/selects.

This design introduces a unified persistence schema for custom OpenAI-compatible providers, free-form model entry across all providers, and decoupled embedding configuration with a synchronization toggle, keeping both Web and Tkinter GUIs identical in capability and maintaining bit-for-bit backend synchronization (see Decision D-005 in `docs/DECISIONS.md`).

## Goals / Non-Goals

**Goals:**

- Provide an extensible `custom_providers` collection in `settings.json` supporting unlimited user-defined OpenAI-compatible endpoints.
- Allow free-form text entry for LLM models and embedding models for both built-in and custom providers, while preserving preset suggestions.
- Decouple embedding settings (`embedding_endpoint`, `embedding_api_key`, `embedding_model`) with a UI toggle to mirror LLM settings when desired.
- Maintain bit-for-bit code synchronization between `web_gui_pageoracle/main.py` and `tkinter_gui/main.py`.
- Pass all automated quality gates (`ruff`, `ty`, `deptry`, `typos`, `rumdl`).

**Non-Goals:**

- Custom non-OpenAI proprietary protocols (e.g. Anthropic native wire protocol for custom endpoints); custom providers use standard OpenAI-compatible API (`/v1/chat/completions`, `/v1/embeddings`).
- Dynamic server capability probing or auto-download of local GGUF models.

## Decisions

### Decision 1: Persistence Schema in `settings.json`

`settings.json` will store:

- `custom_providers`: `list[dict]` where each item contains `name`, `endpoint`, `api_key`, `model`, `embedding_model`.
- `llm_endpoint`: `str` (endpoint for active provider if custom, or optional override).
- `embedding_same_as_llm`: `bool` (defaults to true for custom providers, false for legacy setups).
- `embedding_endpoint`: `str` (independent endpoint for embeddings).
- `embedding_model`: `str` (free-form model string).
- `embedding_api_key`: `str` (independent API key for embeddings).

*Alternatives considered:* Storing custom providers in a separate `providers.json`. Rejected to prevent desynchronization between active settings and available providers.

### Decision 2: OpenAI-Compatible Runtime Client Dispatch

- For custom LLMs: `ChatOpenAI(model=model_name, openai_api_base=endpoint, api_key=api_key or "not-needed", temperature=..., max_tokens=..., top_p=...)`.
- For custom embeddings: `OpenAICompatibleEmbeddings(model_name=model_name, endpoint=endpoint, api_key=api_key)` implemented with standard library `urllib.request` (analogous to the proven `OpenRouterEmbeddings`), posting batches of inputs to `<endpoint>/embeddings`.
- For built-in providers: bypass strict `models` whitelist checks, passing arbitrary user-specified model names directly to provider constructors.

*Alternatives considered:* Requiring `openai` official Python SDK. Rejected to avoid adding heavyweight dependencies when `urllib.request` handles `/embeddings` payloads robustly with zero dependency footprint.

### Decision 3: Frontend UI Patterns

- **Web GUI**: Replace `<select id="modelSelect">` and `<select id="embeddingSelect">` with `<input list="modelDatalist">` and `<datalist>`. Add modal sections for custom provider CRUD and an interactive checkbox `[X] Использовать тот же провайдер и endpoint для эмбеддингов` that toggles embedding endpoint/key visibility and synchronization.
- **Tkinter GUI**: Switch `self.model_cb` and `self.embedding_cb` to `state="normal"` (editable combobox). Add a manage-providers dialog button `[+ Кастомные...]` and a synchronization checkbutton to control embedding inputs.

## Risks / Trade-offs

- **[Risk] Custom endpoint offline or invalid URL** → **Mitigation**: Wrap connection and HTTP calls in descriptive try-except blocks, logging clear error messages to the UI status/log feed rather than crashing background threads.
- **[Risk] Dimension mismatch when switching custom embedding models** → **Mitigation**: Existing `_is_dimension_mismatch_error` and auto-purge in `_init_vectorstore()` already safely rebuilds the Chroma index when vector dimensions change.
- **[Risk] Dual-entrypoint desynchronization** → **Mitigation**: Enforcement via unit test `test_backends_are_bit_for_bit_identical` in `tests/test_book_indexing.py` and quality gate runner.
