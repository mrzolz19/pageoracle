# Proposal

## Why

Currently, PageOracle restricts LLM and embedding choices to hardcoded lists of providers and models. Users cannot connect their own local or remote OpenAI-compatible inference servers (such as Ollama, vLLM, LM Studio, or custom proxies) and cannot use unlisted models from existing providers (such as new OpenAI or OpenRouter models). Furthermore, embedding configuration is tightly coupled and rigid, preventing users from pairing custom LLMs with separate embedding endpoints or local models.

Providing custom provider profiles and free-form model entry empowers users to run PageOracle with any model ecosystem and server infrastructure while retaining simple presets.

## What Changes

- Add capability to define and persist an unlimited number of custom AI providers in `settings.json`, each with provider name, endpoint URL, API key, default LLM model, and default embedding model.
- Enable free-form text input for LLM models across both built-in and custom providers, replacing readonly selection with an editable list/combobox populated with suggestions.
- Decouple embedding configuration from the LLM provider, providing dedicated fields for embedding endpoint, embedding API key, and embedding model name (with suggestions and free-form entry).
- Add a synchronization toggle ("use same provider/endpoint for embeddings") in both Web and Desktop UIs to automatically mirror the LLM endpoint and credentials to embeddings when desired.
- Dispatch custom LLMs to `ChatOpenAI` with `openai_api_base` and custom embeddings to an OpenAI-compatible `/embeddings` HTTP client.
- Update both Web GUI (`web_gui_pageoracle`) and Tkinter Desktop GUI (`tkinter_gui`) with dedicated management views for custom providers while maintaining bit-for-bit backend synchronization.

## Capabilities

### New Capabilities

- `provider-management`: Defines configuration, persistence, validation, and runtime dispatch for custom OpenAI-compatible providers, arbitrary LLM/embedding model entry, and decoupled embedding endpoints with synchronization controls.

### Modified Capabilities

*(None. Existing `book-indexing` capability requirements remain unchanged.)*

## Impact

- **Configuration**: `settings.json` schema extended with `custom_providers`, `llm_endpoint`, `embedding_endpoint`, and `embedding_same_as_llm`.
- **Backend**: `web_gui_pageoracle/main.py` and `tkinter_gui/main.py` updated in exact sync to dispatch custom LLMs (`ChatOpenAI`) and custom embeddings (`OpenAICompatibleEmbeddings`).
- **Web Frontend**: `web_gui_pageoracle/templates/index.html`, `static/app.js`, `static/style.css`, and Flask API endpoints in `web_gui_pageoracle/gui.py`.
- **Tkinter Frontend**: `tkinter_gui/gui.py` `SettingsWindow` updated with custom provider management dialog, editable comboboxes, and embedding sync toggle.
- **Dependencies & Gates**: Uses existing standard dependencies (`langchain-openai`, `urllib.request`); passes all quality gates (`ruff`, `ty`, `deptry`, `typos`, `rumdl`).
