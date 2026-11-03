# Tasks

## 1. Backend Core & Tests (TDD)

- [ ] 1.1 Write unit tests translating GIVEN/WHEN/THEN criteria from `specs/provider-management/spec.md` into `tests/test_provider_management.py` and verify tests fail against current implementation
- [ ] 1.2 Implement `OpenAICompatibleEmbeddings` class and update `_create_embeddings` with endpoint and custom model support in `web_gui_pageoracle/main.py` and `tkinter_gui/main.py`
- [ ] 1.3 Implement custom provider dispatch in `set_model` (`ChatOpenAI` with `openai_api_base`) and remove rigid model whitelist checks for built-in providers in `web_gui_pageoracle/main.py` and `tkinter_gui/main.py`
- [ ] 1.4 Verify `test_backends_are_bit_for_bit_identical` and all provider management backend unit tests pass with `python -m unittest discover -s tests -p "test_*.py"`

## 2. Web GUI Implementation

- [ ] 2.1 Update `load_settings`, `save_settings`, and `_validate_settings` in `web_gui_pageoracle/gui.py` to support `custom_providers`, `llm_endpoint`, `embedding_endpoint`, `embedding_same_as_llm`, and free-form model strings
- [ ] 2.2 Update `web_gui_pageoracle/templates/index.html` with custom providers management view, HTML5 datalists for LLM and embedding models, and embedding synchronization toggle
- [ ] 2.3 Update `web_gui_pageoracle/static/app.js` and `web_gui_pageoracle/static/style.css` to handle custom provider CRUD, dynamic datalist population, and endpoint sync toggling

## 3. Desktop Tkinter GUI Implementation

- [ ] 3.1 Update `SettingsWindow` in `tkinter_gui/gui.py` to use editable comboboxes (`state="normal"`) for LLM and embedding models, permitting arbitrary model text input
- [ ] 3.2 Add custom providers management dialog in `tkinter_gui/gui.py` to add, edit, and delete custom provider profiles
- [ ] 3.3 Add embedding synchronization toggle and dedicated embedding endpoint and API key inputs in `tkinter_gui/gui.py`

## 4. Quality Gates & Final Integration

- [ ] 4.1 Run full test suite with `python -m unittest discover -s tests -p "test_*.py"` and verify all tests pass
- [ ] 4.2 Run `python scripts/check.py` and verify all linters and quality gate checks exit with code 0
