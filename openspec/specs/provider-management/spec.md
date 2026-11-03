# provider-management Specification

## Purpose

Provides configuration, persistence, and runtime dispatch for custom AI providers and models, enabling arbitrary LLM and embedding choices with decoupled endpoint controls.

## Requirements

### Requirement: Custom Provider Persistence

The system SHALL persist an unlimited list of custom AI providers in settings, each specifying a provider name, endpoint URL, API key, default LLM model, and default embedding model.

#### Scenario: Add new custom provider

- **GIVEN** a user configures a new provider with name "Ollama Local", endpoint "<http://localhost:11434/v1>", and model "llama3.2"
- **WHEN** settings are saved
- **THEN** the custom provider is stored in settings and becomes available in the provider selection list

#### Scenario: Delete custom provider

- **GIVEN** a custom provider exists in settings
- **WHEN** the user deletes the custom provider and saves settings
- **THEN** the provider is removed from settings and no longer appears in provider lists

### Requirement: Free-Form LLM Model Selection

The system SHALL allow users to input any valid model name for both built-in and custom providers, offering recommended presets without restricting input to predefined options.

#### Scenario: Enter unlisted model for built-in provider

- **GIVEN** the active provider is "OpenAI"
- **WHEN** the user enters an unlisted model name such as "gpt-4.5-preview"
- **THEN** the system accepts and activates the model without validation error

#### Scenario: Select suggested model

- **GIVEN** the active provider is "DeepSeek"
- **WHEN** the user selects "deepseek-chat" from suggestions
- **THEN** the model is set to "deepseek-chat"

### Requirement: Decoupled Embedding Configuration with Sync Toggle

The system SHALL maintain independent settings for the embedding model, embedding endpoint, and embedding API key, and SHALL provide a synchronization toggle that mirrors LLM endpoint and credentials when enabled.

#### Scenario: Synchronization toggle enabled

- **GIVEN** the synchronization toggle "same as LLM" is enabled
- **WHEN** the user changes the LLM provider or LLM endpoint
- **THEN** the embedding endpoint and API key automatically mirror the LLM endpoint and key

#### Scenario: Synchronization toggle disabled for independent embeddings

- **GIVEN** the synchronization toggle is disabled
- **WHEN** the user configures an external or local embedding model and distinct endpoint
- **THEN** the system retains the separate embedding endpoint and credentials without overriding them from LLM settings

### Requirement: OpenAI-Compatible Runtime Dispatch

The system SHALL dispatch requests for custom providers and custom embedding endpoints to OpenAI-compatible chat and embedding interfaces.

#### Scenario: Dispatch custom LLM

- **GIVEN** the active provider is a custom provider with an endpoint URL
- **WHEN** the backend initializes the language model
- **THEN** it instantiates an OpenAI-compatible chat client configured with that endpoint, model name, and API key

#### Scenario: Dispatch custom embeddings

- **GIVEN** an embedding endpoint URL is configured
- **WHEN** the backend initializes embeddings
- **THEN** it sends vectorization requests to the `<endpoint>/embeddings` endpoint with the specified embedding model name
