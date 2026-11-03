"""
Unit tests for custom AI providers and model management.
Translates GIVEN/WHEN/THEN criteria from specs/provider-management/spec.md.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "web_gui_pageoracle"))

import tkinter_gui.main as tk_main  # noqa: E402
import web_gui_pageoracle.gui as web_gui  # noqa: E402
import web_gui_pageoracle.main as web_main  # noqa: E402


class TestProviderManagement(unittest.TestCase):
    """Test suite verifying custom provider persistence, free-form models, and runtime dispatch."""

    # ── Requirement: Custom Provider Persistence ─────────────────────────

    def test_add_and_persist_custom_provider(self) -> None:
        """
        GIVEN a user configures a new provider with name "Ollama Local",
              endpoint "http://localhost:11434/v1", and model "llama3.2"
        WHEN settings are saved and reloaded
        THEN the custom provider is stored in settings and available in custom_providers
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = Path(tmp_dir) / "settings.json"
            with patch.object(web_gui, "SETTINGS_FILE", settings_path):
                provider_data = {
                    "name": "Ollama Local",
                    "endpoint": "http://localhost:11434/v1",
                    "api_key": "",
                    "model": "llama3.2",
                    "embedding_model": "nomic-embed-text",
                }
                settings = web_gui.load_settings()
                settings["custom_providers"] = [provider_data]
                web_gui.save_settings(settings)

                reloaded = web_gui.load_settings()
                self.assertIn("custom_providers", reloaded)
                self.assertEqual(len(reloaded["custom_providers"]), 1)
                self.assertEqual(reloaded["custom_providers"][0]["name"], "Ollama Local")
                self.assertEqual(reloaded["custom_providers"][0]["endpoint"], "http://localhost:11434/v1")
                self.assertEqual(reloaded["custom_providers"][0]["model"], "llama3.2")

    def test_delete_custom_provider(self) -> None:
        """
        GIVEN a custom provider exists in settings
        WHEN the user deletes the custom provider and saves settings
        THEN the provider is removed from settings
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = Path(tmp_dir) / "settings.json"
            with patch.object(web_gui, "SETTINGS_FILE", settings_path):
                settings = web_gui.load_settings()
                settings["custom_providers"] = [
                    {
                        "name": "Ollama Local",
                        "endpoint": "http://localhost:11434/v1",
                        "api_key": "",
                        "model": "llama3.2",
                        "embedding_model": "nomic-embed-text",
                    }
                ]
                web_gui.save_settings(settings)

                # Delete the provider
                settings["custom_providers"] = []
                web_gui.save_settings(settings)

                reloaded = web_gui.load_settings()
                self.assertEqual(reloaded.get("custom_providers"), [])

    # ── Requirement: Free-Form LLM Model Selection ────────────────────────

    def test_enter_unlisted_model_for_builtin_provider(self) -> None:
        """
        GIVEN the active provider is "OpenAI"
        WHEN the user enters an unlisted model name such as "gpt-4.5-preview"
        THEN the system accepts and validates the model without error
        """
        payload = {
            "provider": "OpenAI",
            "model": "gpt-4.5-preview",
            "embedding_model": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
            "llm_api_key": "sk-proj-test1234",
            "embedding_api_key": "sk-or-test1234",
            "temperature": 0.2,
            "max_tokens": 4096,
            "top_p": 0.9,
            "score_threshold": 0.6,
        }
        validated = web_gui._validate_settings(payload)
        self.assertEqual(validated["model"], "gpt-4.5-preview")

    def test_select_suggested_model(self) -> None:
        """
        GIVEN the active provider is "DeepSeek"
        WHEN the user selects "deepseek-chat" from suggestions
        THEN the model is set to "deepseek-chat"
        """
        payload = {
            "provider": "DeepSeek",
            "model": "deepseek-chat",
            "embedding_model": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
            "llm_api_key": "sk-deepseek-test",
            "embedding_api_key": "sk-or-test",
            "temperature": 0.2,
            "max_tokens": 4096,
            "top_p": 0.9,
            "score_threshold": 0.6,
        }
        validated = web_gui._validate_settings(payload)
        self.assertEqual(validated["model"], "deepseek-chat")

    # ── Requirement: Decoupled Embedding Configuration with Sync Toggle ───

    def test_embedding_sync_toggle_enabled(self) -> None:
        """
        GIVEN the synchronization toggle "same as LLM" is enabled
        WHEN the user configures LLM endpoint and credentials
        THEN embedding endpoint and API key mirror LLM endpoint and key
        """
        payload = {
            "provider": "Custom",
            "model": "mistral-large",
            "llm_endpoint": "https://custom-llm.example.com/v1",
            "llm_api_key": "key-abc-123",
            "embedding_same_as_llm": True,
            "embedding_model": "text-embedding-3-small",
            "embedding_endpoint": "",
            "embedding_api_key": "",
            "temperature": 0.2,
            "max_tokens": 4096,
            "top_p": 0.9,
            "score_threshold": 0.6,
        }
        validated = web_gui._validate_settings(payload)
        self.assertTrue(validated["embedding_same_as_llm"])
        self.assertEqual(validated["embedding_endpoint"], "https://custom-llm.example.com/v1")
        self.assertEqual(validated["embedding_api_key"], "key-abc-123")

    def test_embedding_sync_toggle_disabled_retains_independent_values(self) -> None:
        """
        GIVEN the synchronization toggle is disabled
        WHEN the user configures an external embedding model and distinct endpoint
        THEN the system retains the separate embedding endpoint and credentials
        """
        payload = {
            "provider": "Custom",
            "model": "llama3.2",
            "llm_endpoint": "http://localhost:11434/v1",
            "llm_api_key": "",
            "embedding_same_as_llm": False,
            "embedding_model": "bge-large-en-v1.5",
            "embedding_endpoint": "http://localhost:8080/v1",
            "embedding_api_key": "emb-key-999",
            "temperature": 0.2,
            "max_tokens": 4096,
            "top_p": 0.9,
            "score_threshold": 0.6,
        }
        validated = web_gui._validate_settings(payload)
        self.assertFalse(validated["embedding_same_as_llm"])
        self.assertEqual(validated["embedding_endpoint"], "http://localhost:8080/v1")
        self.assertEqual(validated["embedding_api_key"], "emb-key-999")

    def test_validate_settings_custom_provider_requires_endpoint(self) -> None:
        """
        GIVEN active provider is "Custom" without any endpoint URL configured
        WHEN settings validation is run
        THEN a ValueError is raised requiring an endpoint
        """
        payload = {
            "provider": "Custom",
            "model": "llama3.2",
            "llm_endpoint": "",
            "embedding_model": "bge-m3",
            "temperature": 0.2,
            "max_tokens": 4096,
            "top_p": 0.9,
            "score_threshold": 0.6,
        }
        with self.assertRaises(ValueError) as ctx:
            web_gui._validate_settings(payload)
        self.assertIn("Endpoint", str(ctx.exception))

    def test_validate_settings_matching_custom_provider_resolves_defaults(self) -> None:
        """
        GIVEN a saved custom provider "Local vLLM" with default endpoint and model
        WHEN the user selects "Local vLLM" without providing an explicit endpoint or model
        THEN validation populates endpoint and model from the saved custom provider
        """
        payload = {
            "provider": "Local vLLM",
            "model": "",
            "llm_endpoint": "",
            "embedding_model": "",
            "custom_providers": [
                {
                    "name": "Local vLLM",
                    "endpoint": "http://localhost:8000/v1",
                    "api_key": "vllm-key",
                    "model": "qwen2.5-coder",
                    "embedding_model": "bge-large-en",
                }
            ],
            "temperature": 0.2,
            "max_tokens": 4096,
            "top_p": 0.9,
            "score_threshold": 0.6,
        }
        validated = web_gui._validate_settings(payload)
        self.assertEqual(validated["model"], "qwen2.5-coder")
        self.assertEqual(validated["llm_endpoint"], "http://localhost:8000/v1")
        self.assertEqual(validated["llm_api_key"], "vllm-key")
        self.assertEqual(validated["embedding_model"], "bge-large-en")

    def test_load_settings_legacy_endpoint_migration(self) -> None:
        """
        GIVEN a settings.json file containing legacy "endpoint" key
        WHEN load_settings is called
        THEN "llm_endpoint" is populated with the legacy endpoint value
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = Path(tmp_dir) / "settings.json"
            settings_path.write_text(
                json.dumps({"provider": "Custom", "endpoint": "http://legacy:8000/v1"}),
                encoding="utf-8",
            )
            with patch.object(web_gui, "SETTINGS_FILE", settings_path):
                settings = web_gui.load_settings()
                self.assertEqual(settings.get("llm_endpoint"), "http://legacy:8000/v1")

    def test_api_settings_route(self) -> None:
        """
        GIVEN valid settings payload with custom provider and sync toggle
        WHEN POST /api/settings is called on Flask test client
        THEN it returns 200 with ok=True and updates persisted settings
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = Path(tmp_dir) / "settings.json"
            with patch.object(web_gui, "SETTINGS_FILE", settings_path):
                client = web_gui.app.test_client()
                payload = {
                    "provider": "Ollama",
                    "model": "deepseek-r1",
                    "llm_endpoint": "http://localhost:11434/v1",
                    "llm_api_key": "",
                    "embedding_same_as_llm": True,
                    "embedding_model": "nomic-embed-text",
                    "custom_providers": [
                        {
                            "name": "Ollama",
                            "endpoint": "http://localhost:11434/v1",
                            "api_key": "",
                            "model": "deepseek-r1",
                            "embedding_model": "nomic-embed-text",
                        }
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2048,
                    "top_p": 0.8,
                    "score_threshold": 0.5,
                }
                # Mock backend to avoid initializing real models during HTTP route test
                with patch.object(web_gui.state, "backend") as mock_backend:
                    mock_backend.set_model.return_value = True
                    mock_backend.set_embeddings.return_value = True
                    mock_backend.set_score_threshold.return_value = True
                    mock_backend.embedding_model_name = "nomic-embed-text"

                    resp = client.post("/api/settings", json=payload)
                    self.assertEqual(resp.status_code, 200)
                    data = resp.get_json()
                    self.assertTrue(data.get("ok"))

    # ── Requirement: OpenAI-Compatible Runtime Dispatch ───────────────────

    def test_dispatch_custom_llm(self) -> None:
        """
        GIVEN the active provider is a custom provider or endpoint
        WHEN the backend sets the model
        THEN it instantiates an OpenAI-compatible chat client configured with that endpoint and model
        """
        backend = web_main.PageOracleBackend(books_dir=".")
        # Mock langchain_openai.ChatOpenAI
        with patch("langchain_openai.ChatOpenAI") as mock_chat_openai:
            mock_instance = MagicMock()
            mock_chat_openai.return_value = mock_instance

            success = backend.set_model(
                provider_name="Custom",
                model_name="my-custom-llm",
                api_key="custom-secret-key",
                endpoint="http://localhost:11434/v1",
            )
            self.assertTrue(success)
            mock_chat_openai.assert_called_once()
            _, kwargs = mock_chat_openai.call_args
            self.assertEqual(kwargs.get("model"), "my-custom-llm")
            self.assertEqual(kwargs.get("openai_api_base"), "http://localhost:11434/v1")

    def test_dispatch_custom_embeddings_class(self) -> None:
        """
        GIVEN an embedding endpoint URL and model name
        WHEN OpenAICompatibleEmbeddings sends vectorization requests
        THEN it posts to <endpoint>/embeddings and returns parsed float vectors
        """
        self.assertTrue(
            hasattr(web_main, "OpenAICompatibleEmbeddings"),
            "web_main must implement OpenAICompatibleEmbeddings",
        )
        embeddings_cls = web_main.OpenAICompatibleEmbeddings
        emb = embeddings_cls(
            model_name="custom-embed-model",
            endpoint="http://localhost:11434/v1",
            api_key="test-api-key",
        )

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "data": [
                    {"embedding": [0.1, 0.2, 0.3]},
                    {"embedding": [0.4, 0.5, 0.6]},
                ]
            }
        ).encode("utf-8")

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = mock_response

            vectors = emb.embed_documents(["hello world", "another text"])
            self.assertEqual(len(vectors), 2)
            self.assertEqual(vectors[0], [0.1, 0.2, 0.3])
            self.assertEqual(vectors[1], [0.4, 0.5, 0.6])

            # Check requested URL
            req = mock_urlopen.call_args[0][0]
            self.assertEqual(req.full_url, "http://localhost:11434/v1/embeddings")
            self.assertEqual(req.headers.get("Authorization"), "Bearer test-api-key")

    def test_backend_create_embeddings_custom(self) -> None:
        """
        GIVEN an unlisted embedding model and/or custom endpoint
        WHEN backend._create_embeddings is called
        THEN it returns an OpenAI-compatible embeddings wrapper and does not raise an unknown model error
        """
        backend = web_main.PageOracleBackend(books_dir=".")
        emb = backend._create_embeddings(
            embedding_model="text-embedding-3-small",
            api_key="sk-test",
            endpoint="https://api.openai.com/v1",
        )
        self.assertIsNotNone(emb)
        self.assertEqual(backend.embedding_model_name, "text-embedding-3-small")

        # Verify tk_main as well
        tk_backend = tk_main.PageOracleBackend(books_dir=".")
        tk_emb = tk_backend._create_embeddings(
            embedding_model="text-embedding-3-small",
            api_key="sk-test",
            endpoint="https://api.openai.com/v1",
        )
        self.assertIsNotNone(tk_emb)
        self.assertEqual(tk_backend.embedding_model_name, "text-embedding-3-small")


if __name__ == "__main__":
    unittest.main()
