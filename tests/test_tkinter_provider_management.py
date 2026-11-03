"""
Unit tests for Tkinter GUI custom providers, free-form models, and embedding sync.
Translates GIVEN/WHEN/THEN criteria from specs/provider-management/spec.md for Tkinter GUI (Tasks 3.1, 3.2, 3.3).
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from tkinter import Tk
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tkinter_gui"))

import tkinter_gui.gui as tk_gui  # noqa: E402
from tkinter_gui.main import PROVIDERS  # noqa: E402


class TestTkinterProviderManagement(unittest.TestCase):
    """Test suite for Tkinter GUI SettingsWindow, CustomProvidersDialog, and settings persistence."""

    def setUp(self) -> None:
        try:
            self.root = Tk()
            self.root.withdraw()
        except Exception as e:
            self.skipTest(f"Tkinter display not available: {e}")

    def tearDown(self) -> None:
        try:
            self.root.update_idletasks()
            self.root.destroy()
        except Exception:
            pass

    # ── Task 3.1: Free-Form Model Comboboxes ─────────────────────────────

    def test_settings_window_model_comboboxes_are_editable(self) -> None:
        """
        GIVEN SettingsWindow is initialized
        WHEN we inspect model_cb and embedding_cb
        THEN their state is 'normal' (editable), permitting arbitrary model input
        """
        current_settings = {
            "provider": "OpenAI",
            "model": "gpt-4.5-preview",
            "embedding_model": "text-embedding-3-small",
            "llm_api_key": "sk-test",
            "embedding_api_key": "sk-test",
            "temperature": 0.2,
            "max_tokens": 4096,
            "top_p": 0.9,
            "score_threshold": 0.6,
        }
        saved_data = {}

        def on_save(data: dict) -> None:
            saved_data.update(data)

        win = tk_gui.SettingsWindow(self.root, current_settings, PROVIDERS, on_save)
        try:
            # Both model and embedding comboboxes must have state == 'normal'
            self.assertEqual(str(win.model_cb.cget("state")), "normal")
            self.assertEqual(str(win.embedding_cb.cget("state")), "normal")

            # Must allow free-form custom model strings without being overwritten
            win.model_var.set("gpt-4.5-custom-fine-tuned")
            win.embedding_var.set("my-custom-bge-embed")

            # Calling _save should preserve the custom model names
            with patch.object(tk_gui, "save_settings"):
                win._save()

            self.assertEqual(saved_data.get("model"), "gpt-4.5-custom-fine-tuned")
            self.assertEqual(saved_data.get("embedding_model"), "my-custom-bge-embed")
        finally:
            win.win.destroy()

    # ── Task 3.2: Custom Providers Management Dialog ─────────────────────

    def test_custom_providers_dialog_crud(self) -> None:
        """
        GIVEN CustomProvidersDialog with an initial list
        WHEN adding, editing, and deleting a provider
        THEN the custom providers list is updated accordingly
        """
        initial_providers = [
            {
                "name": "Ollama Local",
                "endpoint": "http://localhost:11434/v1",
                "api_key": "",
                "model": "llama3.2",
                "embedding_model": "nomic-embed-text",
            }
        ]
        updated_providers: list[dict] = []

        def on_update(providers: list[dict]) -> None:
            nonlocal updated_providers
            updated_providers = list(providers)

        # Dialog must exist and be instantiable
        self.assertTrue(
            hasattr(tk_gui, "CustomProvidersDialog"),
            "tk_gui must implement CustomProvidersDialog",
        )
        dialog = tk_gui.CustomProvidersDialog(self.root, initial_providers, on_update)
        try:
            # 1. Add new provider
            dialog.name_entry.delete(0, "end")
            dialog.name_entry.insert(0, "vLLM Remote")
            dialog.endpoint_entry.delete(0, "end")
            dialog.endpoint_entry.insert(0, "http://remote:8000/v1")
            dialog.api_key_entry.delete(0, "end")
            dialog.api_key_entry.insert(0, "vllm-secret")
            dialog.model_entry.delete(0, "end")
            dialog.model_entry.insert(0, "qwen2.5-coder")
            dialog.embedding_model_entry.delete(0, "end")
            dialog.embedding_model_entry.insert(0, "bge-m3")

            dialog._add_provider()
            names = [p["name"] for p in dialog.custom_providers]
            self.assertIn("vLLM Remote", names)

            # 2. Edit existing provider
            dialog._select_provider_by_index(0)  # Ollama Local
            dialog.model_entry.delete(0, "end")
            dialog.model_entry.insert(0, "llama3.3")
            dialog._update_selected_provider()

            ollama = next(p for p in dialog.custom_providers if p["name"] == "Ollama Local")
            self.assertEqual(ollama["model"], "llama3.3")

            # 3. Delete provider
            dialog._select_provider_by_index(0)  # Ollama Local
            dialog._delete_provider()
            remaining_names = [p["name"] for p in dialog.custom_providers]
            self.assertNotIn("Ollama Local", remaining_names)
            self.assertIn("vLLM Remote", remaining_names)

            # 4. Save and close
            dialog._on_ok()
            self.assertEqual(len(updated_providers), 1)
            self.assertEqual(updated_providers[0]["name"], "vLLM Remote")
        finally:
            dialog.win.destroy()

    def test_settings_window_custom_provider_selection_populates_fields(self) -> None:
        """
        GIVEN SettingsWindow with configured custom providers
        WHEN a custom provider is selected
        THEN its endpoint, model, and embedding defaults are populated
        """
        custom_providers = [
            {
                "name": "Local vLLM",
                "endpoint": "http://localhost:8000/v1",
                "api_key": "vllm-key",
                "model": "qwen2.5-coder",
                "embedding_model": "bge-large-en",
            }
        ]
        current_settings = {
            "provider": "Local vLLM",
            "model": "qwen2.5-coder",
            "embedding_model": "bge-large-en",
            "llm_endpoint": "http://localhost:8000/v1",
            "llm_api_key": "vllm-key",
            "embedding_api_key": "",
            "custom_providers": custom_providers,
        }
        win = tk_gui.SettingsWindow(self.root, current_settings, PROVIDERS, lambda _: None)
        try:
            self.assertIn("Local vLLM", win.provider_cb["values"])
            win.provider_var.set("Local vLLM")
            win._on_provider_change()

            self.assertEqual(win.llm_endpoint_entry.get().strip(), "http://localhost:8000/v1")
            self.assertEqual(win.model_var.get().strip(), "qwen2.5-coder")
        finally:
            win.win.destroy()

    # ── Task 3.3: Embedding Sync Toggle & Dedicated Inputs ────────────────

    def test_settings_window_sync_toggle_enabled(self) -> None:
        """
        GIVEN sync toggle is enabled in SettingsWindow
        WHEN user enters LLM endpoint and LLM API key and saves
        THEN saved settings mirror LLM endpoint and key to embedding endpoint and key
        """
        current_settings = {
            "provider": "Custom",
            "model": "mistral-large",
            "embedding_model": "text-embedding-3-small",
            "llm_endpoint": "http://custom-llm:8000/v1",
            "llm_api_key": "secret-llm-key",
            "embedding_endpoint": "",
            "embedding_api_key": "",
            "embedding_same_as_llm": True,
            "temperature": 0.2,
            "max_tokens": 4096,
            "top_p": 0.9,
            "score_threshold": 0.6,
        }
        saved_data = {}

        def on_save(data: dict) -> None:
            saved_data.update(data)

        win = tk_gui.SettingsWindow(self.root, current_settings, PROVIDERS, on_save)
        try:
            self.assertTrue(win.sync_embeddings_var.get())
            with patch.object(tk_gui, "save_settings"):
                win._save()

            self.assertTrue(saved_data.get("embedding_same_as_llm"))
            self.assertEqual(saved_data.get("embedding_endpoint"), "http://custom-llm:8000/v1")
            self.assertEqual(saved_data.get("embedding_api_key"), "secret-llm-key")
        finally:
            win.win.destroy()

    def test_settings_window_sync_toggle_disabled_allows_independent_embeddings(self) -> None:
        """
        GIVEN sync toggle is disabled in SettingsWindow
        WHEN user configures separate embedding endpoint and key
        THEN saved settings preserve independent embedding endpoint and key
        """
        current_settings = {
            "provider": "Custom",
            "model": "llama3.2",
            "embedding_model": "bge-m3",
            "llm_endpoint": "http://llm-server:8000/v1",
            "llm_api_key": "llm-key",
            "embedding_endpoint": "http://embed-server:9000/v1",
            "embedding_api_key": "embed-key-xyz",
            "embedding_same_as_llm": False,
            "temperature": 0.2,
            "max_tokens": 4096,
            "top_p": 0.9,
            "score_threshold": 0.6,
        }
        saved_data = {}

        def on_save(data: dict) -> None:
            saved_data.update(data)

        win = tk_gui.SettingsWindow(self.root, current_settings, PROVIDERS, on_save)
        try:
            self.assertFalse(win.sync_embeddings_var.get())
            with patch.object(tk_gui, "save_settings"):
                win._save()

            self.assertFalse(saved_data.get("embedding_same_as_llm"))
            self.assertEqual(saved_data.get("embedding_endpoint"), "http://embed-server:9000/v1")
            self.assertEqual(saved_data.get("embedding_api_key"), "embed-key-xyz")
        finally:
            win.win.destroy()

    # ── Persistence Schema & Migration ───────────────────────────────────

    def test_tkinter_load_and_save_settings_schema(self) -> None:
        """
        GIVEN tk_gui.load_settings and save_settings
        WHEN settings contain custom_providers, endpoints, and sync flag
        THEN they are properly persisted and loaded
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = Path(tmp_dir) / "settings.json"
            with patch.object(tk_gui, "SETTINGS_FILE", settings_path):
                defaults = tk_gui.load_settings()
                self.assertIn("custom_providers", defaults)
                self.assertIn("llm_endpoint", defaults)
                self.assertIn("embedding_endpoint", defaults)
                self.assertIn("embedding_same_as_llm", defaults)

                test_payload = dict(defaults)
                test_payload["custom_providers"] = [{"name": "Local Ollama", "endpoint": "http://localhost:11434/v1"}]
                test_payload["llm_endpoint"] = "http://localhost:11434/v1"
                test_payload["embedding_endpoint"] = "http://localhost:11434/v1"
                test_payload["embedding_same_as_llm"] = True
                tk_gui.save_settings(test_payload)

                reloaded = tk_gui.load_settings()
                self.assertEqual(len(reloaded["custom_providers"]), 1)
                self.assertEqual(reloaded["custom_providers"][0]["name"], "Local Ollama")
                self.assertEqual(reloaded["llm_endpoint"], "http://localhost:11434/v1")
                self.assertTrue(reloaded["embedding_same_as_llm"])


if __name__ == "__main__":
    unittest.main()
