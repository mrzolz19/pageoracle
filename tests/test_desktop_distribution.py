"""
Unit tests for standalone desktop distribution and runtime path resolution.
Translates GIVEN/WHEN/THEN criteria from specs/desktop-distribution/spec.md (Tasks 1.1, 1.2, 1.3).
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tkinter_gui"))
sys.path.insert(0, str(REPO_ROOT / "web_gui_pageoracle"))

import tkinter_gui.gui as tk_gui  # noqa: E402
import tkinter_gui.main as tk_main  # noqa: E402
import web_gui_pageoracle.main as web_main  # noqa: E402


class TestDesktopDistribution(unittest.TestCase):
    """Test suite verifying frozen execution detection and adjacent path resolution."""

    def test_get_base_dir_normal_mode(self) -> None:
        """
        GIVEN application running under normal Python execution
        WHEN get_base_dir is called
        THEN it resolves to the parent directory of the source file
        """
        with patch.object(sys, "frozen", False, create=True):
            self.assertEqual(tk_main.get_base_dir(), (REPO_ROOT / "tkinter_gui").resolve())
            self.assertEqual(web_main.get_base_dir(), (REPO_ROOT / "web_gui_pageoracle").resolve())
            self.assertEqual(tk_gui.get_base_dir(), (REPO_ROOT / "tkinter_gui").resolve())

    def test_get_base_dir_frozen_mode(self) -> None:
        """
        GIVEN application running under frozen execution (sys.frozen = True)
        WHEN get_base_dir is called
        THEN it resolves to the parent directory of sys.executable
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_exe = Path(tmp_dir) / "PageOracle.exe"
            with (
                patch.object(sys, "frozen", True, create=True),
                patch.object(sys, "executable", str(fake_exe)),
            ):
                self.assertEqual(tk_main.get_base_dir(), Path(tmp_dir).resolve())
                self.assertEqual(web_main.get_base_dir(), Path(tmp_dir).resolve())
                self.assertEqual(tk_gui.get_base_dir(), Path(tmp_dir).resolve())

    def test_backend_path_resolution_normal_mode(self) -> None:
        """
        GIVEN PageOracleBackend initialized under normal Python execution
        WHEN no custom paths are provided
        THEN default relative paths are preserved
        """
        with patch.object(sys, "frozen", False, create=True):
            backend = tk_main.PageOracleBackend()
            self.assertEqual(backend.books_dir, ".")
            self.assertEqual(backend.persist_dir, "./chroma_intro_db")
            self.assertEqual(backend.history_path, Path("chat_history.json"))

    def test_backend_path_resolution_frozen_mode(self) -> None:
        """
        GIVEN PageOracleBackend initialized under frozen execution
        WHEN sys.frozen is True and sys.executable points to standalone exe
        THEN books, chroma, and history paths are anchored adjacent to executable
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dist_dir = Path(tmp_dir).resolve()
            fake_exe = dist_dir / "PageOracle.exe"
            books_dir = dist_dir / "books"
            books_dir.mkdir()

            with (
                patch.object(sys, "frozen", True, create=True),
                patch.object(sys, "executable", str(fake_exe)),
            ):
                backend = tk_main.PageOracleBackend()
                self.assertEqual(Path(backend.books_dir).resolve(), books_dir)
                self.assertEqual(Path(backend.persist_dir).resolve(), dist_dir / "chroma_intro_db")
                self.assertEqual(Path(backend.history_path).resolve(), dist_dir / "chat_history.json")

    def test_tkinter_gui_settings_path_frozen_mode(self) -> None:
        """
        GIVEN Tkinter GUI running under frozen execution
        WHEN load_settings or save_settings is called
        THEN settings are read from and written to settings.json adjacent to executable
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dist_dir = Path(tmp_dir).resolve()
            fake_exe = dist_dir / "PageOracle.exe"
            expected_settings_file = dist_dir / "settings.json"

            with (
                patch.object(sys, "frozen", True, create=True),
                patch.object(sys, "executable", str(fake_exe)),
            ):
                self.assertEqual(tk_gui.get_settings_file().resolve(), expected_settings_file)
                tk_gui.save_settings({"provider": "CustomTest", "temperature": 0.5})
                self.assertTrue(expected_settings_file.exists())
                loaded = tk_gui.load_settings()
                self.assertEqual(loaded.get("provider"), "CustomTest")
                self.assertEqual(loaded.get("temperature"), 0.5)

    def test_pageoracle_spec_structure(self) -> None:
        """
        GIVEN root PyInstaller specification file `pageoracle.spec`
        WHEN analyzing its configuration
        THEN entrypoint is tkinter_gui/gui.py, bundle format is onedir (COLLECT),
             hiddenimports include critical RAG submodules, and web dependencies are excluded
        """
        spec_path = REPO_ROOT / "pageoracle.spec"
        self.assertTrue(spec_path.exists(), "pageoracle.spec must exist at repository root")

        content = spec_path.read_text(encoding="utf-8")
        # Entrypoint script
        self.assertIn("tkinter_gui/gui.py", content)
        # Directory distribution (COLLECT)
        self.assertIn("coll = COLLECT(", content)
        # Output exe name
        self.assertIn('name="PageOracle"', content)
        # Excluded unused packages
        for excluded in ["flask", "jinja2", "werkzeug", "pytest"]:
            self.assertIn(f'"{excluded}"', content)
        # Critical hidden imports and data hooks
        for required in [
            "chromadb",
            "certifi",
            "langchain",
            "langchain_community",
            "langgraph",
            "sentence_transformers",
        ]:
            self.assertIn(f'"{required}"', content)

    @patch("PyInstaller.utils.hooks.copy_metadata", return_value=[("mock_meta", ".")])
    @patch("PyInstaller.utils.hooks.collect_submodules", return_value=["mock_sub"])
    @patch("PyInstaller.utils.hooks.collect_data_files", return_value=[("mock_data", ".")])
    def test_pageoracle_spec_syntax_and_analysis_evaluation(
        self,
        mock_collect_data: MagicMock,
        mock_collect_sub: MagicMock,
        mock_copy_meta: MagicMock,
    ) -> None:
        """
        GIVEN pageoracle.spec
        WHEN compiling and executing the spec with PyInstaller builtins
        THEN Analysis receives tkinter_gui/gui.py and COLLECT targets PageOracle
        """
        spec_path = REPO_ROOT / "pageoracle.spec"
        code = compile(spec_path.read_text(encoding="utf-8"), str(spec_path), "exec")

        mock_analysis = MagicMock()
        mock_pyz = MagicMock()
        mock_exe = MagicMock()
        mock_collect = MagicMock()

        env = {
            "Analysis": mock_analysis,
            "PYZ": mock_pyz,
            "EXE": mock_exe,
            "COLLECT": mock_collect,
            "__file__": str(spec_path),
        }
        exec(code, env)  # noqa: S102

        mock_analysis.assert_called_once()
        args, kwargs = mock_analysis.call_args
        self.assertEqual(args[0], ["tkinter_gui/gui.py"])
        self.assertIn("tkinter_gui", kwargs.get("pathex", []))

        mock_collect.assert_called_once()
        _, coll_kwargs = mock_collect.call_args
        self.assertEqual(coll_kwargs.get("name"), "PageOracle")
