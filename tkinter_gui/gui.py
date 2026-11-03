"""
PageOracle GUI (Tkinter-Designer style layout)
Интерфейс в стиле Tkinter-Designer:
- Canvas как основа лейаута
- Точное позиционирование блоков по Figma-структуре
- Привязка к backend из main.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from tkinter import (
    END,
    NORMAL,
    WORD,
    BooleanVar,
    Button,
    Canvas,
    Entry,
    Listbox,
    Menu,
    StringVar,
    TclError,
    Text,
    Tk,
    Toplevel,
    filedialog,
    messagebox,
    ttk,
)

from main import PROVIDERS, PageOracleBackend

# Tkinter-Designer style constants
OUTPUT_PATH = Path(__file__).parent
SETTINGS_FILE = OUTPUT_PATH / "settings.json"


# Color tokens from Figma mockups
BG = "#1E1E2E"
SIDEBAR_BG = "#2B2B3D"
SURFACE = "#363648"
PRIMARY = "#7C5CFC"
PRIMARY_HOVER = "#9579FF"
SECONDARY = "#4ECDC4"
TEXT_PRI = "#FFFFFF"
TEXT_SEC = "#A0A0B8"
INPUT_BG = "#2E2E42"
BORDER = "#45455A"
SUCCESS = "#4CAF50"
ERROR = "#FF6B6B"
LOG_BG = "#1A1A28"
INPUT_PLACEHOLDER = "Задайте вопрос по загруженным книгам…"
EMBEDDING_OPTIONS = [
    "nvidia/llama-nemotron-embed-vl-1b-v2:free",
    "BAAI/bge-m3",
    "text-search-doc/latest",
]


class TextRedirector(io.TextIOBase):
    """Прокидывает print-логи в виджет логов, сохраняя вывод и в stdout."""

    def __init__(self, callback, original):
        self.callback = callback
        self.original = original

    def write(self, text):
        if text and text.strip():
            self.callback(text)
        if self.original:
            self.original.write(text)
        return len(text) if text else 0

    def flush(self):
        if self.original:
            self.original.flush()


def load_settings() -> dict:
    """Загружает настройки UI/модели и мягко мигрирует legacy api_key и endpoint."""
    defaults = {
        "provider": "YandexGPT",
        "model": "yandexgpt-5.1/latest",
        "embedding_model": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
        "llm_api_key": "",
        "embedding_api_key": "",
        "yc_folder_id": "",
        "custom_providers": [],
        "llm_endpoint": "",
        "embedding_endpoint": "",
        "embedding_same_as_llm": False,
        "temperature": 0.2,
        "max_tokens": 4096,
        "top_p": 0.9,
        "score_threshold": 0.6,
    }
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, encoding="utf-8") as file:
                payload = json.load(file)
            legacy_api_key = str(payload.get("api_key", "")).strip()
            if legacy_api_key:
                payload.setdefault("llm_api_key", legacy_api_key)
                payload.setdefault("embedding_api_key", legacy_api_key)
            legacy_endpoint = str(payload.get("endpoint", "")).strip()
            if legacy_endpoint and not payload.get("llm_endpoint"):
                payload["llm_endpoint"] = legacy_endpoint
            defaults.update(payload)
        except Exception:
            pass
    return defaults


def save_settings(data: dict) -> None:
    """Сохраняет настройки приложения в settings.json."""
    with open(SETTINGS_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


class CustomProvidersDialog:
    """Диалог управления кастомными OpenAI-совместимыми провайдерами."""

    def __init__(self, parent, initial_providers: list[dict], on_update):
        self.on_update = on_update
        self.custom_providers: list[dict] = [dict(p) for p in initial_providers]
        self.selected_index: int | None = None

        win_width = 620
        win_height = 490

        self.win = Toplevel(parent)
        self.win.title("PageOracle - Кастомные провайдеры")
        self.win.configure(bg=BG)
        self.win.geometry(f"{win_width}x{win_height}")
        self.win.resizable(False, False)
        self.win.transient(parent)
        self.win.grab_set()

        self.win.update_idletasks()
        px = parent.winfo_x() + (parent.winfo_width() - win_width) // 2
        py = parent.winfo_y() + (parent.winfo_height() - win_height) // 2
        self.win.geometry(f"+{px}+{py}")

        canvas = Canvas(self.win, bg=BG, width=win_width, height=win_height, bd=0, highlightthickness=0)
        canvas.place(x=0, y=0)

        canvas.create_rectangle(16, 16, win_width - 16, win_height - 16, fill=BG, outline=BORDER, width=1)
        canvas.create_text(
            28, 28, anchor="nw", text="Управление кастомными провайдерами", fill=TEXT_PRI, font=("Segoe UI", 14, "bold")
        )

        canvas.create_text(28, 64, anchor="nw", text="Список провайдеров", fill=TEXT_SEC, font=("Segoe UI", 10))
        self.providers_listbox = Listbox(
            self.win,
            bg=SURFACE,
            fg=TEXT_PRI,
            selectbackground=PRIMARY,
            selectforeground=TEXT_PRI,
            font=("Segoe UI", 10),
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        self.providers_listbox.place(x=28, y=88, width=190, height=316)
        self.providers_listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        right_x = 238
        field_w = 350
        field_h = 30

        canvas.create_text(right_x, 64, anchor="nw", text="Название провайдера *", fill=TEXT_SEC, font=("Segoe UI", 10))
        self.name_entry = Entry(
            self.win, font=("Segoe UI", 10), bg=INPUT_BG, fg=TEXT_PRI, insertbackground=TEXT_PRI, relief="flat", bd=0
        )
        SettingsWindow._enable_entry_undo(self.name_entry)
        self.name_entry.place(x=right_x, y=86, width=field_w, height=field_h)

        canvas.create_text(
            right_x,
            124,
            anchor="nw",
            text="Endpoint URL * (напр. http://localhost:11434/v1)",
            fill=TEXT_SEC,
            font=("Segoe UI", 10),
        )
        self.endpoint_entry = Entry(
            self.win, font=("Segoe UI", 10), bg=INPUT_BG, fg=TEXT_PRI, insertbackground=TEXT_PRI, relief="flat", bd=0
        )
        SettingsWindow._enable_entry_undo(self.endpoint_entry)
        self.endpoint_entry.place(x=right_x, y=146, width=field_w, height=field_h)

        canvas.create_text(
            right_x, 184, anchor="nw", text="API Ключ (опционально для локальных)", fill=TEXT_SEC, font=("Segoe UI", 10)
        )
        self.api_key_entry = Entry(
            self.win,
            font=("Segoe UI", 10),
            bg=INPUT_BG,
            fg=TEXT_PRI,
            insertbackground=TEXT_PRI,
            relief="flat",
            bd=0,
            show="*",
        )
        SettingsWindow._enable_entry_undo(self.api_key_entry)
        self.api_key_entry.place(x=right_x, y=206, width=field_w, height=field_h)

        canvas.create_text(
            right_x, 244, anchor="nw", text="LLM модель по умолчанию", fill=TEXT_SEC, font=("Segoe UI", 10)
        )
        self.model_entry = Entry(
            self.win, font=("Segoe UI", 10), bg=INPUT_BG, fg=TEXT_PRI, insertbackground=TEXT_PRI, relief="flat", bd=0
        )
        SettingsWindow._enable_entry_undo(self.model_entry)
        self.model_entry.place(x=right_x, y=266, width=field_w, height=field_h)

        canvas.create_text(
            right_x, 304, anchor="nw", text="Embedding-модель по умолчанию", fill=TEXT_SEC, font=("Segoe UI", 10)
        )
        self.embedding_model_entry = Entry(
            self.win, font=("Segoe UI", 10), bg=INPUT_BG, fg=TEXT_PRI, insertbackground=TEXT_PRI, relief="flat", bd=0
        )
        SettingsWindow._enable_entry_undo(self.embedding_model_entry)
        self.embedding_model_entry.place(x=right_x, y=326, width=field_w, height=field_h)

        btn_add = Button(
            self.win,
            text="+ Добавить",
            bg=SURFACE,
            fg=TEXT_PRI,
            activebackground=BORDER,
            activeforeground=TEXT_PRI,
            relief="flat",
            cursor="hand2",
            font=("Segoe UI", 9, "bold"),
            command=self._add_provider,
        )
        btn_add.place(x=right_x, y=372, width=105, height=32)

        btn_update = Button(
            self.win,
            text="💾 Обновить",
            bg=SURFACE,
            fg=TEXT_PRI,
            activebackground=BORDER,
            activeforeground=TEXT_PRI,
            relief="flat",
            cursor="hand2",
            font=("Segoe UI", 9),
            command=self._update_selected_provider,
        )
        btn_update.place(x=right_x + 115, y=372, width=110, height=32)

        btn_delete = Button(
            self.win,
            text="🗑 Удалить",
            bg=SURFACE,
            fg=ERROR,
            activebackground=BORDER,
            activeforeground=ERROR,
            relief="flat",
            cursor="hand2",
            font=("Segoe UI", 9),
            command=self._delete_provider,
        )
        btn_delete.place(x=right_x + 235, y=372, width=115, height=32)

        btn_cancel = Button(
            self.win,
            text="Отмена",
            bg=SURFACE,
            fg=TEXT_SEC,
            activebackground=BORDER,
            activeforeground=TEXT_PRI,
            relief="flat",
            cursor="hand2",
            font=("Segoe UI", 10),
            command=self.win.destroy,
        )
        btn_cancel.place(x=380, y=424, width=95, height=36)

        btn_ok = Button(
            self.win,
            text="Готово",
            bg=PRIMARY,
            fg=TEXT_PRI,
            activebackground=PRIMARY_HOVER,
            activeforeground=TEXT_PRI,
            relief="flat",
            cursor="hand2",
            font=("Segoe UI", 10, "bold"),
            command=self._on_ok,
        )
        btn_ok.place(x=488, y=424, width=100, height=36)

        self._refresh_listbox()
        if self.custom_providers:
            self._select_provider_by_index(0)

    @staticmethod
    def _set_entry_text(entry: Entry, text: str) -> None:
        entry.delete(0, END)
        entry.insert(0, text)

    def _refresh_listbox(self) -> None:
        self.providers_listbox.delete(0, END)
        for p in self.custom_providers:
            self.providers_listbox.insert(END, p.get("name", "Unnamed"))

    def _on_listbox_select(self, _evt=None) -> None:
        sel = self.providers_listbox.curselection()
        if sel:
            self._select_provider_by_index(sel[0])

    def _select_provider_by_index(self, idx: int) -> None:
        if 0 <= idx < len(self.custom_providers):
            self.selected_index = idx
            p = self.custom_providers[idx]
            self._set_entry_text(self.name_entry, str(p.get("name", "")))
            self._set_entry_text(self.endpoint_entry, str(p.get("endpoint", "")))
            self._set_entry_text(self.api_key_entry, str(p.get("api_key", "")))
            self._set_entry_text(self.model_entry, str(p.get("model", "")))
            self._set_entry_text(self.embedding_model_entry, str(p.get("embedding_model", "")))
            self.providers_listbox.selection_clear(0, END)
            self.providers_listbox.selection_set(idx)
            self.providers_listbox.activate(idx)

    def _add_provider(self) -> None:
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Внимание", "Введите название провайдера.", parent=self.win)
            return
        endpoint = self.endpoint_entry.get().strip()
        api_key = self.api_key_entry.get().strip()
        model = self.model_entry.get().strip()
        embedding_model = self.embedding_model_entry.get().strip()

        existing_idx = next((i for i, p in enumerate(self.custom_providers) if p.get("name") == name), None)
        item = {
            "name": name,
            "endpoint": endpoint,
            "api_key": api_key,
            "model": model,
            "embedding_model": embedding_model,
        }
        if existing_idx is not None:
            self.custom_providers[existing_idx] = item
            self.selected_index = existing_idx
        else:
            self.custom_providers.append(item)
            self.selected_index = len(self.custom_providers) - 1

        self._refresh_listbox()
        if self.selected_index is not None:
            self._select_provider_by_index(self.selected_index)

    def _update_selected_provider(self) -> None:
        if self.selected_index is None or not (0 <= self.selected_index < len(self.custom_providers)):
            messagebox.showwarning("Внимание", "Выберите провайдера из списка для обновления.", parent=self.win)
            return
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Внимание", "Введите название провайдера.", parent=self.win)
            return
        self.custom_providers[self.selected_index] = {
            "name": name,
            "endpoint": self.endpoint_entry.get().strip(),
            "api_key": self.api_key_entry.get().strip(),
            "model": self.model_entry.get().strip(),
            "embedding_model": self.embedding_model_entry.get().strip(),
        }
        self._refresh_listbox()
        self._select_provider_by_index(self.selected_index)

    def _delete_provider(self) -> None:
        if self.selected_index is None or not (0 <= self.selected_index < len(self.custom_providers)):
            messagebox.showwarning("Внимание", "Выберите провайдера из списка для удаления.", parent=self.win)
            return
        self.custom_providers.pop(self.selected_index)
        self.selected_index = None
        self._set_entry_text(self.name_entry, "")
        self._set_entry_text(self.endpoint_entry, "")
        self._set_entry_text(self.api_key_entry, "")
        self._set_entry_text(self.model_entry, "")
        self._set_entry_text(self.embedding_model_entry, "")
        self._refresh_listbox()

    def _on_ok(self) -> None:
        self.on_update(self.custom_providers)
        self.win.destroy()


class SettingsWindow:
    """Модальное окно конфигурации провайдера, модели и параметров генерации."""

    @staticmethod
    def _enable_entry_undo(entry: Entry) -> None:
        with contextlib.suppress(TclError):
            entry.tk.call(str(entry), "configure", "-undo", 1)

    def __init__(self, parent, current: dict, providers: dict, on_save):
        self.providers = providers
        self.on_save = on_save
        self.custom_providers: list[dict] = [dict(p) for p in current.get("custom_providers", [])]
        win_width = 560
        win_height = 844

        self.win = Toplevel(parent)
        self.win.title("PageOracle - Настройки")
        self.win.configure(bg=BG)
        self.win.geometry(f"{win_width}x{win_height}")
        self.win.resizable(False, False)
        self.win.transient(parent)
        self.win.grab_set()

        self.win.update_idletasks()
        px = parent.winfo_x() + (parent.winfo_width() - win_width) // 2
        py = parent.winfo_y() + (parent.winfo_height() - win_height) // 2
        self.win.geometry(f"+{px}+{py}")

        canvas = Canvas(self.win, bg=BG, width=win_width, height=win_height, bd=0, highlightthickness=0)
        canvas.place(x=0, y=0)

        canvas.create_rectangle(16, 16, 544, 834, fill=BG, outline=BORDER, width=1)
        canvas.create_text(34, 30, anchor="nw", text="⚙ Настройки", fill=TEXT_PRI, font=("Segoe UI", 18, "bold"))

        canvas.create_text(34, 70, anchor="nw", text="Провайдер ИИ", fill=TEXT_SEC, font=("Segoe UI", 10))
        self.provider_var = StringVar(value=current.get("provider", "DeepSeek"))
        self.provider_cb = ttk.Combobox(
            self.win,
            textvariable=self.provider_var,
            state="readonly",
            font=("Segoe UI", 11),
        )
        self._refresh_provider_list()
        self.provider_cb.place(x=34, y=92, width=344, height=32)
        self.provider_cb.bind("<<ComboboxSelected>>", self._on_provider_change)

        btn_manage_providers = Button(
            self.win,
            text="⚙ Провайдеры...",
            bg=SURFACE,
            fg=TEXT_PRI,
            activebackground=BORDER,
            activeforeground=TEXT_PRI,
            relief="flat",
            cursor="hand2",
            font=("Segoe UI", 9, "bold"),
            command=self._open_custom_providers_dialog,
        )
        btn_manage_providers.place(x=388, y=92, width=138, height=32)

        canvas.create_text(34, 134, anchor="nw", text="LLM Endpoint URL", fill=TEXT_SEC, font=("Segoe UI", 10))
        self.llm_endpoint_entry = Entry(
            self.win,
            font=("Segoe UI", 11),
            bg=INPUT_BG,
            fg=TEXT_PRI,
            insertbackground=TEXT_PRI,
            relief="flat",
            bd=0,
        )
        self._enable_entry_undo(self.llm_endpoint_entry)
        self.llm_endpoint_entry.place(x=34, y=156, width=492, height=32)
        self.llm_endpoint_entry.insert(0, current.get("llm_endpoint", ""))

        canvas.create_text(34, 198, anchor="nw", text="LLM модель", fill=TEXT_SEC, font=("Segoe UI", 10))
        self.model_var = StringVar(value=current.get("model", "deepseek-chat"))
        self.model_cb = ttk.Combobox(self.win, textvariable=self.model_var, font=("Segoe UI", 11), state="normal")
        self._update_model_list()
        self.model_cb.place(x=34, y=220, width=492, height=32)

        canvas.create_text(34, 262, anchor="nw", text="API ключ от LLM", fill=TEXT_SEC, font=("Segoe UI", 10))
        self.llm_api_entry = Entry(
            self.win,
            font=("Segoe UI", 11),
            show="*",
            bg=INPUT_BG,
            fg=TEXT_PRI,
            insertbackground=TEXT_PRI,
            relief="flat",
            bd=0,
        )
        self._enable_entry_undo(self.llm_api_entry)
        self.llm_api_entry.place(x=34, y=284, width=492, height=32)
        self.llm_api_entry.insert(0, current.get("llm_api_key", ""))

        canvas.create_text(
            34, 326, anchor="nw", text="YC_FOLDER_ID (для YandexGPT)", fill=TEXT_SEC, font=("Segoe UI", 10)
        )
        self.yc_folder_entry = Entry(
            self.win,
            font=("Segoe UI", 11),
            bg=INPUT_BG,
            fg=TEXT_PRI,
            insertbackground=TEXT_PRI,
            relief="flat",
            bd=0,
        )
        self._enable_entry_undo(self.yc_folder_entry)
        self.yc_folder_entry.place(x=34, y=348, width=492, height=32)
        self.yc_folder_entry.insert(0, current.get("yc_folder_id", ""))

        canvas.create_line(34, 390, 526, 390, fill=BORDER)

        canvas.create_text(34, 400, anchor="nw", text="Embedding-модель", fill=TEXT_SEC, font=("Segoe UI", 10))
        self.embedding_var = StringVar(value=current.get("embedding_model", EMBEDDING_OPTIONS[0]))
        self.embedding_cb = ttk.Combobox(
            self.win,
            textvariable=self.embedding_var,
            values=EMBEDDING_OPTIONS,
            state="normal",
            font=("Segoe UI", 11),
        )
        self.embedding_cb.place(x=34, y=422, width=492, height=32)

        self.sync_embeddings_var = BooleanVar(value=bool(current.get("embedding_same_as_llm", False)))
        sync_cb = ttk.Checkbutton(
            self.win,
            text="Использовать тот же провайдер и endpoint для эмбеддингов",
            variable=self.sync_embeddings_var,
            command=self._on_sync_toggle,
        )
        sync_cb.place(x=34, y=462)

        canvas.create_text(34, 494, anchor="nw", text="Embedding Endpoint URL", fill=TEXT_SEC, font=("Segoe UI", 10))
        self.embedding_endpoint_entry = Entry(
            self.win,
            font=("Segoe UI", 11),
            bg=INPUT_BG,
            fg=TEXT_PRI,
            insertbackground=TEXT_PRI,
            relief="flat",
            bd=0,
        )
        self._enable_entry_undo(self.embedding_endpoint_entry)
        self.embedding_endpoint_entry.place(x=34, y=516, width=492, height=32)
        self.embedding_endpoint_entry.insert(0, current.get("embedding_endpoint", ""))

        canvas.create_text(34, 558, anchor="nw", text="API ключ embedding-модели", fill=TEXT_SEC, font=("Segoe UI", 10))
        self.embedding_api_entry = Entry(
            self.win,
            font=("Segoe UI", 11),
            show="*",
            bg=INPUT_BG,
            fg=TEXT_PRI,
            insertbackground=TEXT_PRI,
            relief="flat",
            bd=0,
        )
        self._enable_entry_undo(self.embedding_api_entry)
        self.embedding_api_entry.place(x=34, y=580, width=492, height=32)
        self.embedding_api_entry.insert(0, current.get("embedding_api_key", ""))

        self.show_var = StringVar(value="0")
        show_btn = ttk.Checkbutton(
            self.win,
            text="Показать ключи",
            variable=self.show_var,
            command=self._toggle_key,
        )
        show_btn.place(x=34, y=620)

        canvas.create_line(34, 650, 526, 650, fill=BORDER)

        canvas.create_text(34, 658, anchor="nw", text="Temperature", fill=TEXT_SEC, font=("Segoe UI", 10))
        self.temperature_entry = Entry(
            self.win,
            font=("Segoe UI", 11),
            bg=INPUT_BG,
            fg=TEXT_PRI,
            insertbackground=TEXT_PRI,
            relief="flat",
            bd=0,
        )
        self._enable_entry_undo(self.temperature_entry)
        self.temperature_entry.place(x=34, y=678, width=236, height=32)
        self.temperature_entry.insert(0, str(current.get("temperature", 0.3)))

        canvas.create_text(290, 658, anchor="nw", text="Max Tokens", fill=TEXT_SEC, font=("Segoe UI", 10))
        self.max_tokens_entry = Entry(
            self.win,
            font=("Segoe UI", 11),
            bg=INPUT_BG,
            fg=TEXT_PRI,
            insertbackground=TEXT_PRI,
            relief="flat",
            bd=0,
        )
        self._enable_entry_undo(self.max_tokens_entry)
        self.max_tokens_entry.place(x=290, y=678, width=236, height=32)
        self.max_tokens_entry.insert(0, str(current.get("max_tokens", 2048)))

        canvas.create_text(34, 718, anchor="nw", text="Top P", fill=TEXT_SEC, font=("Segoe UI", 10))
        self.top_p_entry = Entry(
            self.win,
            font=("Segoe UI", 11),
            bg=INPUT_BG,
            fg=TEXT_PRI,
            insertbackground=TEXT_PRI,
            relief="flat",
            bd=0,
        )
        self._enable_entry_undo(self.top_p_entry)
        self.top_p_entry.place(x=34, y=738, width=236, height=32)
        self.top_p_entry.insert(0, str(current.get("top_p", 0.8)))

        canvas.create_text(290, 718, anchor="nw", text="Score Threshold", fill=TEXT_SEC, font=("Segoe UI", 10))
        self.score_threshold_entry = Entry(
            self.win,
            font=("Segoe UI", 11),
            bg=INPUT_BG,
            fg=TEXT_PRI,
            insertbackground=TEXT_PRI,
            relief="flat",
            bd=0,
        )
        self._enable_entry_undo(self.score_threshold_entry)
        self.score_threshold_entry.place(x=290, y=738, width=236, height=32)
        self.score_threshold_entry.insert(0, str(current.get("score_threshold", 0.6)))

        cancel_btn = Button(
            self.win,
            text="Отмена",
            bg=SURFACE,
            fg=TEXT_SEC,
            activebackground=BORDER,
            activeforeground=TEXT_PRI,
            relief="flat",
            command=self.win.destroy,
            cursor="hand2",
            font=("Segoe UI", 11),
        )
        cancel_btn.place(x=306, y=786, width=100, height=38)

        save_btn = Button(
            self.win,
            text="💾 Сохранить",
            bg=PRIMARY,
            fg=TEXT_PRI,
            activebackground=PRIMARY_HOVER,
            activeforeground=TEXT_PRI,
            relief="flat",
            command=self._save,
            cursor="hand2",
            font=("Segoe UI", 11, "bold"),
        )
        save_btn.place(x=416, y=786, width=110, height=38)

        self._on_sync_toggle()

    def _open_custom_providers_dialog(self) -> None:
        def on_update(updated_providers: list[dict]) -> None:
            self.custom_providers = list(updated_providers)
            self._refresh_provider_list()

        CustomProvidersDialog(self.win, self.custom_providers, on_update)

    def _refresh_provider_list(self) -> None:
        all_values = list(self.providers.keys())
        for p in self.custom_providers:
            name = p.get("name")
            if name and name not in all_values:
                all_values.append(name)
        if "Custom" not in all_values:
            all_values.append("Custom")
        self.provider_cb["values"] = all_values

    def _on_sync_toggle(self) -> None:
        if self.sync_embeddings_var.get():
            self._sync_embeddings_with_llm()
            self.embedding_endpoint_entry.configure(state="disabled")
            self.embedding_api_entry.configure(state="disabled")
        else:
            self.embedding_endpoint_entry.configure(state="normal")
            self.embedding_api_entry.configure(state="normal")

    def _sync_embeddings_with_llm(self) -> None:
        llm_ep = self.llm_endpoint_entry.get().strip()
        llm_key = self.llm_api_entry.get().strip()
        self.embedding_endpoint_entry.configure(state="normal")
        self.embedding_api_entry.configure(state="normal")
        self.embedding_endpoint_entry.delete(0, END)
        self.embedding_endpoint_entry.insert(0, llm_ep)
        self.embedding_api_entry.delete(0, END)
        self.embedding_api_entry.insert(0, llm_key)
        if self.sync_embeddings_var.get():
            self.embedding_endpoint_entry.configure(state="disabled")
            self.embedding_api_entry.configure(state="disabled")

    def _toggle_key(self) -> None:
        mask = "" if self.show_var.get() == "1" else "*"
        self.llm_api_entry.configure(show=mask)
        self.embedding_api_entry.configure(show=mask)

    def _on_provider_change(self, _evt=None) -> None:
        provider = self.provider_var.get().strip()
        matching_custom = next((p for p in self.custom_providers if p.get("name") == provider), None)
        if matching_custom:
            if matching_custom.get("endpoint"):
                self.llm_endpoint_entry.delete(0, END)
                self.llm_endpoint_entry.insert(0, matching_custom["endpoint"])
            if matching_custom.get("model"):
                self.model_var.set(matching_custom["model"])
            if matching_custom.get("api_key"):
                self.llm_api_entry.delete(0, END)
                self.llm_api_entry.insert(0, matching_custom["api_key"])
            if matching_custom.get("embedding_model"):
                self.embedding_var.set(matching_custom["embedding_model"])
            if self.sync_embeddings_var.get():
                self._sync_embeddings_with_llm()
        elif provider in self.providers:
            models = self.providers[provider].get("models", [])
            if models and (not self.model_var.get() or self.model_var.get() not in models):
                self.model_var.set(models[0])
        self._update_model_list()

    def _update_model_list(self) -> None:
        provider = self.provider_var.get().strip()
        models = self.providers.get(provider, {}).get("models", [])
        matching_custom = next((p for p in self.custom_providers if p.get("name") == provider), None)
        if matching_custom and matching_custom.get("model"):
            models = [matching_custom["model"]]
        self.model_cb["values"] = models
        if models and not self.model_var.get():
            self.model_var.set(models[0])

    def _save(self) -> None:
        temperature = self._read_float(self.temperature_entry, "Temperature", 0.0, 2.0)
        if temperature is None:
            return

        max_tokens = self._read_int(self.max_tokens_entry, "Max Tokens", 1)
        if max_tokens is None:
            return

        top_p = self._read_float(self.top_p_entry, "Top P", 0.0, 1.0)
        if top_p is None:
            return

        score_threshold = self._read_float(
            self.score_threshold_entry,
            "Score Threshold",
            0.0,
            1.0,
        )
        if score_threshold is None:
            return

        provider = self.provider_var.get().strip()
        model = self.model_var.get().strip()
        embedding_model = self.embedding_var.get().strip()
        llm_endpoint = self.llm_endpoint_entry.get().strip()
        llm_api_key = self.llm_api_entry.get().strip()
        yc_folder_id = self.yc_folder_entry.get().strip()

        if self.sync_embeddings_var.get():
            embedding_endpoint = llm_endpoint
            embedding_api_key = llm_api_key
        else:
            embedding_endpoint = self.embedding_endpoint_entry.get().strip()
            embedding_api_key = self.embedding_api_entry.get().strip()

        matching_custom = next((p for p in self.custom_providers if p.get("name") == provider), None)
        is_custom = (
            (provider == "Custom") or bool(matching_custom) or (provider not in self.providers and bool(llm_endpoint))
        )

        if not provider:
            messagebox.showwarning("Внимание", "Укажите провайдера.", parent=self.win)
            return

        if not is_custom and provider not in self.providers:
            messagebox.showwarning("Внимание", "Неизвестный провайдер.", parent=self.win)
            return

        if not model:
            messagebox.showwarning("Внимание", "Введите название модели.", parent=self.win)
            return

        if not embedding_model:
            messagebox.showwarning("Внимание", "Введите название embedding-модели.", parent=self.win)
            return

        if is_custom and not llm_endpoint:
            messagebox.showwarning("Внимание", "Укажите LLM Endpoint URL для кастомного провайдера.", parent=self.win)
            return

        if not is_custom and not llm_api_key:
            messagebox.showwarning("Внимание", "Введите API ключ LLM.", parent=self.win)
            return

        if embedding_model == "nvidia/llama-nemotron-embed-vl-1b-v2:free" and not embedding_api_key:
            messagebox.showwarning("Внимание", "Введите API ключ Embedding для OpenRouter модели.", parent=self.win)
            return

        uses_yandex = provider == "YandexGPT" or embedding_model == "text-search-doc/latest"
        if uses_yandex and not yc_folder_id:
            messagebox.showwarning("Внимание", "Введите YC_FOLDER_ID для Yandex моделей.", parent=self.win)
            return

        data = {
            "provider": provider,
            "model": model,
            "embedding_model": embedding_model,
            "llm_api_key": llm_api_key,
            "embedding_api_key": embedding_api_key,
            "yc_folder_id": yc_folder_id,
            "custom_providers": self.custom_providers,
            "llm_endpoint": llm_endpoint,
            "embedding_endpoint": embedding_endpoint,
            "embedding_same_as_llm": bool(self.sync_embeddings_var.get()),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "score_threshold": score_threshold,
        }
        save_settings(data)
        self.on_save(data)
        self.win.destroy()

    def _read_float(
        self,
        entry: Entry,
        label: str,
        min_value: float,
        max_value: float,
    ) -> float | None:
        raw = entry.get().strip()
        try:
            value = float(raw)
        except ValueError:
            messagebox.showwarning("Внимание", f"{label} должен быть числом.", parent=self.win)
            return None

        if value < min_value or value > max_value:
            messagebox.showwarning(
                "Внимание",
                f"{label} должен быть в диапазоне {min_value:g}..{max_value:g}.",
                parent=self.win,
            )
            return None
        return value

    def _read_int(self, entry: Entry, label: str, min_value: int) -> int | None:
        raw = entry.get().strip()
        try:
            value = int(raw)
        except ValueError:
            messagebox.showwarning("Внимание", f"{label} должен быть целым числом.", parent=self.win)
            return None

        if value < min_value:
            messagebox.showwarning(
                "Внимание",
                f"{label} должен быть больше или равен {min_value}.",
                parent=self.win,
            )
            return None
        return value


class PageOracleApp:
    """Tkinter-приложение: слой UI поверх PageOracleBackend."""

    @staticmethod
    def _enable_entry_undo(entry: Entry) -> None:
        with contextlib.suppress(TclError):
            entry.tk.call(str(entry), "configure", "-undo", 1)

    def __init__(self) -> None:
        self.settings = load_settings()
        self.backend: PageOracleBackend | None = None
        self.is_initialized = False
        self.is_busy = False
        self.mode = "auto"
        self.history_file = OUTPUT_PATH / "chat_history.json"

        self.window = Tk()
        self.window.title("PageOracle - AI Book Reader")
        self.window.geometry("1200x800")
        self.window.configure(bg=BG)
        self.window.resizable(False, False)

        self._setup_styles()
        self._build_layout()
        self._setup_text_editing()
        self._redirect_stdout()
        self._start_init()

    def _setup_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "TCombobox",
            fieldbackground=INPUT_BG,
            background=SURFACE,
            foreground=TEXT_PRI,
            arrowcolor=TEXT_SEC,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", INPUT_BG)],
            selectbackground=[("readonly", INPUT_BG)],
            selectforeground=[("readonly", TEXT_PRI)],
        )
        style.configure(
            "TCheckbutton",
            background=BG,
            foreground=TEXT_SEC,
        )

    def _build_layout(self) -> None:
        self.canvas = Canvas(self.window, bg=BG, width=1200, height=800, bd=0, highlightthickness=0)
        self.canvas.place(x=0, y=0)

        self.canvas.create_rectangle(0, 0, 280, 800, fill=SIDEBAR_BG, outline="")
        self.canvas.create_text(16, 24, anchor="nw", text="PageOracle", fill=PRIMARY, font=("Segoe UI", 22, "bold"))
        self.canvas.create_text(16, 61, anchor="nw", text="AI Book Reader", fill=TEXT_SEC, font=("Segoe UI", 10))
        self.canvas.create_line(16, 94, 264, 94, fill=BORDER)

        self.btn_load = Button(
            self.window,
            text="  +  Загрузить книгу",
            bg=PRIMARY,
            fg=TEXT_PRI,
            activebackground=PRIMARY_HOVER,
            activeforeground=TEXT_PRI,
            relief="flat",
            cursor="hand2",
            anchor="w",
            font=("Segoe UI", 11, "bold"),
            command=self._on_load_book,
        )
        self.btn_load.place(x=16, y=112, width=248, height=56)

        self.canvas.create_text(
            16, 184, anchor="nw", text="Загруженные книги", fill=TEXT_SEC, font=("Segoe UI", 10, "bold")
        )

        self.books_text = Text(
            self.window,
            bg=SURFACE,
            fg=TEXT_PRI,
            font=("Segoe UI", 10),
            wrap=WORD,
            relief="flat",
            bd=0,
            padx=12,
            pady=12,
            state="disabled",
            cursor="arrow",
        )
        self.books_text.place(x=16, y=205, width=248, height=482)

        self.canvas.create_line(16, 706, 264, 706, fill=BORDER)

        self.btn_settings = Button(
            self.window,
            text="  ⚙  Настройки",
            bg=SURFACE,
            fg=TEXT_SEC,
            activebackground=BORDER,
            activeforeground=TEXT_PRI,
            relief="flat",
            cursor="hand2",
            anchor="w",
            font=("Segoe UI", 11),
            command=self._on_settings,
        )
        self.btn_settings.place(x=16, y=724, width=248, height=52)

        self.canvas.create_rectangle(280, 0, 1200, 800, fill=BG, outline="")
        self.canvas.create_text(304, 14, anchor="nw", text="Чат с книгой", fill=TEXT_PRI, font=("Segoe UI", 18, "bold"))

        self.status_dot = self.canvas.create_oval(960, 23, 968, 31, fill=SECONDARY, outline="")
        self.status_label = self.canvas.create_text(
            976,
            18,
            anchor="nw",
            text="Готов к работе",
            fill=SUCCESS,
            font=("Segoe UI", 10),
        )

        self.canvas.create_text(304, 70, anchor="nw", text="Режим:", fill=TEXT_SEC, font=("Segoe UI", 10))
        self.canvas.create_rectangle(360, 58, 680, 95, fill=SURFACE, outline=BORDER, width=1)

        self.btn_mode_auto = Button(
            self.window,
            text="🤖 Авто",
            bg=PRIMARY,
            fg=TEXT_PRI,
            activebackground=PRIMARY_HOVER,
            activeforeground=TEXT_PRI,
            relief="flat",
            cursor="hand2",
            command=lambda: self._set_mode("auto"),
            font=("Segoe UI", 10, "bold"),
        )
        self.btn_mode_auto.place(x=363, y=61, width=94, height=31)

        self.btn_mode_analysis = Button(
            self.window,
            text="📊 Анализ",
            bg=SURFACE,
            fg=TEXT_SEC,
            activebackground=PRIMARY_HOVER,
            activeforeground=TEXT_PRI,
            relief="flat",
            cursor="hand2",
            command=lambda: self._set_mode("analysis"),
            font=("Segoe UI", 10),
        )
        self.btn_mode_analysis.place(x=458, y=61, width=104, height=31)

        self.btn_mode_quote = Button(
            self.window,
            text="📎 Поиск цитат",
            bg=SURFACE,
            fg=TEXT_SEC,
            activebackground=BORDER,
            activeforeground=TEXT_PRI,
            relief="flat",
            cursor="hand2",
            command=lambda: self._set_mode("quote"),
            font=("Segoe UI", 10),
        )
        self.btn_mode_quote.place(x=563, y=61, width=114, height=31)

        self.chat_text = Text(
            self.window,
            bg=SURFACE,
            fg=TEXT_PRI,
            font=("Segoe UI", 11),
            wrap=WORD,
            relief="flat",
            bd=0,
            padx=18,
            pady=14,
            state="disabled",
            spacing3=8,
        )
        self.chat_text.place(x=304, y=100, width=872, height=471)

        self.chat_text.tag_configure("user_name", foreground=SECONDARY, font=("Segoe UI", 10, "bold"))
        self.chat_text.tag_configure("ai_name", foreground=PRIMARY, font=("Segoe UI", 10, "bold"))
        self.chat_text.tag_configure("user_msg", foreground=TEXT_PRI, font=("Segoe UI", 11), lmargin1=8, lmargin2=8)
        self.chat_text.tag_configure("ai_msg", foreground="#D8D8EC", font=("Segoe UI", 11), lmargin1=8, lmargin2=8)
        self.chat_text.tag_configure("thinking", foreground=TEXT_SEC, font=("Segoe UI", 10, "italic"))
        self.chat_text.tag_configure("error_msg", foreground=ERROR, font=("Segoe UI", 11))

        self.canvas.create_rectangle(304, 583, 1056, 631, fill=INPUT_BG, outline=BORDER, width=1)
        self.input_entry = Entry(
            self.window,
            bg=INPUT_BG,
            fg=TEXT_PRI,
            insertbackground=TEXT_PRI,
            relief="flat",
            bd=0,
            font=("Segoe UI", 11),
        )
        self._enable_entry_undo(self.input_entry)
        self.input_entry.place(x=324, y=598, width=712, height=20)
        self.input_entry.bind("<Return>", lambda _evt: self._on_send())
        self.input_entry.insert(0, INPUT_PLACEHOLDER)
        self.input_entry.bind("<FocusIn>", self._clear_placeholder)

        self.btn_send = Button(
            self.window,
            text="▶",
            bg=PRIMARY,
            fg=TEXT_PRI,
            activebackground=PRIMARY_HOVER,
            activeforeground=TEXT_PRI,
            relief="flat",
            cursor="hand2",
            command=self._on_send,
            font=("Segoe UI", 12, "bold"),
        )
        self.btn_send.place(x=1048, y=583, width=40, height=50)

        self.btn_clear_chat = Button(
            self.window,
            text="🗑",
            bg=SURFACE,
            fg=TEXT_SEC,
            activebackground=BORDER,
            activeforeground=TEXT_PRI,
            relief="flat",
            cursor="hand2",
            command=self._on_clear_chat,
            font=("Segoe UI", 10),
        )
        self.btn_clear_chat.place(x=1092, y=583, width=40, height=50)

        self.btn_clear_history = Button(
            self.window,
            text="🧠",
            bg=SURFACE,
            fg=TEXT_SEC,
            activebackground=BORDER,
            activeforeground=TEXT_PRI,
            relief="flat",
            cursor="hand2",
            command=self._on_clear_history,
            font=("Segoe UI", 10),
        )
        self.btn_clear_history.place(x=1138, y=583, width=38, height=50)

        self.canvas.create_text(
            304, 645, anchor="nw", text="Системные логи", fill=TEXT_SEC, font=("Segoe UI", 9, "bold")
        )

        self.btn_clear_logs = Button(
            self.window,
            text="Очистить",
            bg=BG,
            fg=TEXT_SEC,
            activebackground=BG,
            activeforeground=TEXT_PRI,
            relief="flat",
            cursor="hand2",
            command=self._on_clear_logs,
            font=("Segoe UI", 9),
        )
        self.btn_clear_logs.place(x=1123, y=642, width=50, height=16)

        self.log_text = Text(
            self.window,
            bg=LOG_BG,
            fg="#8888AA",
            font=("Consolas", 9),
            wrap=WORD,
            relief="flat",
            bd=0,
            padx=12,
            pady=10,
            height=7,
            state="disabled",
        )
        self.log_text.place(x=304, y=664, width=872, height=120)
        self.log_text.tag_configure("timestamp", foreground="#666680")
        self.log_text.tag_configure("info", foreground="#8888AA")
        self.log_text.tag_configure("success", foreground=SUCCESS)
        self.log_text.tag_configure("error", foreground=ERROR)

        for widget in (self.books_text, self.chat_text, self.log_text):
            widget.bind("<MouseWheel>", lambda event, target=widget: target.yview_scroll(-event.delta // 120, "units"))

        self._append_chat(
            "PageOracle",
            "Добро пожаловать! Загрузите книгу (.txt) через боковую панель и задайте вопрос.",
            is_ai=True,
        )

    def _setup_text_editing(self) -> None:
        """Подключает горячие клавиши и контекстное меню для Entry/Text."""
        self._context_target = None
        self.text_context_menu = Menu(self.window, tearoff=0)
        self.text_context_menu.add_command(label="Отменить", command=lambda: self._context_action("undo"))
        self.text_context_menu.add_command(label="Повторить", command=lambda: self._context_action("redo"))
        self.text_context_menu.add_separator()
        self.text_context_menu.add_command(label="Копировать", command=lambda: self._context_action("copy"))
        self.text_context_menu.add_command(label="Вставить", command=lambda: self._context_action("paste"))
        self.text_context_menu.add_command(label="Вырезать", command=lambda: self._context_action("cut"))
        self.text_context_menu.add_command(label="Удалить", command=lambda: self._context_action("delete"))
        self.text_context_menu.add_separator()
        self.text_context_menu.add_command(label="Выделить всё", command=lambda: self._context_action("select_all"))

        self.window.bind_all("<Control-c>", self._on_copy, add="+")
        self.window.bind_all("<Control-C>", self._on_copy, add="+")
        self.window.bind_all("<Control-a>", self._on_select_all, add="+")
        self.window.bind_all("<Control-A>", self._on_select_all, add="+")
        self.window.bind_all("<Control-z>", self._on_undo, add="+")
        self.window.bind_all("<Control-Z>", self._on_undo, add="+")
        self.window.bind_all("<Control-y>", self._on_redo, add="+")
        self.window.bind_all("<Control-Y>", self._on_redo, add="+")

        self.window.bind_all("<Button-3>", self._show_text_context_menu, add="+")

    def _resolve_text_widget(self, event=None):
        widget = getattr(event, "widget", None)
        if isinstance(widget, (Entry, Text)):
            return widget
        focused = self.window.focus_get()
        if isinstance(focused, (Entry, Text)):
            return focused
        return None

    def _is_widget_editable(self, widget) -> bool:
        try:
            state = str(widget.cget("state"))
        except Exception:
            state = "normal"
        return state not in {"disabled", "readonly"}

    def _context_action(self, action: str) -> None:
        widget = self._context_target
        if not isinstance(widget, (Entry, Text)):
            return

        if action == "undo":
            if self._is_widget_editable(widget):
                widget.event_generate("<<Undo>>")
        elif action == "redo":
            if self._is_widget_editable(widget):
                widget.event_generate("<<Redo>>")
        elif action == "copy":
            widget.event_generate("<<Copy>>")
        elif action == "paste":
            if self._is_widget_editable(widget):
                widget.event_generate("<<Paste>>")
        elif action == "cut":
            if self._is_widget_editable(widget):
                widget.event_generate("<<Cut>>")
        elif action == "delete":
            if self._is_widget_editable(widget):
                widget.event_generate("<Delete>")
        elif action == "select_all":
            if isinstance(widget, Entry):
                widget.selection_range(0, END)
                widget.icursor(END)
            else:
                widget.tag_add("sel", "1.0", "end-1c")
                widget.mark_set("insert", "end-1c")

    def _show_text_context_menu(self, event):
        """Показывает единое контекстное меню для текущего текстового виджета."""
        widget = self._resolve_text_widget(event)
        if not isinstance(widget, (Entry, Text)):
            return

        self._context_target = widget
        widget.focus_set()
        self.text_context_menu.tk_popup(event.x_root, event.y_root)
        self.text_context_menu.grab_release()
        return "break"

    def _on_copy(self, event):
        widget = self._resolve_text_widget(event)
        if not isinstance(widget, (Entry, Text)):
            return None
        widget.event_generate("<<Copy>>")
        return "break"

    def _on_paste(self, event):
        widget = self._resolve_text_widget(event)
        if not isinstance(widget, (Entry, Text)):
            return None
        if not self._is_widget_editable(widget):
            return "break"
        widget.event_generate("<<Paste>>")
        return "break"

    def _on_cut(self, event):
        widget = self._resolve_text_widget(event)
        if not isinstance(widget, (Entry, Text)):
            return None
        if not self._is_widget_editable(widget):
            return "break"
        widget.event_generate("<<Cut>>")
        return "break"

    def _on_select_all(self, event):
        widget = self._resolve_text_widget(event)
        if not isinstance(widget, (Entry, Text)):
            return None

        if isinstance(widget, Entry):
            widget.selection_range(0, END)
            widget.icursor(END)
        else:
            widget.tag_add("sel", "1.0", "end-1c")
            widget.mark_set("insert", "end-1c")
        return "break"

    def _on_undo(self, event):
        widget = self._resolve_text_widget(event)
        if not isinstance(widget, (Entry, Text)):
            return None
        if not self._is_widget_editable(widget):
            return "break"
        # For Entry widgets class bindings may run before bind_all; avoid double undo.
        if isinstance(widget, Entry):
            return None
        widget.event_generate("<<Undo>>")
        return "break"

    def _on_redo(self, event):
        widget = self._resolve_text_widget(event)
        if not isinstance(widget, (Entry, Text)):
            return None
        if not self._is_widget_editable(widget):
            return "break"
        widget.event_generate("<<Redo>>")
        return "break"

    def _redirect_stdout(self) -> None:
        self._orig_stdout = sys.stdout

        def callback(text: str) -> None:
            self.window.after(0, self._append_log, text.rstrip())

        sys.stdout = TextRedirector(callback, self._orig_stdout)

    def _clear_placeholder(self, _event=None) -> None:
        if self.input_entry.get().strip() == INPUT_PLACEHOLDER:
            self.input_entry.delete(0, END)

    def _append_log(self, text: str, level: str = "info") -> None:
        self.log_text.configure(state=NORMAL)
        stamp = datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")
        self.log_text.insert(END, f"[{stamp}] ", "timestamp")

        tag = level
        if "[Ошибка]" in text or "[!]" in text:
            tag = "error"
        elif "[Готово]" in text or "готов" in text.lower():
            tag = "success"

        self.log_text.insert(END, text + "\n", tag)
        self.log_text.see(END)
        self.log_text.configure(state="disabled")

    def _append_chat(self, sender: str, text: str, is_ai: bool = False) -> None:
        self.chat_text.configure(state=NORMAL)
        if self.chat_text.get("1.0", END).strip():
            self.chat_text.insert(END, "\n")

        icon = "🤖 " if is_ai else "🧑 "
        self.chat_text.insert(END, f"{icon}{sender}\n", "ai_name" if is_ai else "user_name")
        self.chat_text.insert(END, f"{text}\n", "ai_msg" if is_ai else "user_msg")
        self.chat_text.see(END)
        self.chat_text.configure(state="disabled")

    def _set_status(self, text: str, color: str) -> None:
        self.canvas.itemconfig(self.status_label, text=text, fill=color)
        self.canvas.itemconfig(self.status_dot, fill=color)

    def _set_busy(self, busy: bool, message: str = "") -> None:
        """Переключает UI в занятое/свободное состояние на время фоновых задач."""
        self.is_busy = busy
        if busy:
            self.btn_send.configure(state="disabled", bg=SURFACE)
            self._set_status(message or "Занят", SECONDARY)
            return

        self.btn_send.configure(state="normal", bg=PRIMARY)
        if self.is_initialized:
            self._set_status("Готов к работе", SUCCESS)
        else:
            self._set_status("Не инициализирован", ERROR)

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        if mode == "auto":
            self.btn_mode_auto.configure(bg=PRIMARY, fg=TEXT_PRI, font=("Segoe UI", 10, "bold"))
            self.btn_mode_analysis.configure(bg=SURFACE, fg=TEXT_SEC, font=("Segoe UI", 10))
            self.btn_mode_quote.configure(bg=SURFACE, fg=TEXT_SEC, font=("Segoe UI", 10))
        elif mode == "analysis":
            self.btn_mode_auto.configure(bg=SURFACE, fg=TEXT_SEC, font=("Segoe UI", 10))
            self.btn_mode_analysis.configure(bg=PRIMARY, fg=TEXT_PRI, font=("Segoe UI", 10, "bold"))
            self.btn_mode_quote.configure(bg=SURFACE, fg=TEXT_SEC, font=("Segoe UI", 10))
        else:
            self.btn_mode_auto.configure(bg=SURFACE, fg=TEXT_SEC, font=("Segoe UI", 10))
            self.btn_mode_analysis.configure(bg=SURFACE, fg=TEXT_SEC, font=("Segoe UI", 10))
            self.btn_mode_quote.configure(bg=PRIMARY, fg=TEXT_PRI, font=("Segoe UI", 10, "bold"))

    def _refresh_books(self) -> None:
        self.books_text.configure(state=NORMAL)
        self.books_text.delete("1.0", END)
        if self.backend and self.backend.loaded_books:
            for name in self.backend.loaded_books:
                self.books_text.insert(END, f"📖 {name}\n")
        else:
            self.books_text.insert(END, "Книги не загружены")
        self.books_text.configure(state="disabled")

    def _start_init(self) -> None:
        self._set_status("Инициализация…", SECONDARY)
        self._run_in_thread(self._init_worker)

    def _run_in_thread(self, target, *args) -> None:
        # Единая точка запуска фоновых задач для UI-операций.
        # Все обновления виджетов из worker-потоков возвращаем через window.after.
        threading.Thread(target=target, args=args, daemon=True).start()

    def _apply_provider_env(self, settings: dict) -> None:
        llm_api_key = str(settings.get("llm_api_key", "")).strip()
        embedding_api_key = str(settings.get("embedding_api_key", "")).strip()
        yc_folder_id = str(settings.get("yc_folder_id", "")).strip()
        if llm_api_key:
            os.environ["YC_API_KEY"] = llm_api_key
        elif embedding_api_key:
            os.environ["YC_API_KEY"] = embedding_api_key
        if yc_folder_id:
            os.environ["YC_FOLDER_ID"] = yc_folder_id

    def _init_worker(self) -> None:
        try:
            self._apply_provider_env(self.settings)
            self.backend = PageOracleBackend(log_callback=print)
            self.backend.initialize(
                provider=self.settings.get("provider", "DeepSeek"),
                model_name=self.settings.get("model", "deepseek-chat"),
                llm_api_key=self.settings.get("llm_api_key", ""),
                embedding_api_key=self.settings.get("embedding_api_key", ""),
                embedding_model=self.settings.get("embedding_model", EMBEDDING_OPTIONS[0]),
                temperature=float(self.settings.get("temperature", 0.3)),
                max_tokens=int(self.settings.get("max_tokens", 2048)),
                top_p=float(self.settings.get("top_p", 0.8)),
                score_threshold=float(self.settings.get("score_threshold", 0.6)),
                endpoint=self.settings.get("llm_endpoint", ""),
                embedding_endpoint=self.settings.get("embedding_endpoint", ""),
                custom_providers=self.settings.get("custom_providers", []),
            )
            self.is_initialized = True
            self.window.after(0, self._on_init_ok)
        except Exception as err:
            self.window.after(0, self._on_init_error, str(err))

    def _on_init_ok(self) -> None:
        self._set_status("Готов к работе", SUCCESS)
        self._refresh_books()
        if self.backend and self.backend.load_history(str(self.history_file)):
            self._append_log(f"[История] Загружено сообщений: {self.backend.history_size()}", "info")

    def _on_init_error(self, error_text: str) -> None:
        self._set_status("Ошибка инициализации", ERROR)
        self._append_log(f"[Ошибка] {error_text}", "error")

    def _on_load_book(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Выберите книгу (.txt)",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")],
        )
        if not file_path:
            return
        if not self.backend:
            self._append_log("[Ошибка] Система не инициализирована.", "error")
            return

        self._set_busy(True, "Загрузка книги…")
        self._run_in_thread(self._load_book_worker, file_path)

    def _load_book_worker(self, file_path: str) -> None:
        try:
            if not self.backend:
                raise RuntimeError("Бэкенд не инициализирован")
            self.backend.add_document(file_path)
            self.window.after(0, self._refresh_books)
        except Exception as err:
            self.window.after(0, self._append_log, f"[Ошибка] {err}", "error")
        finally:
            self.window.after(0, self._set_busy, False)

    def _on_send(self) -> None:
        question = self.input_entry.get().strip()
        if not question or self.is_busy:
            return
        if question == INPUT_PLACEHOLDER:
            return
        if not self.is_initialized:
            self._append_chat("Система", "Подождите, идет инициализация…", is_ai=True)
            return

        self.input_entry.delete(0, END)
        self._append_chat("Вы", question, is_ai=False)
        self._show_thinking()
        self._set_busy(True, "PageOracle думает…")

        self._run_in_thread(self._ask_worker, question, self.mode)

    def _show_thinking(self) -> None:
        self.chat_text.configure(state=NORMAL)
        self.chat_text.insert(END, "\n🤖 PageOracle думает…\n", "thinking")
        self.chat_text.see(END)
        self.chat_text.configure(state="disabled")

    def _remove_thinking(self) -> None:
        self.chat_text.configure(state=NORMAL)
        try:
            idx = self.chat_text.search("🤖 PageOracle думает…", "1.0", END)
            if idx:
                self.chat_text.delete(f"{idx} linestart", f"{idx} lineend+1c")
        except Exception:
            pass
        self.chat_text.configure(state="disabled")

    def _ask_worker(self, question: str, mode: str) -> None:
        """Выполняет запрос к backend в фоновом потоке, не блокируя GUI."""
        try:
            if not self.backend:
                raise RuntimeError("Бэкенд не инициализирован")
            answer = self.backend.ask(question, mode=mode)
            self.window.after(0, self._on_answer, answer)
        except Exception as err:
            self.window.after(0, self._on_answer_error, str(err))
        finally:
            self.window.after(0, self._set_busy, False)

    def _on_answer(self, answer: str) -> None:
        """Обрабатывает успешный ответ: чат, автосохранение истории и debug-лог."""
        self._remove_thinking()
        self._append_chat("PageOracle", answer, is_ai=True)
        if self.backend:
            self.backend.save_history(str(self.history_file))
            debug = self.backend.get_last_debug_info()
            self._append_log(
                f"[Router] route={debug.get('route_decision')} "
                f"manual_override={debug.get('manual_override')} "
                f"history={debug.get('history_size')} "
                f"retrieval_query={debug.get('retrieval_query')}",
                "info",
            )

    def _on_answer_error(self, error_text: str) -> None:
        self._remove_thinking()
        self.chat_text.configure(state=NORMAL)
        self.chat_text.insert(END, f"\n🤖 [Ошибка] {error_text}\n", "error_msg")
        self.chat_text.see(END)
        self.chat_text.configure(state="disabled")

    def _on_clear_chat(self) -> None:
        self.chat_text.configure(state=NORMAL)
        self.chat_text.delete("1.0", END)
        self.chat_text.configure(state="disabled")
        self._append_chat("PageOracle", "Окно чата очищено. История памяти не удалена.", is_ai=True)

    def _on_clear_history(self) -> None:
        if not self.backend:
            return
        if not messagebox.askyesno("Очистка истории", "Удалить историю диалога из файла?"):
            return
        self.backend.clear_history()
        try:
            if self.history_file.exists():
                self.history_file.unlink()
        except Exception as err:
            self._append_log(f"[История] Не удалось удалить файл: {err}", "error")
        self._append_log("[История] История диалога очищена.", "info")
        self._append_chat("PageOracle", "История памяти очищена. Можем начать новый диалог.", is_ai=True)

    def _on_clear_logs(self) -> None:
        self.log_text.configure(state=NORMAL)
        self.log_text.delete("1.0", END)
        self.log_text.configure(state="disabled")

    def _on_settings(self) -> None:
        SettingsWindow(self.window, self.settings, PROVIDERS, self._apply_settings)

    def _apply_settings(self, new_settings: dict) -> None:
        self.settings = new_settings
        if not self.backend:
            return
        self._set_busy(True, "Переключение модели…")
        self._run_in_thread(self._switch_model_worker, new_settings)

    def _switch_model_worker(self, settings: dict) -> None:
        try:
            if not self.backend:
                return
            self._apply_provider_env(settings)
            embedding_model = settings.get("embedding_model", EMBEDDING_OPTIONS[0])
            embedding_endpoint = settings.get("embedding_endpoint", "")
            ok = self.backend.set_model(
                settings["provider"],
                settings["model"],
                settings.get("llm_api_key", ""),
                temperature=float(settings.get("temperature", 0.3)),
                max_tokens=int(settings.get("max_tokens", 2048)),
                top_p=float(settings.get("top_p", 0.8)),
                endpoint=settings.get("llm_endpoint", ""),
                custom_providers=settings.get("custom_providers", []),
            )
            embedding_changed = embedding_model != self.backend.embedding_model_name or embedding_endpoint != getattr(
                self.backend, "embedding_endpoint", ""
            )
            if (
                ok
                and embedding_changed
                and not self.backend.set_embeddings(
                    embedding_model,
                    settings.get("embedding_api_key", ""),
                    endpoint=embedding_endpoint,
                )
            ):
                ok = False
            if ok and not self.backend.set_score_threshold(float(settings.get("score_threshold", 0.6))):
                ok = False

            if ok:
                self.window.after(
                    0,
                    self._append_log,
                    f"[Готово] Переключено на {settings['provider']} / {settings['model']} "
                    f"embedding={settings.get('embedding_model', EMBEDDING_OPTIONS[0])} "
                    f"(temperature={settings.get('temperature', 0.3)}, "
                    f"max_tokens={settings.get('max_tokens', 2048)}, "
                    f"top_p={settings.get('top_p', 0.8)}, "
                    f"score_threshold={settings.get('score_threshold', 0.6)})",
                    "success",
                )
            else:
                self.window.after(0, self._append_log, "[Ошибка] Не удалось переключить модель.", "error")
        except Exception as err:
            self.window.after(0, self._append_log, f"[Ошибка] {err}", "error")
        finally:
            self.window.after(0, self._set_busy, False)

    def run(self) -> None:
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
        self.window.mainloop()

    def _on_close(self) -> None:
        if hasattr(self, "_orig_stdout"):
            sys.stdout = self._orig_stdout
        with contextlib.suppress(TclError):
            self.text_context_menu.unpost()
        self.window.destroy()


def main() -> None:
    app = PageOracleApp()
    app.run()


if __name__ == "__main__":
    main()
