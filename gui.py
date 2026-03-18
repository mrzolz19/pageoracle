"""
PageOracle GUI (Tkinter-Designer style layout)
Интерфейс в стиле Tkinter-Designer:
- Canvas как основа лейаута
- Точное позиционирование блоков по Figma-структуре
- Привязка к backend из main.py
"""
from __future__ import annotations

import io
import json
import sys
import threading
from datetime import datetime
from pathlib import Path
from tkinter import (
	END,
	NORMAL,
	WORD,
	Button,
	Canvas,
	Entry,
	StringVar,
	Text,
	Tk,
	Toplevel,
	filedialog,
	messagebox,
)
from tkinter import ttk

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


class TextRedirector(io.TextIOBase):
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
	defaults = {
		"provider": "DeepSeek",
		"model": "deepseek-chat",
		"api_key": "",
		"temperature": 0.3,
		"max_tokens": 2048,
		"top_p": 0.8,
		"score_threshold": 0.6,
	}
	if SETTINGS_FILE.exists():
		try:
			with open(SETTINGS_FILE, "r", encoding="utf-8") as file:
				payload = json.load(file)
			defaults.update(payload)
		except Exception:
			pass
	return defaults


def save_settings(data: dict) -> None:
	with open(SETTINGS_FILE, "w", encoding="utf-8") as file:
		json.dump(data, file, ensure_ascii=False, indent=2)


class SettingsWindow:
	def __init__(self, parent, current: dict, providers: dict, on_save):
		self.providers = providers
		self.on_save = on_save
		win_width = 560
		win_height = 640

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

		canvas.create_rectangle(16, 16, 544, 624, fill=BG, outline=BORDER, width=1)
		canvas.create_text(34, 40, anchor="nw", text="⚙ Настройки", fill=TEXT_PRI, font=("Segoe UI", 18, "bold"))

		canvas.create_text(34, 92, anchor="nw", text="Провайдер ИИ", fill=TEXT_SEC, font=("Segoe UI", 11))
		self.provider_var = StringVar(value=current.get("provider", "DeepSeek"))
		self.provider_cb = ttk.Combobox(
			self.win,
			textvariable=self.provider_var,
			values=list(providers.keys()),
			state="readonly",
			font=("Segoe UI", 11),
		)
		self.provider_cb.place(x=34, y=116, width=492, height=36)
		self.provider_cb.bind("<<ComboboxSelected>>", self._on_provider_change)

		canvas.create_text(34, 176, anchor="nw", text="Модель", fill=TEXT_SEC, font=("Segoe UI", 11))
		self.model_var = StringVar(value=current.get("model", "deepseek-chat"))
		self.model_cb = ttk.Combobox(self.win, textvariable=self.model_var, font=("Segoe UI", 11), state="readonly")
		self._update_model_list()
		self.model_cb.place(x=34, y=200, width=492, height=36)

		canvas.create_text(34, 260, anchor="nw", text="API Ключ", fill=TEXT_SEC, font=("Segoe UI", 11))
		self.api_entry = Entry(
			self.win,
			font=("Segoe UI", 11),
			show="*",
			bg=INPUT_BG,
			fg=TEXT_PRI,
			insertbackground=TEXT_PRI,
			relief="flat",
			bd=0,
		)
		self.api_entry.place(x=34, y=284, width=492, height=36)
		self.api_entry.insert(0, current.get("api_key", ""))

		self.show_var = StringVar(value="0")
		show_btn = ttk.Checkbutton(
			self.win,
			text="Показать ключ",
			variable=self.show_var,
			command=self._toggle_key,
		)
		show_btn.place(x=34, y=326)

		canvas.create_text(34, 364, anchor="nw", text="Temperature", fill=TEXT_SEC, font=("Segoe UI", 11))
		self.temperature_entry = Entry(
			self.win,
			font=("Segoe UI", 11),
			bg=INPUT_BG,
			fg=TEXT_PRI,
			insertbackground=TEXT_PRI,
			relief="flat",
			bd=0,
		)
		self.temperature_entry.place(x=34, y=388, width=236, height=36)
		self.temperature_entry.insert(0, str(current.get("temperature", 0.3)))

		canvas.create_text(290, 364, anchor="nw", text="Max Tokens", fill=TEXT_SEC, font=("Segoe UI", 11))
		self.max_tokens_entry = Entry(
			self.win,
			font=("Segoe UI", 11),
			bg=INPUT_BG,
			fg=TEXT_PRI,
			insertbackground=TEXT_PRI,
			relief="flat",
			bd=0,
		)
		self.max_tokens_entry.place(x=290, y=388, width=236, height=36)
		self.max_tokens_entry.insert(0, str(current.get("max_tokens", 2048)))

		canvas.create_text(34, 436, anchor="nw", text="Top P", fill=TEXT_SEC, font=("Segoe UI", 11))
		self.top_p_entry = Entry(
			self.win,
			font=("Segoe UI", 11),
			bg=INPUT_BG,
			fg=TEXT_PRI,
			insertbackground=TEXT_PRI,
			relief="flat",
			bd=0,
		)
		self.top_p_entry.place(x=34, y=460, width=236, height=36)
		self.top_p_entry.insert(0, str(current.get("top_p", 0.8)))

		canvas.create_text(290, 436, anchor="nw", text="Score Threshold", fill=TEXT_SEC, font=("Segoe UI", 11))
		self.score_threshold_entry = Entry(
			self.win,
			font=("Segoe UI", 11),
			bg=INPUT_BG,
			fg=TEXT_PRI,
			insertbackground=TEXT_PRI,
			relief="flat",
			bd=0,
		)
		self.score_threshold_entry.place(x=290, y=460, width=236, height=36)
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
		cancel_btn.place(x=306, y=568, width=100, height=40)

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
		save_btn.place(x=416, y=568, width=110, height=40)

	def _toggle_key(self) -> None:
		self.api_entry.configure(show="" if self.show_var.get() == "1" else "*")

	def _on_provider_change(self, _evt=None) -> None:
		self._update_model_list()

	def _update_model_list(self) -> None:
		provider = self.provider_var.get()
		models = self.providers.get(provider, {}).get("models", [])
		self.model_cb["values"] = models
		if models and self.model_var.get() not in models:
			self.model_var.set(models[0])

	def _save(self) -> None:
		try:
			temperature = float(self.temperature_entry.get().strip())
		except ValueError:
			messagebox.showwarning("Внимание", "Temperature должен быть числом.", parent=self.win)
			return

		if temperature < 0 or temperature > 2:
			messagebox.showwarning("Внимание", "Temperature должен быть в диапазоне 0..2.", parent=self.win)
			return

		try:
			max_tokens = int(self.max_tokens_entry.get().strip())
		except ValueError:
			messagebox.showwarning("Внимание", "Max Tokens должен быть целым числом.", parent=self.win)
			return

		if max_tokens < 1:
			messagebox.showwarning("Внимание", "Max Tokens должен быть больше 0.", parent=self.win)
			return

		try:
			top_p = float(self.top_p_entry.get().strip())
		except ValueError:
			messagebox.showwarning("Внимание", "Top P должен быть числом.", parent=self.win)
			return

		if top_p < 0 or top_p > 1:
			messagebox.showwarning("Внимание", "Top P должен быть в диапазоне 0..1.", parent=self.win)
			return

		try:
			score_threshold = float(self.score_threshold_entry.get().strip())
		except ValueError:
			messagebox.showwarning("Внимание", "Score Threshold должен быть числом.", parent=self.win)
			return

		if score_threshold < 0 or score_threshold > 1:
			messagebox.showwarning("Внимание", "Score Threshold должен быть в диапазоне 0..1.", parent=self.win)
			return

		data = {
			"provider": self.provider_var.get(),
			"model": self.model_var.get(),
			"api_key": self.api_entry.get().strip(),
			"temperature": temperature,
			"max_tokens": max_tokens,
			"top_p": top_p,
			"score_threshold": score_threshold,
		}
		if not data["api_key"]:
			messagebox.showwarning("Внимание", "Введите API ключ.", parent=self.win)
			return
		save_settings(data)
		self.on_save(data)
		self.win.destroy()


class PageOracleApp:
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
		self.canvas.create_text(16, 57, anchor="nw", text="AI Book Reader", fill=TEXT_SEC, font=("Segoe UI", 10))
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

		self.canvas.create_text(16, 184, anchor="nw", text="Загруженные книги", fill=TEXT_SEC, font=("Segoe UI", 10, "bold"))

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

		self.status_dot = self.canvas.create_oval(1065, 23, 1073, 31, fill=SECONDARY, outline="")
		self.status_label = self.canvas.create_text(
			1081,
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
		self.input_entry.place(x=324, y=598, width=712, height=20)
		self.input_entry.bind("<Return>", lambda _evt: self._on_send())
		self.input_entry.insert(0, "Задайте вопрос по загруженным книгам…")
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
			text="🗑Ч",
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

		self.canvas.create_text(304, 645, anchor="nw", text="Системные логи", fill=TEXT_SEC, font=("Segoe UI", 9, "bold"))

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
		self.btn_clear_logs.place(x=1126, y=642, width=50, height=16)

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

	def _redirect_stdout(self) -> None:
		self._orig_stdout = sys.stdout
		sys.stdout = TextRedirector(lambda text: self.window.after(0, self._append_log, text.rstrip()), self._orig_stdout)

	def _clear_placeholder(self, _event=None) -> None:
		if self.input_entry.get().strip() == "Задайте вопрос по загруженным книгам…":
			self.input_entry.delete(0, END)

	def _append_log(self, text: str, level: str = "info") -> None:
		self.log_text.configure(state=NORMAL)
		stamp = datetime.now().strftime("%H:%M:%S")
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
		threading.Thread(target=self._init_worker, daemon=True).start()

	def _init_worker(self) -> None:
		try:
			self.backend = PageOracleBackend(log_callback=print)
			self.backend.initialize(
				provider=self.settings.get("provider", "DeepSeek"),
				model_name=self.settings.get("model", "deepseek-chat"),
				api_key=self.settings.get("api_key", ""),
				temperature=float(self.settings.get("temperature", 0.3)),
				max_tokens=int(self.settings.get("max_tokens", 2048)),
				top_p=float(self.settings.get("top_p", 0.8)),
				score_threshold=float(self.settings.get("score_threshold", 0.6)),
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
		threading.Thread(target=self._load_book_worker, args=(file_path,), daemon=True).start()

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
		if question == "Задайте вопрос по загруженным книгам…":
			return
		if not self.is_initialized:
			self._append_chat("Система", "Подождите, идет инициализация…", is_ai=True)
			return

		self.input_entry.delete(0, END)
		self._append_chat("Вы", question, is_ai=False)
		self._show_thinking()
		self._set_busy(True, "PageOracle думает…")

		threading.Thread(target=self._ask_worker, args=(question, self.mode), daemon=True).start()

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
		self._remove_thinking()
		self._append_chat("PageOracle", answer, is_ai=True)
		if self.backend:
			self.backend.save_history(str(self.history_file))
			debug = self.backend.get_last_debug_info()
			self._append_log(
				f"[Router] route={debug.get('route_decision')} "
				f"manual_override={debug.get('manual_override')} "
				f"history={debug.get('history_size')}",
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
		if not messagebox.askyesno("Очистка истории", "Удалить историю диалога из памяти и файла?"):
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
		threading.Thread(target=self._switch_model_worker, args=(new_settings,), daemon=True).start()

	def _switch_model_worker(self, settings: dict) -> None:
		try:
			if not self.backend:
				return
			ok = self.backend.set_model(
				settings["provider"],
				settings["model"],
				settings["api_key"],
				temperature=float(settings.get("temperature", 0.3)),
				max_tokens=int(settings.get("max_tokens", 2048)),
				top_p=float(settings.get("top_p", 0.8)),
			)
			if ok and not self.backend.set_score_threshold(float(settings.get("score_threshold", 0.6))):
				ok = False

			if ok:
				self.window.after(
					0,
					self._append_log,
					f"[Готово] Переключено на {settings['provider']} / {settings['model']} "
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
		sys.stdout = self._orig_stdout
		self.window.destroy()


def main() -> None:
	app = PageOracleApp()
	app.run()


if __name__ == "__main__":
	main()
