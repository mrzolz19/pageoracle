"""Web GUI for PageOracle.

Flask-слой над существующим PageOracleBackend из main.py:
- хранит состояние интерфейса,
- принимает действия пользователя через REST API,
- сериализует состояние для фронтенда.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from flask import Flask, jsonify, render_template, request

from main import PROVIDERS, PageOracleBackend

BASE_DIR = Path(__file__).parent
SETTINGS_FILE = BASE_DIR / "settings.json"
HISTORY_FILE = BASE_DIR / "chat_history.json"

SUCCESS = "#4CAF50"
ERROR = "#FF6B6B"
SECONDARY = "#4ECDC4"

EMBEDDING_OPTIONS = [
    "nvidia/llama-nemotron-embed-vl-1b-v2:free",
    "BAAI/bge-m3",
    "text-search-doc/latest",
]


def now_ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def load_settings() -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "provider": "YandexGPT",
        "model": "yandexgpt-5.1/latest",
        "embedding_model": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
        "llm_api_key": "",
        "embedding_api_key": "",
        "yc_folder_id": "",
        "temperature": 0.2,
        "max_tokens": 4096,
        "top_p": 0.9,
        "score_threshold": 0.6,
    }
    if SETTINGS_FILE.exists():
        try:
            payload = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            legacy_api_key = str(payload.get("api_key", "")).strip()
            if legacy_api_key:
                payload.setdefault("llm_api_key", legacy_api_key)
                payload.setdefault("embedding_api_key", legacy_api_key)
            defaults.update(payload)
        except Exception:
            pass
    return defaults


def save_settings(data: dict[str, Any]) -> None:
    SETTINGS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class WebAppState:
    """Потокобезопасное in-memory состояние веб-интерфейса."""

    def __init__(self) -> None:
        # state_lock защищает чтение/обновление структуры состояния.
        self.state_lock = threading.RLock()
        # operation_lock сериализует долгие операции (chat/upload/settings),
        # чтобы не запускать их параллельно и не ломать backend state.
        self.operation_lock = threading.Lock()
        self.settings = load_settings()
        self.backend: PageOracleBackend | None = None
        self.is_initialized = False
        self.is_busy = False
        self.status_text = "Инициализация…"
        self.status_color = SECONDARY
        self.mode = "auto"
        self.chat_messages: list[dict[str, Any]] = [
            {
                "sender": "PageOracle",
                "is_ai": True,
                "text": "Добро пожаловать! Загрузите книгу (.txt) через боковую панель и задайте вопрос.",
            }
        ]
        self.logs: list[dict[str, str]] = []
        self._append_log("[Инициализация] Веб-интерфейс запущен.", "info")

    def _append_log(self, text: str, level: str = "info") -> None:
        # Нормализуем уровень, чтобы фронтенд мог стабильно красить логи.
        normalized = level
        text_l = text.lower()
        if "[ошибка]" in text_l or "[!]" in text_l:
            normalized = "error"
        elif "[готово]" in text_l or "готов" in text_l:
            normalized = "success"

        with self.state_lock:
            self.logs.append({"ts": now_ts(), "text": text, "level": normalized})
            self.logs = self.logs[-500:]

    def backend_log_callback(self, text: str) -> None:
        """Callback для backend.print/log -> web logs."""
        for line in str(text).splitlines():
            line = line.strip()
            if line:
                self._append_log(line, "info")

    def append_chat(self, sender: str, text: str, is_ai: bool) -> None:
        """Добавляет сообщение в чат и ограничивает длину истории на UI."""
        with self.state_lock:
            self.chat_messages.append(
                {
                    "sender": sender,
                    "is_ai": is_ai,
                    "text": text,
                }
            )
            self.chat_messages = self.chat_messages[-200:]

    def set_busy(self, busy: bool, message: str | None = None) -> None:
        """Переключает глобальный busy-флаг и текст статуса."""
        with self.state_lock:
            self.is_busy = busy
            if busy:
                self.status_text = message or "Занят"
                self.status_color = SECONDARY
            else:
                if self.is_initialized:
                    self.status_text = "Готов к работе"
                    self.status_color = SUCCESS
                else:
                    self.status_text = "Не инициализирован"
                    self.status_color = ERROR

    def snapshot(self) -> dict[str, Any]:
        """Снимок состояния для /api/state (без мутаций)."""
        with self.state_lock:
            return {
                "settings": self.settings,
                "is_initialized": self.is_initialized,
                "is_busy": self.is_busy,
                "status": {
                    "text": self.status_text,
                    "color": self.status_color,
                },
                "mode": self.mode,
                "chat": list(self.chat_messages),
                "logs": list(self.logs),
                "books": list((self.backend.loaded_books if self.backend else [])),
                "providers": PROVIDERS,
                "embedding_options": EMBEDDING_OPTIONS,
            }


state = WebAppState()
app = Flask(__name__)


def _apply_provider_env(settings: dict[str, Any]) -> None:
    """Подготавливает env-переменные, которые ожидает backend/providers."""
    llm_api_key = str(settings.get("llm_api_key", "")).strip()
    embedding_api_key = str(settings.get("embedding_api_key", "")).strip()
    yc_folder_id = str(settings.get("yc_folder_id", "")).strip()
    if llm_api_key:
        os.environ["YC_API_KEY"] = llm_api_key
    elif embedding_api_key:
        os.environ["YC_API_KEY"] = embedding_api_key
    if yc_folder_id:
        os.environ["YC_FOLDER_ID"] = yc_folder_id


def _parse_float(
    payload: dict[str, Any],
    field: str,
    label: str,
    min_value: float,
    max_value: float,
) -> float:
    raw = str(payload.get(field, "")).strip()
    try:
        value = float(raw)
    except ValueError as err:
        raise ValueError(f"{label} должен быть числом.") from err
    if value < min_value or value > max_value:
        raise ValueError(f"{label} должен быть в диапазоне {min_value:g}..{max_value:g}.")
    return value


def _parse_int(payload: dict[str, Any], field: str, label: str, min_value: int) -> int:
    raw = str(payload.get(field, "")).strip()
    try:
        value = int(raw)
    except ValueError as err:
        raise ValueError(f"{label} должен быть целым числом.") from err
    if value < min_value:
        raise ValueError(f"{label} должен быть больше или равен {min_value}.")
    return value


def _validate_settings(payload: dict[str, Any]) -> dict[str, Any]:
    """Валидирует входные настройки формы и возвращает нормализованный словарь."""
    provider = str(payload.get("provider", "")).strip()
    model = str(payload.get("model", "")).strip()
    embedding_model = str(payload.get("embedding_model", "")).strip()
    llm_api_key = str(payload.get("llm_api_key", "")).strip()
    embedding_api_key = str(payload.get("embedding_api_key", "")).strip()
    yc_folder_id = str(payload.get("yc_folder_id", "")).strip()

    if provider not in PROVIDERS:
        raise ValueError("Неизвестный провайдер.")

    models = PROVIDERS[provider].get("models", [])
    if model not in models:
        raise ValueError("Выберите корректную модель для провайдера.")

    if embedding_model not in EMBEDDING_OPTIONS:
        raise ValueError("Выберите корректную embedding-модель.")

    if not llm_api_key:
        raise ValueError("Введите API ключ LLM.")

    if embedding_model == "nvidia/llama-nemotron-embed-vl-1b-v2:free" and not embedding_api_key:
        raise ValueError("Введите API ключ Embedding для OpenRouter модели.")

    uses_yandex = provider == "YandexGPT" or embedding_model == "text-search-doc/latest"
    if uses_yandex and not yc_folder_id:
        raise ValueError("Введите YC_FOLDER_ID для Yandex моделей.")

    return {
        "provider": provider,
        "model": model,
        "embedding_model": embedding_model,
        "llm_api_key": llm_api_key,
        "embedding_api_key": embedding_api_key,
        "yc_folder_id": yc_folder_id,
        "temperature": _parse_float(payload, "temperature", "Temperature", 0.0, 2.0),
        "max_tokens": _parse_int(payload, "max_tokens", "Max Tokens", 1),
        "top_p": _parse_float(payload, "top_p", "Top P", 0.0, 1.0),
        "score_threshold": _parse_float(payload, "score_threshold", "Score Threshold", 0.0, 1.0),
    }


def initialize_backend() -> None:
    """Полная инициализация backend из текущих settings."""
    with state.operation_lock:
        state.set_busy(True, "Инициализация…")
        try:
            _apply_provider_env(state.settings)
            backend = PageOracleBackend(log_callback=state.backend_log_callback)
            backend.initialize(
                provider=state.settings.get("provider", "YandexGPT"),
                model_name=state.settings.get("model", "yandexgpt-5.1/latest"),
                llm_api_key=state.settings.get("llm_api_key", ""),
                embedding_api_key=state.settings.get("embedding_api_key", ""),
                embedding_model=state.settings.get("embedding_model", EMBEDDING_OPTIONS[0]),
                temperature=float(state.settings.get("temperature", 0.2)),
                max_tokens=int(state.settings.get("max_tokens", 4096)),
                top_p=float(state.settings.get("top_p", 0.9)),
                score_threshold=float(state.settings.get("score_threshold", 0.6)),
            )

            with state.state_lock:
                state.backend = backend
                state.is_initialized = True

            if backend.load_history(str(HISTORY_FILE)):
                state._append_log(
                    f"[История] Загружено сообщений: {backend.history_size()}",
                    "info",
                )
        except Exception as err:
            with state.state_lock:
                state.is_initialized = False
            state._append_log(f"[Ошибка] {err}", "error")
        finally:
            state.set_busy(False)


def initialize_backend_async() -> None:
    """Запускает initialize_backend в фоне, не блокируя Flask worker."""
    thread = threading.Thread(target=initialize_backend, daemon=True)
    thread.start()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/state")
def api_state():
    # Основная точка синхронизации состояния фронта.
    return jsonify({"ok": True, "data": state.snapshot()})


@app.post("/api/init")
def api_init():
    if state.is_busy:
        return jsonify({"ok": False, "error": "Система уже занята."}), 409
    initialize_backend_async()
    return jsonify({"ok": True})


@app.post("/api/upload")
def api_upload():
    # Загрузка книги делается через временный файл, а затем add_document
    # переносит/индексирует его штатной логикой backend.
    if state.is_busy:
        return jsonify({"ok": False, "error": "Система занята. Подождите завершения операции."}), 409

    if not state.is_initialized or not state.backend:
        return jsonify({"ok": False, "error": "Система не инициализирована."}), 400

    uploaded = request.files.get("book")
    if not uploaded:
        return jsonify({"ok": False, "error": "Файл не передан."}), 400

    original_name = str(uploaded.filename or "").strip()
    if not original_name.lower().endswith(".txt"):
        return jsonify({"ok": False, "error": "Поддерживаются только .txt файлы."}), 400

    with state.operation_lock:
        state.set_busy(True, "Загрузка книги…")
        temp_path: Path | None = None
        try:
            with NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                uploaded.save(tmp)
                temp_path = Path(tmp.name)

            state.backend.add_document(str(temp_path))
            return jsonify({"ok": True})
        except Exception as err:
            state._append_log(f"[Ошибка] {err}", "error")
            return jsonify({"ok": False, "error": str(err)}), 500
        finally:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            state.set_busy(False)


@app.post("/api/chat")
def api_chat():
    # Один запрос чата = атомарная операция: добавить вопрос,
    # получить ответ, обновить историю, записать debug-лог.
    if state.is_busy:
        return jsonify({"ok": False, "error": "Система занята. Подождите завершения операции."}), 409

    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    mode = str(payload.get("mode", "auto")).strip()

    if not question:
        return jsonify({"ok": False, "error": "Вопрос пустой."}), 400

    if mode not in {"auto", "analysis", "quote"}:
        mode = "auto"

    if not state.is_initialized or not state.backend:
        return jsonify({"ok": False, "error": "Система еще инициализируется."}), 400

    with state.operation_lock:
        state.set_busy(True, "PageOracle думает…")
        try:
            state.append_chat("Вы", question, is_ai=False)
            answer = state.backend.ask(question, mode=mode)
            state.append_chat("PageOracle", answer, is_ai=True)
            state.backend.save_history(str(HISTORY_FILE))

            debug = state.backend.get_last_debug_info()
            state._append_log(
                "[Router] "
                f"route={debug.get('route_decision')} "
                f"manual_override={debug.get('manual_override')} "
                f"history={debug.get('history_size')} "
                f"retrieval_query={debug.get('retrieval_query')}",
                "info",
            )
            return jsonify({"ok": True, "answer": answer})
        except Exception as err:
            state._append_log(f"[Ошибка] {err}", "error")
            state.append_chat("PageOracle", f"[Ошибка] {err}", is_ai=True)
            return jsonify({"ok": False, "error": str(err)}), 500
        finally:
            state.set_busy(False)


@app.post("/api/settings")
def api_settings():
    # Сохраняем настройки сразу, а затем пытаемся применить их к backend.
    payload = request.get_json(silent=True) or {}
    try:
        validated = _validate_settings(payload)
    except ValueError as err:
        return jsonify({"ok": False, "error": str(err)}), 400

    with state.state_lock:
        state.settings = validated
    save_settings(validated)

    if state.is_busy:
        return jsonify({"ok": True, "warning": "Настройки сохранены. Примените после завершения текущей операции."})

    if not state.backend or not state.is_initialized:
        initialize_backend_async()
        return jsonify({"ok": True, "warning": "Настройки сохранены. Запущена инициализация."})

    with state.operation_lock:
        state.set_busy(True, "Переключение модели…")
        try:
            _apply_provider_env(validated)
            ok = state.backend.set_model(
                validated["provider"],
                validated["model"],
                validated.get("llm_api_key", ""),
                temperature=float(validated.get("temperature", 0.2)),
                max_tokens=int(validated.get("max_tokens", 4096)),
                top_p=float(validated.get("top_p", 0.9)),
            )

            if ok and validated["embedding_model"] != state.backend.embedding_model_name:
                ok = state.backend.set_embeddings(
                    validated["embedding_model"],
                    validated.get("embedding_api_key", ""),
                )

            if ok:
                ok = state.backend.set_score_threshold(
                    float(validated.get("score_threshold", 0.6))
                )

            if not ok:
                return jsonify({"ok": False, "error": "Не удалось применить настройки."}), 500

            state._append_log(
                "[Готово] Переключено на "
                f"{validated['provider']} / {validated['model']} "
                f"embedding={validated['embedding_model']} "
                f"(temperature={validated['temperature']}, "
                f"max_tokens={validated['max_tokens']}, "
                f"top_p={validated['top_p']}, "
                f"score_threshold={validated['score_threshold']})",
                "success",
            )
            return jsonify({"ok": True})
        except Exception as err:
            state._append_log(f"[Ошибка] {err}", "error")
            return jsonify({"ok": False, "error": str(err)}), 500
        finally:
            state.set_busy(False)


@app.post("/api/mode")
def api_mode():
    payload = request.get_json(silent=True) or {}
    mode = str(payload.get("mode", "auto")).strip()
    if mode not in {"auto", "analysis", "quote"}:
        return jsonify({"ok": False, "error": "Некорректный режим."}), 400

    with state.state_lock:
        state.mode = mode
    return jsonify({"ok": True})


@app.post("/api/chat/clear")
def api_clear_chat():
    with state.state_lock:
        state.chat_messages = [
            {
                "sender": "PageOracle",
                "is_ai": True,
                "text": "Окно чата очищено. История памяти не удалена.",
            }
        ]
    return jsonify({"ok": True})


@app.post("/api/history/clear")
def api_clear_history():
    # Очищаем и in-memory историю backend, и persisted chat_history.json.
    if not state.backend:
        return jsonify({"ok": False, "error": "Бэкенд не инициализирован."}), 400

    with state.operation_lock:
        state.set_busy(True, "Очистка истории…")
        try:
            state.backend.clear_history()
            if HISTORY_FILE.exists():
                HISTORY_FILE.unlink()
            state._append_log("[История] История диалога очищена.", "info")
            state.append_chat("PageOracle", "История памяти очищена. Можем начать новый диалог.", is_ai=True)
            return jsonify({"ok": True})
        except Exception as err:
            state._append_log(f"[История] Не удалось удалить файл: {err}", "error")
            return jsonify({"ok": False, "error": str(err)}), 500
        finally:
            state.set_busy(False)


@app.post("/api/logs/clear")
def api_clear_logs():
    with state.state_lock:
        state.logs = []
    return jsonify({"ok": True})


initialize_backend_async()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
