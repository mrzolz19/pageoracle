import re
import json
import shutil
import importlib
import gc
import time
import threading
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
from typing import Any
from typing import TypedDict, Literal, cast, Annotated
from pydantic import BaseModel, Field, ConfigDict
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.retrievers import BaseRetriever
from langchain.tools import tool
from sentence_transformers import CrossEncoder
import hashlib
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition



RETRIEVER_K = 20
RETRIEVER_FETCH_K = 50
RERANK_TOP_K = 7

EMBEDDING_MODELS = {
    "nvidia/llama-nemotron-embed-vl-1b-v2:free": {
        "provider": "openrouter",
        "query_prefix": "",
        "doc_prefix": "",
    },
    "BAAI/bge-m3": {
        "provider": "huggingface",
        "query_prefix": "search_query: ",
        "doc_prefix": "search_document: ",
    },
    "text-search-doc/latest": {
        "provider": "yandex",
        "query_prefix": "",
        "doc_prefix": "",
    },
}
DEFAULT_EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"

# ───────────────────── провайдеры LLM (для GUI настроек) ─────────────
PROVIDERS = {
    "DeepSeek": {
        "class": "ChatDeepSeek",
        "package": "langchain_deepseek",
        "env_key": "DEEPSEEK_API_KEY",
        "models": ["deepseek-chat"],
    },
    "OpenAI": {
        "class": "ChatOpenAI",
        "package": "langchain_openai",
        "env_key": "OPENAI_API_KEY",
        "models": [
            "gpt-5.4-pro-2026-03-05",
            "gpt-5.4-2026-03-05",
            "gpt-5-mini-2025-08-07",
            "gpt-5-nano-2025-08-07",
        ],
    },
    "Anthropic": {
        "class": "ChatAnthropic",
        "package": "langchain_anthropic",
        "env_key": "ANTHROPIC_API_KEY",
        "models": ["claude-sonnet-4-6", "claude-opus-4-6"],
    },
    "Google": {
        "class": "ChatGoogleGenerativeAI",
        "package": "langchain_google_genai",
        "env_key": "GOOGLE_API_KEY",
        "models": [
            "gemini-3.1-pro-preview",
            "gemini-3-flash-preview",
            "gemini-3.1-flash-lite-preview",
        ],
    },
    "OpenRouter": {
        "class": "ChatOpenRouter",
        "package": "langchain_openrouter",
        "env_key": "OPENROUTER_API_KEY",
        "models": [
            "deepseek/deepseek-chat",
            "qwen/qwen3-235b-a22b",
            "stepfun/step-3.5-flash:free",
            "arcee-ai/trinity-large-preview:free",
        ],
    },
    "YandexGPT": {
        "class": "ChatYandexGPT",
        "package": "langchain_community.chat_models",
        "env_key": "YC_API_KEY",
        "models": [
            "aliceai-llm/latest",
            "deepseek-v32/latest",
            "yandexgpt-5.1/latest",
            "yandexgpt-5-pro/latest",
            "yandexgpt-5-lite/latest",
        ],
    },
    "GigaChat": {
        "class": "GigaChat",
        "package": "langchain_gigachat",
        "env_key": "GIGACHAT_CREDENTIALS",
        "verify_ssl_certs": False,
        "models": ["gigachat-2", "gigachat-2-pro", "gigachat-2-max"],
    }
}

# ───────────────────────── парсинг структуры книги ───────────────────
_ORDINAL_MAP = {
    "ПЕРВАЯ": "1",
    "ВТОРАЯ": "2",
    "ТРЕТЬЯ": "3",
    "ЧЕТВЁРТАЯ": "4",
    "ЧЕТВЕРТАЯ": "4",
    "ПЯТАЯ": "5",
    "ШЕСТАЯ": "6",
    "СЕДЬМАЯ": "7",
    "ВОСЬМАЯ": "8",
}
_ROMAN_MAP = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
    "XI": 11,
    "XII": 12,
    "XIII": 13,
    "XIV": 14,
    "XV": 15,
    "XVI": 16,
    "XVII": 17,
    "XVIII": 18,
    "XIX": 19,
    "XX": 20,
}

_PART_RE = re.compile(r"^\s*ЧАСТЬ\s+(\w+)\s*$", re.IGNORECASE)
_CHAPTER_RE = re.compile(r"^\s*(?:Глава|ГЛАВА)\s+([IVXLCDM]+|\d+)\s*$")
_ROMAN_RE = re.compile(r"^\s*([IVXLCDM]{1,7})\.?\s*$")
_EPILOGUE_RE = re.compile(r"^\s*ЭПИЛОГ\s*$", re.IGNORECASE)


def annotate_book(docs: list[Document], book_title: str) -> list[Document]:
    result: list[Document] = []
    for doc in docs:
        lines = doc.page_content.split("\n")
        current_part = "Вступление"
        current_chapter = "—"
        seg_lines: list[str] = []

        def flush(part: str, chapter: str) -> None:
            text = "\n".join(seg_lines).strip()
            if text:
                result.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": doc.metadata.get("source", ""),
                            "book_title": book_title,
                            "part": part,
                            "chapter": chapter,
                        },
                    )
                )
            seg_lines.clear()

        for line in lines:
            pm = _PART_RE.match(line)
            cm = _CHAPTER_RE.match(line)
            rm = _ROMAN_RE.match(line)
            em = _EPILOGUE_RE.match(line)

            if pm or em:
                flush(current_part, current_chapter)
                if em:
                    current_part = "Эпилог"
                    current_chapter = "—"
                else:
                    ordinal = pm.group(1).upper() if pm else ""
                    current_part = f"Часть {_ORDINAL_MAP.get(ordinal, ordinal)}"
                    current_chapter = "—"
                seg_lines.append(line)
            elif cm or rm:
                flush(current_part, current_chapter)
                if cm:
                    chap_raw = cm.group(1)
                elif rm:
                    chap_raw = rm.group(1)
                else:
                    chap_raw = ""
                chap_str = chap_raw.upper()
                current_chapter = f"Глава {_ROMAN_MAP.get(chap_str, chap_str)}"
                seg_lines.append(line)
            else:
                seg_lines.append(line)

        flush(current_part, current_chapter)
    return result


def format_docs(docs: list[Document]) -> str:
    parts: list[str] = []
    for doc in docs:
        m = doc.metadata
        ref = " | ".join(
            filter(
                None,
                [
                    m.get("book_title"),
                    m.get("part"),
                    m.get("chapter"),
                ],
            )
        )
        parts.append(f"[{ref}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def ensure_context(input_dict: dict) -> dict:
    """
    Если retriever не нашёл ничего полезного и контекст пустой,
    явно помечаем это в контексте, чтобы модель не фантазировала.
    """
    context = input_dict.get("context", "").strip()
    if not context:
        input_dict["context"] = (
            "Контекст пуст: ретривер не нашёл ни одного подходящего фрагмента. "
            "Если ответ важен, лучше явно сказать пользователю об этом."
        )
    return input_dict


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks).strip()
    return str(content)


# ─────────────────────────── промпты ─────────────────────────────────
ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты — эксперт-аналитик и универсальный исследователь текстов. Ты профессионально работаешь с литературой любого жанра: художественной, научной, технической, документальной, публицистической и др. Отвечай СТРОГО НА РУССКОМ ЯЗЫКЕ.\n"
            "Каждый фрагмент контекста помечен меткой, указывающей на структуру источника (например: [Название источника | Раздел/Часть | Глава/Параграф]). "
            "Используй эти метки как источник при цитировании.\n\n"
            "Структура ответа — два блока:\n\n"
            "Цитаты из текста\n"
            "Приведи несколько самых релеватных прямых цитат, взятых ТОЛЬКО из предоставленного контекста. "
            "Если подходящих цитат в контексте нет — так и напиши.\n"
            "После каждой цитаты укажи источник в скобках, опираясь на метку: "
            "(«Название источника», Раздел/Часть, Глава/Параграф).\n\n"
            "Анализ\n"
            "Краткий, емкий ответ на вопрос пользователя на основе приведённых цитат. "
            "Учитывай специфику текста: для научной литературы оперируй фактами и терминами, для художественной — сюжетом и образами. "
            "Не выходи за рамки предоставленного контекста и не придумывай факты. "
            "Если информации для полноценного ответа недостаточно — честно сообщи об этом.",
        ),
        MessagesPlaceholder("history"),
        ("human", "Контекст:\n{context}\n\nВопрос: {question}"),
    ]
)

QUOTE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты — внимательный исследователь и архивариус текстов. Ты работаешь с источниками любого типа (художественная литература, научные статьи, документация, нон-фикшн). Отвечай СТРОГО НА РУССКОМ ЯЗЫКЕ.\n"
            "Каждый фрагмент контекста помечен структурной меткой (например: [Название источника | Раздел/Часть | Глава/Параграф]).\n\n"
            "Твоя задача — найти и привести ТОЧНЫЕ цитаты из предоставленного контекста, "
            "максимально полно и релевантно отвечающие на запрос пользователя.\n\n"
            "Формат ответа:\n"
            "Для каждой найденной цитаты:\n"
            "1. Приведи фрагмент текста дословно, заключив его в кавычки.\n"
            "2. Укажи точный источник, используя данные из метки: («Название источника», Раздел/Часть, Глава/Параграф).\n"
            "3. Одно-три предложения — краткое обоснование, почему этот фрагмент отвечает на запрос пользователя (с учетом жанра текста).\n\n"
            "Не добавляй отсебятины, общих рассуждений или внешних знаний. Только точные цитаты с атрибуцией.\n"
            "Если в контексте нет подходящих фрагментов для ответа — так и напиши."
        ),
        MessagesPlaceholder("history"),
        ("human", "Контекст:\n{context}\n\nЧто найти: {question}"),
    ]
)

GRADE_PROMPT = (
    "Ты проверяешь релевантность найденного контекста вопросу пользователя.\\n"
    "Вопрос:\\n{question}\\n\\n"
    "Контекст:\\n{context}\\n\\n"
    "Верни binary_score=yes, если контекст действительно помогает ответить на вопрос, иначе no.\\n"
    "Также определи answer_style: quote если нужен ответ в формате цитат, иначе analysis."
)

REWRITE_PROMPT = (
    "Ты переписываешь запрос для поиска по книгам.\\n"
    "Исходный вопрос:\\n{question}\\n\\n"
    "Текущий поисковый запрос:\\n{query}\\n\\n"
    "Верни только improved retrieval-запрос в поле rewritten_query. "
    "Если улучшить нельзя, верни исходный query без изменений."
)


# ═══════════════════════ BACKEND CLASS ═══════════════════════════════

class AgentState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    question: str
    history: list[Any]
    mode: Literal["auto", "analysis", "quote"]
    force_answer_style: Literal["", "analysis", "quote"]
    route_decision: str
    answer_style: Literal["analysis", "quote"]
    retrieval_query: str
    rewrite_count: int
    max_rewrites: int
    final_answer: str


class GradeDecision(BaseModel):
    """Структурированный результат проверки релевантности контекста и стиля ответа."""

    model_config = ConfigDict(
        title="grade_decision",
        json_schema_extra={
            "description": (
                "Решение о релевантности контекста вопросу и рекомендуемом стиле ответа."
            )
        },
    )

    binary_score: Literal["yes", "no"] = Field(
        description="yes если контекст релевантен вопросу, иначе no"
    )
    answer_style: Literal["analysis", "quote"] = Field(
        description="Предпочтительный стиль ответа по запросу: analysis или quote"
    )


class RewriteDecision(BaseModel):
    """Структурированный результат переписывания поискового запроса."""

    model_config = ConfigDict(
        title="rewrite_decision",
        json_schema_extra={
            "description": (
                "Результат улучшения поискового запроса для ретривера."
            )
        },
    )

    rewritten_query: str = Field(default="", description="Новый поисковый запрос")

class PrefixedEmbeddings(Embeddings):
    def __init__(self, base, query_prefix="", doc_prefix=""):
        self.base = base
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.base.embed_documents([self.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.base.embed_query(self.query_prefix + text)


class RateLimitedEmbeddings(Embeddings):
    def __init__(
        self,
        base: Embeddings,
        requests_per_second: float,
        embed_documents_batch_size: int = 1,
    ):
        self.base = base
        self.requests_per_second = max(float(requests_per_second), 0.1)
        self.min_interval = 1.0 / self.requests_per_second
        self.embed_documents_batch_size = max(int(embed_documents_batch_size), 1)
        self._lock = threading.Lock()
        self._next_allowed_ts = 0.0

    def _throttle(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait_for = self._next_allowed_ts - now
            if wait_for > 0:
                time.sleep(wait_for)
                now = time.monotonic()
            self._next_allowed_ts = now + self.min_interval

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        batch_size = self.embed_documents_batch_size
        for start in range(0, len(texts), batch_size):
            self._throttle()
            batch = texts[start : start + batch_size]
            batch_vectors = self.base.embed_documents(batch)
            vectors.extend(batch_vectors)

        return vectors

    def embed_query(self, text: str) -> list[float]:
        self._throttle()
        return self.base.embed_query(text)


class OpenRouterEmbeddings(Embeddings):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key.strip()
        self.endpoint = "https://openrouter.ai/api/v1/embeddings"
        self.batch_size = 32

    def _parse_response_json(self, raw_bytes: bytes) -> dict[str, Any]:
        text = raw_bytes.decode("utf-8", errors="replace").strip()
        if not text:
            raise RuntimeError("OpenRouter embeddings вернул пустой ответ.")

        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and start < end:
            maybe_json = text[start : end + 1]
            try:
                payload = json.loads(maybe_json)
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                pass

        snippet = text[:500].replace("\n", " ")
        raise RuntimeError(
            "OpenRouter embeddings вернул ответ в неожиданном формате (не JSON). "
            f"Фрагмент ответа: {snippet}"
        )

    def _request_embeddings(self, inputs: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise ValueError("Не задан OPENROUTER_API_KEY для embedding модели.")

        payload = {
            "model": self.model_name,
            "input": inputs,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = self._parse_response_json(response.read())
        except urllib.error.HTTPError as err:
            details = err.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"Ошибка OpenRouter embeddings: HTTP {err.code}. {details}"
            ) from err
        except urllib.error.URLError as err:
            raise RuntimeError(f"Сеть недоступна для OpenRouter embeddings: {err}") from err

        data = result.get("data", [])
        vectors: list[list[float]] = []
        for item in data:
            if isinstance(item, dict):
                emb = item.get("embedding")
                if isinstance(emb, list):
                    vectors.append([float(v) for v in emb])

        if len(vectors) != len(inputs):
            raise RuntimeError("OpenRouter embeddings вернул некорректное число векторов.")
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._request_embeddings(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vectors = self._request_embeddings([text])
        return vectors[0] if vectors else []

class SimpleReranker:
    def __init__(self, model_name: str):
        """
        Args:
            model_name: Название модели cross-encoder
        """
        self.model = CrossEncoder(model_name)
    
    def rerank(
        self, query: str, documents: list[Document], top_n: int = RERANK_TOP_K
    ) -> list[Document]:
        """      
        Args:
            query: Поисковый запрос
            documents: Список документов для переоценки
            top_n: Сколько документов возвращает
        Returns:
            Список топ-N документов после rerank
        """

        if not documents:
            return []
        
        # Подготавливаем пары (query, document)
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Получаем скоры релевантности
        scores = self.model.predict(pairs)
        
        # Сортируем документы по скору
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Отбираем топ-N документов (score записываем в метаданные)
        result = []
        for doc, score in doc_score_pairs[:top_n]:
            doc.metadata = doc.metadata or {}
            doc.metadata["rerank_score"] = float(score)
            result.append(doc)
        
        return result

class HybridRerankerRetriever(BaseRetriever):
    first_retriever: BaseRetriever
    second_retriever: BaseRetriever
    reranker: SimpleReranker
    k: int = RERANK_TOP_K

    def _dedup_docs(self, docs: list[Document]) -> list[Document]:
        '''объединение результатов с дедупликацией'''
        seen: set[str] = set()
        result: list[Document] = []
        for d in docs:
            content = d.page_content
            uid = hashlib.md5(content.encode("utf-8")).hexdigest()
            if uid not in seen:
                seen.add(uid)
                result.append(d)
        return result
    
    def _get_relevant_documents(self, query: str, **_: Any) -> list[Document]:
        first_docs = self.first_retriever.invoke(query)
        second_docs = self.second_retriever.invoke(query)
        merged = self._dedup_docs(first_docs + second_docs)

        return self.reranker.rerank(query, merged, self.k)

class PageOracleBackend:
    def __init__(
        self, books_dir=".", persist_dir="./chroma_intro_db", log_callback=None
    ):
        self.books_dir = books_dir
        self.persist_dir = persist_dir
        self.log = log_callback or print
        self.vectorstore = None
        self.mmr_retriever = None
        self.quote_retriever = None
        self.rag_chain = None
        self.quote_chain = None
        self.model = None
        self.loaded_books: list[str] = []
        self.embeddings = None
        self.splits: list = []
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        self.temperature = 0.2  
        self.max_tokens = 4096
        self.top_p = 0.9
        self.score_threshold = 0.6
        self.history: list[dict[str, str]] = []
        self.history_max_messages = 20
        self.history_path = Path("chat_history.json")
        self.graph_app = None
        self.last_route_decision = "unknown"
        self.last_manual_override = False
        self.last_retrieval_query = ""
        self.reranker: SimpleReranker | None = None
        self.model_supports_tools = True
        self.model_supports_structured_output = True
        self.embedding_model_name = DEFAULT_EMBEDDING_MODEL
        self.provider_name = ""

    def _build_fallback_retrieve_response(self, query: str) -> AIMessage:
        # Унифицируем fallback-вызов retrieve, когда bind_tools недоступен.
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "retrieve",
                    "args": {"query": query},
                    "id": "fallback_retrieve_call",
                    "type": "tool_call",
                }
            ],
        )

    def _should_use_retrieve_without_tools(
        self,
        question: str,
        mode: Literal["auto", "analysis", "quote"],
    ) -> bool:
        if mode in {"analysis", "quote"}:
            return True

        if self._looks_like_quote_request(question):
            return True

        model = self.model
        if not model:
            return True

        # В режиме без tools просим модель явно выбрать маршрут.
        routing_messages = [
            SystemMessage(
                content=(
                    "Ты роутер запросов. Верни только одно слово: RETRIEVE или DIRECT. "
                    "RETRIEVE — если вопрос требует поиска в книгах/документах. "
                    "DIRECT — если можно ответить без поиска по книгам (чат, настройки, приложение)."
                )
            ),
            HumanMessage(content=question),
        ]
        try:
            reply = model.invoke(routing_messages)
            decision = _extract_text(getattr(reply, "content", "")).strip().upper()
        except Exception as err:
            self.log(f"[Router] Не удалось получить решение DIRECT/RETRIEVE: {err}")
            return True

        if "DIRECT" in decision and "RETRIEVE" not in decision:
            return False
        if "RETRIEVE" in decision:
            return True
        return True

    def _is_dimension_mismatch_error(self, err: Exception) -> bool:
        msg = str(err).lower()
        return (
            "expecting embedding with dimension" in msg
            or ("embedding" in msg and "dimension" in msg and " got " in msg)
        )

    def _validate_vectorstore_dimension(self) -> None:
        if not self.vectorstore:
            return
        try:
            if self.vectorstore._collection.count() > 0:
                # Пробный запрос гарантирует, что размерность query-эмбеддинга
                # совместима с размерностью уже сохранённой коллекции.
                self.vectorstore.similarity_search("dimension probe", k=1)
        except Exception as err:
            if self._is_dimension_mismatch_error(err):
                raise RuntimeError(str(err)) from err
            raise

    def _embedding_meta_path(self) -> Path:
        return Path(self.persist_dir) / "embedding_config.json"

    def _save_embedding_meta(self) -> None:
        meta_path = self._embedding_meta_path()
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"embedding_model": self.embedding_model_name}
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_embedding_meta(self) -> str:
        meta_path = self._embedding_meta_path()
        if not meta_path.exists():
            return ""
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            return str(payload.get("embedding_model", "")).strip()
        except Exception:
            return ""

    def _clear_vector_index(self) -> None:
        self.vectorstore = None
        self.mmr_retriever = None
        self.quote_retriever = None
        self.rag_chain = None
        self.quote_chain = None
        gc.collect()
        shutil.rmtree(self.persist_dir, ignore_errors=True)

    def _create_embeddings(self, embedding_model: str, api_key: str) -> Embeddings:
        model_name = embedding_model.strip() or DEFAULT_EMBEDDING_MODEL
        cfg = EMBEDDING_MODELS.get(model_name)
        if not cfg:
            raise ValueError(f"Неизвестная embedding модель: {model_name}")

        provider = str(cfg.get("provider", ""))
        query_prefix = str(cfg.get("query_prefix", ""))
        doc_prefix = str(cfg.get("doc_prefix", ""))
        resolved_api_key = api_key.strip() or os.getenv("OPENROUTER_API_KEY", "").strip()

        if provider == "openrouter":
            if resolved_api_key:
                os.environ["OPENROUTER_API_KEY"] = resolved_api_key
            base_emb: Embeddings = OpenRouterEmbeddings(
                model_name=model_name,
                api_key=resolved_api_key,
            )
        elif provider == "huggingface":
            base_emb = HuggingFaceEmbeddings(model_name=model_name)
        elif provider == "yandex":
            if resolved_api_key:
                os.environ["YC_API_KEY"] = resolved_api_key

            folder_id = os.getenv("YC_FOLDER_ID", "").strip()
            module = importlib.import_module("langchain_community.embeddings.yandex")
            yandex_embeddings_class = getattr(module, "YandexGPTEmbeddings")

            yandex_kwargs: dict[str, Any] = {
                "model_uri": self._build_yandex_embedding_uri(model_name, folder_id),
            }
            if folder_id:
                yandex_kwargs["folder_id"] = folder_id
            if resolved_api_key:
                yandex_kwargs["api_key"] = resolved_api_key

            yandex_base_emb = yandex_embeddings_class(**yandex_kwargs)
            # У Yandex Cloud жёсткий лимит 10 req/s на embeddings,
            # поэтому ограничиваем частоту чуть ниже лимита.
            base_emb = RateLimitedEmbeddings(
                base=yandex_base_emb,
                requests_per_second=8.0,
                embed_documents_batch_size=1,
            )
        else:
            raise ValueError(f"Неизвестный embedding provider: {provider}")

        self.embedding_model_name = model_name
        return PrefixedEmbeddings(
            base_emb,
            query_prefix=query_prefix,
            doc_prefix=doc_prefix,
        )

    def _build_yandex_model_uri(self, model_name: str, folder_id: str) -> str:
        raw_name = model_name.strip()
        if not raw_name:
            raw_name = "yandexgpt-lite/latest"
        if "://" in raw_name:
            return raw_name
        if folder_id:
            return f"gpt://{folder_id}/{raw_name}"
        return raw_name

    def _build_yandex_embedding_uri(self, model_name: str, folder_id: str) -> str:
        raw_name = model_name.strip()
        if not raw_name:
            raw_name = "text-search-doc-1.0/latest"
        if "://" in raw_name:
            return raw_name
        if folder_id:
            return f"emb://{folder_id}/{raw_name}"
        return raw_name

    # ── инициализация ────────────────────────────────────────────────
    def initialize(
        self,
        provider="GigaChat",
        model_name="gigachat-2",
        llm_api_key: str = "",
        embedding_api_key: str = "",
        api_key: str = "",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        top_p: float = 0.9,
        score_threshold: float = 0.6,
    ):
        resolved_llm_api_key = llm_api_key.strip() or api_key.strip()
        resolved_embedding_api_key = embedding_api_key.strip() or api_key.strip()
        self.score_threshold = score_threshold
        self.log("[Инициализация] Загружаем эмбеддинги…")
        self.embeddings = self._create_embeddings(embedding_model, resolved_embedding_api_key)

        self.log("[Инициализация] Загружаем книги…")
        self._load_all_books()

        self.log("[Инициализация] Создаём векторное хранилище…")
        self._init_vectorstore()

        self.log("[Инициализация] Подключаем ИИ модель…")
        self.set_model(
            provider,
            model_name,
            resolved_llm_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        self.log("[Инициализация] Система готова к работе!")

    def _load_all_books(self):
        txt_files = sorted(Path(self.books_dir).glob("*.txt"))
        if not txt_files:
            self.log(f"[!] Нет .txt файлов в «{self.books_dir}»")
            return
        all_annotated: list[Document] = []
        self.loaded_books = []
        for tf in txt_files:
            all_annotated.extend(self._load_and_annotate(str(tf)))
            self.loaded_books.append(tf.name)
        self.splits = self.text_splitter.split_documents(all_annotated)
        self.log(f"Всего чанков: {len(self.splits)}")

    def _normalize_source_path(self, path: Path | str) -> str:
        try:
            return str(Path(path).resolve())
        except Exception:
            return str(path)

    def _get_indexed_documents(self) -> list[Document]:
        if not self.vectorstore:
            return []
        try:
            payload = self.vectorstore.get(include=["documents", "metadatas"])
            texts = payload.get("documents", [])
            metadatas = payload.get("metadatas", [])
            docs: list[Document] = []
            for i, text in enumerate(texts):
                if not isinstance(text, str):
                    continue
                metadata = metadatas[i] if i < len(metadatas) and isinstance(metadatas[i], dict) else {}
                docs.append(Document(page_content=text, metadata=metadata))
            return docs
        except Exception as err:
            self.log(f"[Индекс] Не удалось прочитать документы из кэша: {err}")
            return []

    def _refresh_loaded_books_from_splits(self) -> None:
        names: list[str] = []
        seen: set[str] = set()
        for doc in self.splits:
            source = str((doc.metadata or {}).get("source", "")).strip()
            if not source:
                continue
            name = Path(source).name
            if name and name not in seen:
                seen.add(name)
                names.append(name)
        if names:
            self.loaded_books = names

    def _ensure_book_in_library(self, filepath: str) -> Path:
        src = Path(filepath).resolve()
        books_root = Path(self.books_dir).resolve()
        books_root.mkdir(parents=True, exist_ok=True)

        if src.parent == books_root:
            return src

        target = books_root / src.name
        if target.exists():
            try:
                if src.samefile(target):
                    return target
            except Exception:
                pass

            stem = target.stem
            suffix = target.suffix
            index = 1
            while True:
                candidate = books_root / f"{stem}_{index}{suffix}"
                if not candidate.exists():
                    target = candidate
                    break
                index += 1

        shutil.copy2(src, target)
        self.log(f"[Файлы] Книга скопирована в библиотеку: {target.name}")
        return target

    def _load_and_annotate(self, filepath: str) -> list:
        path = Path(filepath)
        book_title = path.stem.replace("_", " ").replace("-", " ")
        raw_docs = TextLoader(str(path), autodetect_encoding=True).load()
        annotated = annotate_book(raw_docs, book_title)
        normalized_source = self._normalize_source_path(path)
        for doc in annotated:
            doc.metadata = doc.metadata or {}
            doc.metadata["source"] = normalized_source
        self.log(f"  «{path.name}» → {len(annotated)} сегментов (part/chapter)")
        return annotated

    # ── векторное хранилище ──────────────────────────────────────────
    def _init_vectorstore(self):
        persist = Path(self.persist_dir)
        if not self.splits and not persist.exists():
            return

        stored_embedding_model = self._load_embedding_meta()
        if persist.exists() and stored_embedding_model and stored_embedding_model != self.embedding_model_name:
            self.log(
                "[Индекс] Найден индекс для другой embedding модели. Пересоздаём индекс."
            )
            self._clear_vector_index()

        if persist.exists():
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir,
            )
            stored = self.vectorstore._collection.count()
            if stored > 0:
                try:
                    self._validate_vectorstore_dimension()
                    indexed_docs = self._get_indexed_documents()
                    if indexed_docs:
                        self.splits = indexed_docs
                        self._refresh_loaded_books_from_splits()
                        self.log(
                            f"[Индекс] Загружен из кэша ({stored} чанков). Используем сохранённый индекс."
                        )
                    else:
                        self.log("[Индекс] Кэш найден, но документы не прочитаны. Пересоздаём индекс.")
                        self._clear_vector_index()
                        self.vectorstore = self._build_vectorstore()
                except Exception as err:
                    if self._is_dimension_mismatch_error(err):
                        self.log(
                            "[Индекс] Размерность embedding не совпадает с кэшем. "
                            "Пересоздаём индекс для новой модели."
                        )
                        self._clear_vector_index()
                        self.vectorstore = self._build_vectorstore()
                    else:
                        raise
            else:
                self.log("[Индекс] Кэш пуст. Пересоздаём индекс.")
                self._clear_vector_index()
                self.vectorstore = self._build_vectorstore()
        else:
            self.vectorstore = self._build_vectorstore()

        self._create_retrievers()

    def _create_retrievers(self) -> None:
        if not self.vectorstore or not self.splits:
            return
        if self.reranker is None:
            self.reranker = SimpleReranker("BAAI/bge-reranker-base")

        self.mmr_retriever = HybridRerankerRetriever(
            first_retriever=self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": RETRIEVER_K,
                    "fetch_k": RETRIEVER_FETCH_K,
                    "lambda_mult": 0.7,
                },
            ),
            second_retriever=BM25Retriever.from_documents(
                self.splits,
                k=RETRIEVER_K,
                score_threshold=self.score_threshold,
            ),
            reranker=self.reranker,
            k=RERANK_TOP_K,
        )
        self.quote_retriever = HybridRerankerRetriever(
            first_retriever=self.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": RETRIEVER_K, "score_threshold": self.score_threshold},
            ),
            second_retriever=BM25Retriever.from_documents(
                self.splits,
                k=RETRIEVER_K,
                score_threshold=self.score_threshold,
            ),
            reranker=self.reranker,
            k=RERANK_TOP_K,
        )

    def set_score_threshold(self, score_threshold: float) -> bool:
        if score_threshold < 0 or score_threshold > 1:
            self.log("[Ошибка] score_threshold должен быть в диапазоне 0..1.")
            return False
        self.score_threshold = score_threshold
        self._create_retrievers()
        self._create_chains()
        self._create_graph()
        self.log(f"[Ретривер] score_threshold обновлён: {score_threshold}")
        return True

    def _build_vectorstore(self) -> Chroma:
        self.log("[Индекс] Создаём новый индекс…")
        try:
            vectorstore = Chroma.from_documents(
                documents=self.splits,
                embedding=self.embeddings,
                persist_directory=self.persist_dir,
            )
            self._save_embedding_meta()
            return vectorstore
        except Exception:
            # Если создание индекса оборвалось, удаляем частично созданный кэш.
            shutil.rmtree(self.persist_dir, ignore_errors=True)
            raise

    def set_embeddings(self, embedding_model: str, api_key: str) -> bool:
        try:
            self.log(f"[Эмбеддинги] Переключаем embedding модель: {embedding_model}")
            self.embeddings = self._create_embeddings(embedding_model, api_key)
            
            try:
                self._init_vectorstore()
            except Exception as init_err:
                if self._is_dimension_mismatch_error(init_err):
                    self.log(
                        "[Эмбеддинги] Обнаружена несовместимость размерности эмбеддингов. "
                        "Удаляю старую векторную базу и пересоздаю с новыми параметрами…"
                    )
                    self._clear_vector_index()
                    self._init_vectorstore()
                else:
                    raise
            
            self._create_retrievers()
            self._create_chains()
            self._create_graph()
            self.log(f"[Эмбеддинги] Активна модель: {self.embedding_model_name}")
            return True
        except Exception as err:
            self.log(f"[Ошибка эмбеддингов] {err}")
            return False

    # ── модель и цепочки ─────────────────────────────────────────────
    def set_model(
        self,
        provider_name: str,
        model_name: str,
        api_key: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        top_p: float = 0.9,
    ) -> bool:
        cfg = PROVIDERS.get(provider_name)
        if not cfg:
            self.log(f"[Ошибка] Неизвестный провайдер: {provider_name}")
            return False
        env_key = str(cfg.get("env_key", "")).strip()
        if env_key and api_key.strip():
            os.environ[env_key] = api_key
        try:
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.top_p = top_p
            self.provider_name = provider_name
            self.model_supports_tools = True
            self.model_supports_structured_output = True

            module = importlib.import_module(cfg["package"])
            model_class = getattr(module, cfg["class"])

            provider_kwargs: dict[str, Any] = {}
            if provider_name == "GigaChat":
                provider_kwargs["verify_ssl_certs"] = bool(
                    cfg.get("verify_ssl_certs", False)
                )

            if provider_name == "YandexGPT":
                folder_id = os.getenv("YC_FOLDER_ID", "").strip()
                common_kwargs = {
                    "model_uri": self._build_yandex_model_uri(model_name, folder_id),
                    "temperature": temperature,
                    **provider_kwargs,
                }
                if folder_id:
                    common_kwargs["folder_id"] = folder_id
                if api_key.strip():
                    common_kwargs["api_key"] = api_key.strip()
            else:
                common_kwargs = {
                    "model": model_name,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    **provider_kwargs,
                }

            # По документации LangChain для OpenRouter используем выделенный
            # провайдер ChatOpenRouter (langchain_openrouter)
            if provider_name == "OpenRouter" and api_key.strip():
                common_kwargs["api_key"] = api_key.strip()

            # Некоторые провайдеры используют max_output_tokens вместо max_tokens.
            if provider_name == "Google":
                common_kwargs["max_output_tokens"] = max_tokens


            try:
                self.model = model_class(**common_kwargs)
            except TypeError:
                if provider_name == "YandexGPT":
                    folder_id = os.getenv("YC_FOLDER_ID", "").strip()
                    minimal_kwargs: dict[str, Any] = {
                        "model_uri": self._build_yandex_model_uri(model_name, folder_id),
                        **provider_kwargs,
                    }
                    if folder_id:
                        minimal_kwargs["folder_id"] = folder_id
                    if api_key.strip():
                        minimal_kwargs["api_key"] = api_key.strip()
                    self.model = model_class(**minimal_kwargs)
                else:
                    # Fallback для моделей с неполной поддержкой kwargs.
                    try:
                        self.model = model_class(
                            model=model_name,
                            temperature=temperature,
                            top_p=top_p,
                            **provider_kwargs,
                        )
                    except TypeError:
                        self.model = model_class(
                            model=model_name,
                            temperature=temperature,
                            **provider_kwargs,
                        )

            if provider_name == "YandexGPT":
                # ChatYandexGPT в langchain_community не реализует bind_tools / with_structured_output.
                # Явно включаем совместимый fallback-режим, чтобы не пытаться вызывать
                # неподдерживаемые API и не зашумлять логи ошибками.
                self.model_supports_tools = False
                self.model_supports_structured_output = False

            self._create_chains()
            self._create_graph()
            self.log(
                f"[Модель] {provider_name} / {model_name} — подключена "
                f"(temperature={temperature}, max_tokens={max_tokens}, top_p={top_p})."
            )
            return True
        except Exception as e:
            self.log(f"[Ошибка модели] {e}")
            return False

    def _create_chains(self):
        if not self.model or not self.vectorstore:
            return
        mmr = self.mmr_retriever
        quote = self.quote_retriever
        if not mmr or not quote:
            return
        self.rag_chain = (
            {
                "context": RunnableLambda(
                    lambda data: mmr.invoke(cast(dict, data)["question"])
                )
                | RunnableLambda(format_docs),
                "question": RunnableLambda(lambda data: cast(dict, data)["question"]),
                "history": RunnableLambda(
                    lambda data: cast(dict, data).get("history", [])
                ),
            }
            | RunnableLambda(ensure_context)
            | ANALYSIS_PROMPT
            | self.model
            | StrOutputParser()
        )
        self.quote_chain = (
            {
                "context": RunnableLambda(
                    lambda data: quote.invoke(cast(dict, data)["question"])
                )
                | RunnableLambda(format_docs),
                "question": RunnableLambda(lambda data: cast(dict, data)["question"]),
                "history": RunnableLambda(
                    lambda data: cast(dict, data).get("history", [])
                ),
            }
            | RunnableLambda(ensure_context)
            | QUOTE_PROMPT
            | self.model
            | StrOutputParser()
        )

    def _create_graph(self) -> None:
        if not self.model or not self.vectorstore:
            self.graph_app = None
            return

        workflow = StateGraph(AgentState)
        workflow.add_node("generate_query_or_respond", self._node_generate_query_or_respond)
        workflow.add_node("retrieve", ToolNode([self._build_retrieve_tool()]))
        workflow.add_node("rewrite_question", self._node_rewrite_question)
        workflow.add_node("generate_analysis_answer", self._node_generate_analysis_answer)
        workflow.add_node("generate_quote_answer", self._node_generate_quote_answer)

        workflow.add_edge(START, "generate_query_or_respond")
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            },
        )
        workflow.add_conditional_edges(
            "retrieve",
            self._route_after_retrieve,
            {
                "rewrite_question": "rewrite_question",
                "generate_analysis_answer": "generate_analysis_answer",
                "generate_quote_answer": "generate_quote_answer",
            },
        )
        workflow.add_edge("rewrite_question", "generate_query_or_respond")
        workflow.add_edge("generate_analysis_answer", END)
        workflow.add_edge("generate_quote_answer", END)
        self.graph_app = workflow.compile()

    # ── memory / history ────────────────────────────────────────────
    def _trim_history(self) -> None:
        if len(self.history) > self.history_max_messages:
            self.history = self.history[-self.history_max_messages :]

    def append_user_message(self, content: str) -> None:
        self.history.append(
            {
                "role": "user",
                "content": content,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        self._trim_history()

    def append_assistant_message(self, content: str) -> None:
        self.history.append(
            {
                "role": "assistant",
                "content": content,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        self._trim_history()

    def clear_history(self) -> None:
        self.history.clear()

    def history_size(self) -> int:
        return len(self.history)

    def get_recent_history_for_prompt(self) -> list:
        messages = []
        for item in self.history[-self.history_max_messages :]:
            role = item.get("role")
            content = item.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))
        return messages

    def load_history(self, filepath: str | None = None) -> bool:
        path = Path(filepath) if filepath else self.history_path
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                parsed = []
                for item in payload:
                    if not isinstance(item, dict):
                        continue
                    role = item.get("role")
                    content = item.get("content")
                    if role in {"user", "assistant", "system"} and isinstance(
                        content, str
                    ):
                        parsed.append(
                            {
                                "role": role,
                                "content": content,
                                "timestamp": item.get(
                                    "timestamp", datetime.utcnow().isoformat()
                                ),
                            }
                        )
                self.history = parsed
                self._trim_history()
                return True
            return False
        except Exception as err:
            self.log(f"[История] Не удалось загрузить историю: {err}")
            return False

    def save_history(self, filepath: str | None = None) -> bool:
        path = Path(filepath) if filepath else self.history_path
        try:
            path.write_text(
                json.dumps(self.history, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return True
        except Exception as err:
            self.log(f"[История] Не удалось сохранить историю: {err}")
            return False

    # ── router helpers ───────────────────────────────────────────────
    def _looks_like_quote_request(self, question: str) -> bool:
        q = question.lower()
        quote_signals = [
            "цитат",
            "дослов",
            "фрагмент",
            "приведи строк",
            "покажи отрывок",
        ]
        return any(signal in q for signal in quote_signals)

    def _build_retrieve_tool(self):
        @tool
        def retrieve(query: str) -> str:
            """Ищет релевантные фрагменты в загруженных книгах."""
            retriever = (
                self.quote_retriever
                if self._looks_like_quote_request(query)
                else self.mmr_retriever
            )
            if not retriever:
                return "Контекст пуст: ретривер не инициализирован."
            docs = retriever.invoke(query)
            if not docs:
                return "Контекст пуст: ретривер не нашёл подходящих фрагментов."
            return format_docs(cast(list[Document], docs))

        return retrieve

    def _node_generate_query_or_respond(self, state: AgentState) -> AgentState:
        model = self.model
        if not model:
            return {
                "final_answer": "[Ошибка] Модель не инициализирована.",
                "route_decision": "error",
            }

        question = (state.get("question") or "").strip()
        history = state.get("history", [])
        messages = state.get("messages", [])
        mode = state.get("mode", "auto")

        if not messages:
            system_parts = [
                "Ты агент по книгам и текстам.",
                "Сначала реши: ответить напрямую или вызвать инструмент retrieve для поиска контекста.",
                "Если вопрос о содержании книг, почти всегда вызывай retrieve.",
                "Если это вопрос о приложении/настройках/ошибках, отвечай напрямую без retrieve.",
            ]
            if mode == "analysis":
                system_parts.append("Режим зафиксирован: analysis.")
            elif mode == "quote":
                system_parts.append("Режим зафиксирован: quote.")

            messages = [
                SystemMessage(content=" ".join(system_parts)),
                *history[-8:],
                HumanMessage(content=question),
            ]

        retrieve_tool = self._build_retrieve_tool()
        retrieval_query = (state.get("retrieval_query", question) or question).strip()

        if self.model_supports_tools:
            try:
                response = model.bind_tools([retrieve_tool]).invoke(messages)
                content_str = str(getattr(response, "content", ""))
                # Проверка: некоторые модели OpenRouter возвращают ошибку tool_choice в виде обычного текста (или JSON строки) вместо исключения
                if "No endpoints found that support the provided 'tool_choice' value" in content_str or "tool_choice" in content_str:
                    raise Exception("OpenRouter soft error: " + content_str)
                # Если ответ пустой и нет вызовов тулов - модель не справилась с bind_tools
                if not content_str.strip() and not getattr(response, "tool_calls", []):
                    raise Exception("Model returned empty content without tool calls (possibly does not support tools)")
            except Exception as err:
                self.log(f"[Tools] bind_tools недоступен, fallback: {err}")
                self.model_supports_tools = False
                if self._should_use_retrieve_without_tools(question, cast(Literal["auto", "analysis", "quote"], mode)):
                    response = self._build_fallback_retrieve_response(retrieval_query)
                else:
                    response = model.invoke(messages)
        else:
            if self._should_use_retrieve_without_tools(question, cast(Literal["auto", "analysis", "quote"], mode)):
                response = self._build_fallback_retrieve_response(retrieval_query)
            else:
                response = model.invoke(messages)
        forced = cast(Literal["", "analysis", "quote"], state.get("force_answer_style", ""))
        if forced in {"analysis", "quote"}:
            style = forced
        else:
            style = "quote" if self._looks_like_quote_request(question) else "analysis"

        return {
            "messages": [response],
            "answer_style": cast(Literal["analysis", "quote"], style),
            "route_decision": "tool-or-direct",
            "retrieval_query": retrieval_query,
        }

    def _route_after_retrieve(
        self, state: AgentState
    ) -> Literal[
        "rewrite_question",
        "generate_analysis_answer",
        "generate_quote_answer",
    ]:
        model = self.model
        if not model:
            return "rewrite_question"

        messages = state.get("messages", [])
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        if not tool_messages:
            return "rewrite_question"

        question = (state.get("question") or "").strip()
        context = _extract_text(tool_messages[-1].content).strip()
        if not context or context.lower().startswith("контекст пуст"):
            rewrite_count = int(state.get("rewrite_count", 0))
            max_rewrites = int(state.get("max_rewrites", 2))
            if rewrite_count < max_rewrites:
                return "rewrite_question"

            forced = state.get("force_answer_style", "")
            if forced == "quote":
                return "generate_quote_answer"
            if forced == "analysis":
                return "generate_analysis_answer"
            return "generate_quote_answer" if self._looks_like_quote_request(question) else "generate_analysis_answer"

        decision = None
        if getattr(self, "model_supports_structured_output", True):
            try:
                decision = model.with_structured_output(GradeDecision).invoke(
                    [
                        HumanMessage(
                            content=GRADE_PROMPT.format(question=question, context=context)
                        )
                    ]
                )
            except Exception as e:
                self.log(f"[Grade] with_structured_output недоступен, fallback ручного парсинга: {e}")
                self.model_supports_structured_output = False

        if not decision:
            reply = model.invoke([HumanMessage(content=GRADE_PROMPT.format(question=question, context=context))])
            content = _extract_text(getattr(reply, "content", "")).lower()
            binary_score = "yes" if "yes" in content else "no"
            ans_style = "quote" if "quote" in content else "analysis"
            decision = GradeDecision(binary_score=binary_score, answer_style=ans_style)

        if decision.binary_score == "no":
            rewrite_count = int(state.get("rewrite_count", 0))
            max_rewrites = int(state.get("max_rewrites", 2))
            if rewrite_count < max_rewrites:
                return "rewrite_question"

            forced = state.get("force_answer_style", "")
            if forced == "quote":
                return "generate_quote_answer"
            if forced == "analysis":
                return "generate_analysis_answer"
            return "generate_quote_answer" if self._looks_like_quote_request(question) else "generate_analysis_answer"

        forced = state.get("force_answer_style", "")
        if forced == "quote":
            return "generate_quote_answer"
        if forced == "analysis":
            return "generate_analysis_answer"

        return (
            "generate_quote_answer"
            if decision.answer_style == "quote"
            else "generate_analysis_answer"
        )

    def _invoke_prompt_with_context(
        self,
        prompt: ChatPromptTemplate,
        question: str,
        history: list[Any],
        context_text: str,
    ) -> str:
        model = self.model
        if not model:
            raise RuntimeError("Модель не инициализирована")

        payload = ensure_context(
            {
                "context": context_text,
                "question": question,
                "history": history,
            }
        )
        messages = prompt.format_messages(**payload)
        reply = model.invoke(messages)
        return _extract_text(getattr(reply, "content", ""))

    def _node_generate_analysis_answer(self, state: AgentState) -> AgentState:
        question = state.get("question", "")
        history = state.get("history", [])
        tool_messages = [m for m in state.get("messages", []) if isinstance(m, ToolMessage)]
        context_text = _extract_text(tool_messages[-1].content) if tool_messages else ""
        answer = self._invoke_prompt_with_context(
            ANALYSIS_PROMPT, question, history, context_text
        )

        return {
            "messages": [AIMessage(content=answer)],
            "final_answer": answer,
            "route_decision": "retrieval-analysis",
            "retrieval_query": state.get("retrieval_query", question),
        }

    def _node_generate_quote_answer(self, state: AgentState) -> AgentState:
        question = state.get("question", "")
        history = state.get("history", [])
        tool_messages = [m for m in state.get("messages", []) if isinstance(m, ToolMessage)]
        context_text = _extract_text(tool_messages[-1].content) if tool_messages else ""
        if not self.model:
            return {
                "final_answer": "[Ошибка] Модель не инициализирована.",
                "route_decision": "error",
                "retrieval_query": state.get("retrieval_query", question),
            }
        answer = self._invoke_prompt_with_context(
            QUOTE_PROMPT, question, history, context_text
        )

        return {
            "messages": [AIMessage(content=answer)],
            "final_answer": answer,
            "route_decision": "retrieval-quote",
            "retrieval_query": state.get("retrieval_query", question),
        }

    def _node_rewrite_question(self, state: AgentState) -> AgentState:
        model = self.model
        question = (state.get("question") or "").strip()
        query = (state.get("retrieval_query") or question).strip()
        rewrite_count = int(state.get("rewrite_count", 0))

        if not model:
            return {
                "messages": [HumanMessage(content=query or question)],
                "retrieval_query": query or question,
                "rewrite_count": rewrite_count + 1,
                "route_decision": "rewrite",
            }

        decision = None
        if getattr(self, "model_supports_structured_output", True):
            try:
                decision = model.with_structured_output(RewriteDecision).invoke(
                    [
                        HumanMessage(
                            content=REWRITE_PROMPT.format(question=question, query=query)
                        )
                    ]
                )
            except Exception as e:
                self.log(f"[Rewrite] with_structured_output недоступен, fallback ручного парсинга: {e}")
                self.model_supports_structured_output = False

        if not decision:
            reply = model.invoke([HumanMessage(content=REWRITE_PROMPT.format(question=question, query=query))])
            content = _extract_text(getattr(reply, "content", "")).strip()
            rewritten_query = content
            try:
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1 and start < end:
                    data = json.loads(content[start:end+1])
                    if "rewritten_query" in data:
                        rewritten_query = str(data["rewritten_query"])
            except Exception:
                pass
            
            # Очистка от текстовых префиксов, если модель ответила строкой
            rewritten_query = re.sub(
                r"^(?:\*\*?)?(?:rewritten_query|новый запрос|запрос|query)(?:\*\*?)?:\s*",
                "",
                rewritten_query,
                flags=re.IGNORECASE,
            ).strip()
            # Убираем кавычки с краёв если модель их добавила
            if rewritten_query.startswith('"') and rewritten_query.endswith('"'):
                rewritten_query = rewritten_query[1:-1].strip()
            elif rewritten_query.startswith("'") and rewritten_query.endswith("'"):
                rewritten_query = rewritten_query[1:-1].strip()
                
            decision = RewriteDecision(rewritten_query=rewritten_query)

        rewritten_query = decision.rewritten_query.strip() or query or question
        return {
            "messages": [HumanMessage(content=rewritten_query)],
            "retrieval_query": rewritten_query,
            "rewrite_count": rewrite_count + 1,
            "route_decision": "rewrite",
        }

    def get_last_debug_info(self) -> dict:
        return {
            "route_decision": self.last_route_decision,
            "manual_override": self.last_manual_override,
            "retrieval_query": self.last_retrieval_query,
            "history_size": self.history_size(),
        }

    # ── публичные методы ─────────────────────────────────────────────
    def add_document(self, filepath: str):
        path = Path(filepath.strip())
        if not path.exists():
            self.log(f"[Ошибка] Файл не найден: {path}")
            return
        if not self.vectorstore:
            self.log("[Ошибка] Векторная база не инициализирована.")
            return

        path = self._ensure_book_in_library(str(path))
        source_value = self._normalize_source_path(path)

        existing = self.vectorstore.get(where={"source": source_value})
        if existing["ids"]:
            self.log(
                f"[Пропуск] «{path.name}» уже есть в базе ({len(existing['ids'])} чанков)."
            )
            return
        new_splits = self.text_splitter.split_documents(
            self._load_and_annotate(str(path))
        )
        for doc in new_splits:
            doc.metadata = doc.metadata or {}
            doc.metadata["source"] = source_value

        self.vectorstore.add_documents(new_splits)
        if hasattr(self.vectorstore, "persist"):
            self.vectorstore.persist()
        self.splits.extend(new_splits)
        if path.name not in self.loaded_books:
            self.loaded_books.append(path.name)
        self._create_retrievers()
        self._create_chains()
        self._create_graph()
        self.log(f"[Готово] Добавлено {len(new_splits)} чанков из «{path.name}».")

    def ask(self, question: str, mode: str = "auto") -> str:
        question = question.strip()
        if not question:
            return "[Ошибка] Вопрос пустой."

        if mode not in {"auto", "analysis", "quote"}:
            mode = "auto"

        history_messages = self.get_recent_history_for_prompt()
        self.last_manual_override = mode in {"analysis", "quote"}
        self.last_route_decision = "unknown"
        self.last_retrieval_query = question

        if not self.graph_app:
            return "[Ошибка] Система не инициализирована."

        try:
            state: AgentState = {
                "question": question,
                "history": history_messages,
                "messages": [],
                "mode": cast(Literal["auto", "analysis", "quote"], mode),
                "force_answer_style": (
                    "" if mode == "auto" else cast(Literal["analysis", "quote"], mode)
                ),
                "answer_style": "analysis",
                "retrieval_query": question,
                "rewrite_count": 0,
                "max_rewrites": 5,
            }
            result = self.graph_app.invoke(state)
            answer = result.get("final_answer", "") if isinstance(result, dict) else ""
            if not answer and isinstance(result, dict):
                msgs = result.get("messages", [])
                if msgs:
                    answer = _extract_text(getattr(msgs[-1], "content", ""))

            if not answer:
                answer = "Не удалось получить ответ. Попробуйте переформулировать вопрос."

            self.last_route_decision = (
                result.get("route_decision", "auto-unknown")
                if isinstance(result, dict)
                else "auto-unknown"
            )
            self.last_retrieval_query = (
                result.get("retrieval_query", question)
                if isinstance(result, dict)
                else question
            )
            self.append_user_message(question)
            self.append_assistant_message(answer)
            return answer
        except Exception as err:
            self.log(f"[Graph] Ошибка: {err}")
            return "Внутренняя ошибка графа. Попробуйте повторить запрос."

if __name__ == "__main__":
    PageOracleBackend().initialize()
