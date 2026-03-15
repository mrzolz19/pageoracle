# TODO: Добавить больше провайдеров и обновить их 
# TODO: Подправить системные промты (а именно роль), чтобы он лучше понимал контекст нехудожественной литературы
# TODO: Добавить учёт и очистку истории

import re
import shutil
import importlib
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
#from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
#from langsmith import traceable

#load_dotenv()  # Загружаем переменные окружения из .env

os.environ['NO_PROXY'] = '127.0.0.1,localhost'

# ───────────────────── провайдеры LLM (для GUI настроек) ─────────────
PROVIDERS = {
    "DeepSeek": {
        "class": "ChatDeepSeek",
        "package": "langchain_deepseek",
        "env_key": "DEEPSEEK_API_KEY",
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
    "OpenAI": {
        "class": "ChatOpenAI",
        "package": "langchain_openai",
        "env_key": "OPENAI_API_KEY",
        "models": ["gpt-5.4-pro-2026-03-05", "gpt-5.4-2026-03-05", "gpt-5-mini-2025-08-07", "gpt-5-nano-2025-08-07"],
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
        "models": ["gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-3.1-flash-lite-preview"],
    },
    "Mistral": {
        "class": "ChatMistralAI",
        "package": "langchain_mistralai",
        "env_key": "MISTRAL_API_KEY",
        "models": ["mistral-large-latest", "mistral-small-latest"],
    },
    "Groq": {
        "class": "ChatGroq",
        "package": "langchain_groq",
        "env_key": "GROQ_API_KEY",
        "models": ["llama3-70b-8192", "mixtral-8x7b-32768"],
    },
}

# ───────────────────────── парсинг структуры книги ───────────────────
_ORDINAL_MAP = {
    "ПЕРВАЯ": "1", "ВТОРАЯ": "2", "ТРЕТЬЯ": "3",
    "ЧЕТВЁРТАЯ": "4", "ЧЕТВЕРТАЯ": "4",
    "ПЯТАЯ": "5", "ШЕСТАЯ": "6", "СЕДЬМАЯ": "7", "ВОСЬМАЯ": "8",
}
_ROMAN_MAP = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
    "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
    "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15,
    "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20,
}

_PART_RE     = re.compile(r'^\s*ЧАСТЬ\s+(\w+)\s*$', re.IGNORECASE)
_CHAPTER_RE  = re.compile(r'^\s*(?:Глава|ГЛАВА)\s+([IVXLCDM]+|\d+)\s*$')
_ROMAN_RE    = re.compile(r'^\s*([IVXLCDM]{1,7})\.?\s*$')
_EPILOGUE_RE = re.compile(r'^\s*ЭПИЛОГ\s*$', re.IGNORECASE)


def annotate_book(docs: list, book_title: str) -> list:
    result = []
    for doc in docs:
        lines           = doc.page_content.split('\n')
        current_part    = "Вступление"
        current_chapter = "—"
        seg_lines: list = []

        def flush(part: str, chapter: str) -> None:
            text = '\n'.join(seg_lines).strip()
            if text:
                result.append(Document(
                    page_content=text,
                    metadata={
                        "source":     doc.metadata.get("source", ""),
                        "book_title": book_title,
                        "part":       part,
                        "chapter":    chapter,
                    }
                ))
            seg_lines.clear()

        for line in lines:
            pm = _PART_RE.match(line)
            cm = _CHAPTER_RE.match(line)
            rm = _ROMAN_RE.match(line)
            em = _EPILOGUE_RE.match(line)

            if pm or em:
                flush(current_part, current_chapter)
                if em:
                    current_part    = "Эпилог"
                    current_chapter = "—"
                else:
                    ordinal      = pm.group(1).upper()
                    current_part = f"Часть {_ORDINAL_MAP.get(ordinal, ordinal)}"
                    current_chapter = "—"
                seg_lines.append(line)
            elif cm or rm:
                flush(current_part, current_chapter)
                chap_str        = (cm.group(1) if cm else rm.group(1)).upper()
                current_chapter = f"Глава {_ROMAN_MAP.get(chap_str, chap_str)}"
                seg_lines.append(line)
            else:
                seg_lines.append(line)

        flush(current_part, current_chapter)
    return result


def format_docs(docs: list) -> str:
    parts = []
    for doc in docs:
        m   = doc.metadata
        ref = " | ".join(filter(None, [
            m.get("book_title"),
            m.get("part"),
            m.get("chapter"),
        ]))
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

# ─────────────────────────── промпты ─────────────────────────────────
ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Ты литературовед и литературный критик. Отвечай СТРОГО НА РУССКОМ ЯЗЫКЕ.\n"
        "Каждый фрагмент контекста помечен меткой вида [Название книги | Часть X | Глава Y]. "
        "Используй эти метки как источник при цитировании.\n\n"
        "Структура ответа — два блока:\n\n"
        "Цитаты из текста\n"
        "Приведи 2–5 прямых цитат, взятых ТОЛЬКО из предоставленного контекста. "
        "Если цитат нету - так и напиши"
        "После каждой цитаты укажи источник в скобках: "
        "(«Название книги», Часть X, Глава Y).\n\n"
        "Анализ\n"
        "Краткий ответ на вопрос на основе приведённых цитат. "
        "Не выходи за рамки контекста. "
        "Если информации недостаточно — честно сообщи об этом."
    ),
    MessagesPlaceholder("history"),
    (
        "human",
        "Контекст:\n{context}\n\nВопрос: {question}"
    ),
])

QUOTE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Ты литературовед. Отвечай СТРОГО НА РУССКОМ ЯЗЫКЕ.\n"
        "Каждый фрагмент контекста помечен меткой вида [Название книги | Часть X | Глава Y].\n\n"
        "Твоя задача — найти и привести ТОЧНЫЕ цитаты из предоставленного контекста, "
        "максимально близкие к запросу пользователя.\n\n"
        "Формат ответа:\n"
        "Для каждой найденной цитаты:\n"
        "1. Приведи цитату дословно, заключив в кавычки.\n"
        "2. Укажи источник: («Название книги», Часть X, Глава Y).\n"
        "3. Одно-три предложения — почему этот фрагмент отвечает на запрос.\n\n"
        "Не добавляй общих рассуждений. Только цитаты с атрибуцией.\n"
        "Если подходящих цитат нет — так и напиши."
    ),
    MessagesPlaceholder("history"),
    (
        "human",
        "Контекст:\n{context}\n\nЧто найти: {question}"
    ),
])


# ═══════════════════════ BACKEND CLASS ═══════════════════════════════
class PrefixedEmbeddings(Embeddings):
    def __init__(self, base, query_prefix="", doc_prefix=""):
        self.base         = base
        self.query_prefix = query_prefix
        self.doc_prefix   = doc_prefix

    def embed_documents(self, texts):
        return self.base.embed_documents([self.doc_prefix + t for t in texts])

    def embed_query(self, text):
        return self.base.embed_query(self.query_prefix + text)


class PageOracleBackend:
    def __init__(self, books_dir=".", persist_dir="./chroma_intro_db", log_callback=None):
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
            chunk_size=1500,
            chunk_overlap=300,
            add_start_index=True,
        )

    # ── инициализация ────────────────────────────────────────────────
    def initialize(self, provider="DeepSeek", model_name="deepseek-chat", api_key=""):
        self.log("[Инициализация] Загружаем эмбеддинги…")
        base_emb = HuggingFaceEmbeddings(model_name="ai-forever/ru-en-RoSBERTa")
        self.embeddings = PrefixedEmbeddings(
            base_emb, query_prefix="search_query: ", doc_prefix="search_document: ",
        )

        self.log("[Инициализация] Загружаем книги…")
        self._load_all_books()

        self.log("[Инициализация] Создаём векторное хранилище…")
        self._init_vectorstore()

        self.log("[Инициализация] Подключаем ИИ модель…")
        self.set_model(provider, model_name, api_key)

        self.log("[Инициализация] Система готова к работе!")

    def _load_all_books(self):
        txt_files = sorted(Path(self.books_dir).glob("*.txt"))
        if not txt_files:
            self.log(f"[!] Нет .txt файлов в «{self.books_dir}»")
            return
        all_annotated: list = []
        for tf in txt_files:
            all_annotated.extend(self._load_and_annotate(str(tf)))
            self.loaded_books.append(tf.name)
        self.splits = self.text_splitter.split_documents(all_annotated)
        self.log(f"Всего чанков: {len(self.splits)}")

    def _load_and_annotate(self, filepath: str) -> list:
        path = Path(filepath)
        book_title = path.stem.replace("_", " ").replace("-", " ")
        raw_docs = TextLoader(str(path), autodetect_encoding=True).load()
        annotated = annotate_book(raw_docs, book_title)
        self.log(f"  «{path.name}» → {len(annotated)} сегментов (part/chapter)")
        return annotated

    # ── векторное хранилище ──────────────────────────────────────────
    def _init_vectorstore(self):
        if not self.splits:
            return
        persist = Path(self.persist_dir)
        if persist.exists():
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir,
            )
            stored = self.vectorstore._collection.count()
            if stored != len(self.splits):
                self.log(f"[Индекс] Количество чанков изменилось ({stored} → {len(self.splits)}). Пересобираем…")
                shutil.rmtree(self.persist_dir)
                self.vectorstore = self._build_vectorstore()
            else:
                self.log(f"[Индекс] Загружен из кэша ({stored} чанков).")
        else:
            self.vectorstore = self._build_vectorstore()

        self.mmr_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 50, "lambda_mult": 0.65},
        )
        self.quote_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )

    def _build_vectorstore(self) -> Chroma:
        self.log("[Индекс] Создаём новый индекс…")
        return Chroma.from_documents(
            documents=self.splits,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
        )

    # ── модель и цепочки ─────────────────────────────────────────────
    def set_model(self, provider_name: str, model_name: str, api_key: str) -> bool:
        cfg = PROVIDERS.get(provider_name)
        if not cfg:
            self.log(f"[Ошибка] Неизвестный провайдер: {provider_name}")
            return False
        os.environ[cfg["env_key"]] = api_key
        try:
            module = importlib.import_module(cfg["package"])
            model_class = getattr(module, cfg["class"])
            self.model = model_class(model=model_name)
            self._create_chains()
            self.log(f"[Модель] {provider_name} / {model_name} — подключена.")
            return True
        except Exception as e:
            self.log(f"[Ошибка модели] {e}")
            return False

    def _create_chains(self):
        if not self.model or not self.vectorstore:
            return
        self.rag_chain = (
            {"context": self.mmr_retriever | format_docs, "question": RunnablePassthrough(), "history": lambda _: [],}
            | RunnableLambda(ensure_context)
            | ANALYSIS_PROMPT
            | self.model
            | StrOutputParser()
        ).with_config(run_name="rag_chain1")
        self.quote_chain = (
            {"context": self.quote_retriever | format_docs, "question": RunnablePassthrough(), "history": lambda _: [],}
            | RunnableLambda(ensure_context)
            | QUOTE_PROMPT
            | self.model
            | StrOutputParser()
        ).with_config(run_name="rag_chain2")

    # ── публичные методы ─────────────────────────────────────────────
    def add_document(self, filepath: str):
        path = Path(filepath.strip())
        if not path.exists():
            self.log(f"[Ошибка] Файл не найден: {path}")
            return
        if not self.vectorstore:
            self.log("[Ошибка] Векторная база не инициализирована.")
            return
        existing = self.vectorstore.get(where={"source": str(path)})
        if existing["ids"]:
            self.log(f"[Пропуск] «{path.name}» уже есть в базе ({len(existing['ids'])} чанков).")
            return
        new_splits = self.text_splitter.split_documents(self._load_and_annotate(str(path)))
        self.vectorstore.add_documents(new_splits)
        self.loaded_books.append(path.name)
        self.log(f"[Готово] Добавлено {len(new_splits)} чанков из «{path.name}».")

    """@traceable(name="answer_question")
    def answer_question(self, question: str) -> str:
        
        Основная точка входа в RAG.
        Эту функцию мы будем отслеживать в LangSmith как корневой run.
        
        return self.rag_chain.invoke(question)"""

    def ask(self, question: str, mode: str = "analysis") -> str:
        if mode == "quote":
            if not self.quote_chain:
                return "[Ошибка] Система не инициализирована."
            return self.quote_chain.invoke(question)
        else:
            if not self.rag_chain:
                return "[Ошибка] Система не инициализирована."
            return self.rag_chain.invoke(question)


# ═══════════════════════ CLI (обратная совместимость) ═════════════════
if __name__ == "__main__":
    backend = PageOracleBackend()
    backend.initialize()
    print("\nКоманды:")
    print("  /add <путь>    — добавить книгу в базу без пересборки индекса")
    print("  /цитата <текст> — точный поиск цитат (без анализа, только фрагменты)")
    print("  (без префикса)  — тематический анализ с цитатами и рассуждением")
    while True:
        question = input("\nВведите вопрос (или 'exit' для выхода): ").strip()
        if question.lower() in ["exit", "quit", "выйти", "выход", "закрыть", "завершить"]:
            break
        if question.lower().startswith("/add "):
            backend.add_document(question[5:])
            continue
        if question.lower().startswith("/цитата "):
            query = question[8:].strip()
            if query:
                print(backend.ask(query, mode="quote"))
            continue
        if not question:
            continue
        print(backend.ask(question))
