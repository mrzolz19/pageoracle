# Tutorial: pageoracle 
PageOracle — это настольное приложение на базе искусственного интеллекта, разработанное для того, чтобы помочь пользователям взаимодействовать с учебниками. Оно позволяет загружать книги (например, .txt файлы), задавать вопросы об их содержании и получать интеллектуальные ответы, сгенерированные искусственным интеллектом. Система преобразует книги в базу знаний с возможностью поиска, интегрируется с различными большими языковыми моделями (БЯМ) и управляет историей ваших диалогов с помощью интуитивно понятного ИИ-агента на основе графов и удобного графического интерфейса.

## Visual Overview 

```mermaid 
flowchart TD
  A0["User Interface (GUI)
"]
  A1["Application Settings
"]
  A2["Book Processing & Knowledge Base
"]
  A3["LLM Integration & Configuration
"]
  A4["Retrieval Mechanisms
"]
  A5["Conversational AI Agent (LangGraph)
"]
  A6["Chat History & Context Management
"]
  A0 – "Loads/Edits Settings" --> A1
  A0 – "Manages Books" --> A2
  A0 – "Sends Queries to Agent" --> A5
  A0 – "Manages Chat History" --> A6
  A1 – "Configures LLM" --> A3
  A1 – "Configures Retrievers" --> A4
  A2 – "Populates Vector DB for" --> A4
  A3 – "Provides LLM to Agent" --> A5
  A4 – "Provides Context to Agent" --> A5
  A5 – "Uses/Updates History" --> A6
  A6 – "Provides History to Agent" --> A5
```

## Chapters 
1. [User Interface (GUI) ](01_user_interface__gui__.md)
2. [Book Processing & Knowledge Base ](02_book_processing___knowledge_base_.md)
3. [Application Settings ](03_application_settings_.md)
4. [LLM Integration & Configuration ](04_llm_integration___configuration_.md)
5. [Retrieval Mechanisms ](05_retrieval_mechanisms_.md)
6. [Conversational AI Agent (LangGraph) ](06_conversational_ai_agent__langgraph__.md)
7. [Chat History & Context Management ](07_chat_history___context_management_.md)


