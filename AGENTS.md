# PageOracle — Agent Guidelines

## Repository Overview & Dual-Entrypoint Architecture

PageOracle is a Python RAG (Retrieval-Augmented Generation) application for questioning and analyzing books (`.txt` files) using LangChain, LangGraph, Chroma, BM25, and CrossEncoder reranking.

The project contains two distinct GUI frontends sharing identical backend logic:

- `web_gui_pageoracle/`: Flask web application (`gui.py`) serving an HTML/CSS/JS interface on port 5000.
- `tkinter_gui/`: Tkinter desktop application (`gui.py`).

**CRITICAL GOTCHA — Backend Synchronization**:
`web_gui_pageoracle/main.py` and `tkinter_gui/main.py` are separate physical copies of `PageOracleBackend`. Any modification to core RAG logic, prompts, models, retrievers, or state graphs **MUST** be kept in sync across both files.

---

## Developer Commands

Always `cd` into the target application directory before executing commands so relative paths (`./chroma_intro_db`, `settings.json`, `chat_history.json`, and `books_dir="."`) resolve correctly:

### Web GUI (Flask)

```bash
cd web_gui_pageoracle
python gui.py
```

- Listens at `http://127.0.0.1:5000`.
- **Note**: Importing `web_gui_pageoracle/gui.py` initiates `initialize_backend_async()` immediately in a background daemon thread.

### Desktop GUI (Tkinter)

```bash
cd tkinter_gui
python gui.py
```

### Running Tests

`pytest` is not installed by default in the environment. Run test suites with Python's standard `unittest`:

```bash
python -m unittest discover -s tests -p "test_*.py"
python -m unittest tests/test_<name>.py
```

### Quality Gates & Checks

All code changes must pass the automated quality gates before completion:

```bash
python scripts/check.py       # Aggregate check (Ruff, Ty, Deptry, Typos, Rumdl, Agnix, Jactionlint)
python scripts/fix.py         # Apply safe autofixes (Ruff format/check, Rumdl fmt)
# Or with mise:
mise run check
mise run fix
```

- **Enforcement**: Every quality gate check must exit with status 0. Never bypass checks with `--no-verify`.
- **Exceptions**: Lint and type rules are never disabled to get a change through. Exceptions must be narrow, scoped to specific files with an explicit documented reason.
- **Backend Sync**: Any changes to core logic in `web_gui_pageoracle/main.py` must be mirrored bit-for-bit in `tkinter_gui/main.py`.

---

## Environment & Runtime Quirks

- **PyTorch / Torchvision Mismatch**: If you encounter `RuntimeError: operator torchvision::nms does not exist` when importing `transformers` or `sentence_transformers`, `torch` and `torchvision` builds have an ABI mismatch (e.g. `torch` CPU vs `torchvision` CUDA). Ensure compatible wheels are installed per `requirements.txt`.
- **Yandex Cloud Integration**:
  - `ChatYandexGPT` (`langchain_community`) does not support `bind_tools` or `with_structured_output`. The backend explicitly falls back to direct prompt routing (`DIRECT` vs `RETRIEVE`) and manual regex output extraction.
  - Yandex embeddings (`text-search-doc/latest`) enforce a 10 req/s limit; `RateLimitedEmbeddings` throttles requests to 8 req/s.
  - Requires `YC_API_KEY` and `YC_FOLDER_ID`.
- **OpenRouter & Tool Calling**:
  - Default embedding: `nvidia/llama-nemotron-embed-vl-1b-v2:free` via direct HTTP in `OpenRouterEmbeddings` (batch size 32, 60s timeout).
  - Certain OpenRouter models fail on `tool_choice` or return empty content with tools; caught by fallback routing.
- **Chroma Vector Store Dimension Auto-Purge**:
  - Switching embedding models (e.g., between 1024-dim and 4096-dim) causes Chroma dimension mismatch errors.
  - `_init_vectorstore()` checks `embedding_config.json` and dimension probes; it will automatically delete `./chroma_intro_db` and re-index cached document splits.

---

## Key Architectural Patterns

- **Document Ingestion**:
  - `annotate_book()` parses Russian literary structures (Часть/Глава/Эпилог) with regex into metadata (`book_title`, `part`, `chapter`).
  - Text is chunked with `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`.
  - Adding a document via `add_document()` copies the file into `books_dir` and deduplicates by `source` path metadata.
- **Hybrid Retrieval & Reranker**:
  - `HybridRerankerRetriever` combines Chroma vector search (`mmr` or `similarity_score_threshold`) + `BM25Retriever`.
  - Merges documents, deduplicates by MD5 content hash, and reranks the top 7 with `CrossEncoder("BAAI/bge-reranker-base")`.
- **LangGraph Agent Workflow**:
  - Node `generate_query_or_respond`: Routes between tool call `retrieve` or direct response.
  - Conditional edge `_route_after_retrieve`: Uses `GradeDecision` to check if retrieved context helps. If "no" and rewrites < 5, routes to `rewrite_question`.
  - Node `rewrite_question`: Invokes `RewriteDecision` (with regex stripping of markdown/query prefixes) and loops back to query generator.
  - Final nodes: `generate_analysis_answer` (`ANALYSIS_PROMPT` -> citations + analytical explanation) or `generate_quote_answer` (`QUOTE_PROMPT` -> verbatim quotes with structural source labels).
- **Modes**:
  - `auto`: Autonomous routing and style selection.
  - `analysis`: Forces analytical response style.
  - `quote`: Forces verbatim excerpt citations.
- **Persistence**:
  - `settings.json`: Stores active provider, model, API keys, temperature, max_tokens, top_p, and score_threshold.
  - `chat_history.json`: Stores last 20 messages (`user`, `assistant`, `system`).

---

## Development Principles & OpenSpec

- Do not preserve backward compatibility.
- Choose the simplest implementation that fully meets requirements.
- Fix the root cause, not the symptom.
- Suggest best practices, even if they require refactoring.
- **OpenSpec (`/opsx-apply`) TDD Rule**:
  1. For every task in `tasks.md` introducing or modifying functionality, invoke the `tdd` skill.
  2. Write a failing test first translating GIVEN/WHEN/THEN criteria from the corresponding `specs/*.md` file.
  3. Mark the task complete in `tasks.md` only after the test passes and passes the TDD refactoring step.

---

## Tool Selection Guidance

- **Serena**: Primary tool for symbol-level code navigation, inspections, definitions, references, and targeted edits.
- **Graphify**: Dependency mapping, cross-module impact analysis, and architectural exploration.
- Use Serena for known symbols, local bug fixes, and single-file changes. Escalate to Graphify when evaluating multi-component or cross-module architectural changes.

---

<!-- workflow-addons:start -->
## No Sycophancy

The user does not need a compliant assistant that agrees with everything. The goal is truth and grounding in reality.

- Do not agree out of politeness. If the user's premise contradicts facts or data, show the numbers and state directly that the premise is incorrect. Do not soften, do not bury the conclusion at the end, and never start with "great idea, but...".
- Analyze facts, not tone. An objection must be backed by calculations, source links, or counterexamples — not "it seems to me", but "here is the data, here is what follows".
- If you believe your solution is better, defend it with justification rather than yielding at the first pushback. The final choice always remains with the user.
- If the user is right, state it directly and concisely. Agreement must also be justified.
- Acknowledge mistakes immediately and objectively, fix them directly in the files, without lengthy apologies.
- Distinguish established facts from assumptions: "data demonstrates" and "my hypothesis" have different statuses. Clearly mark weak inferences as weak.
- Do not tailor analysis to a desired outcome. If data is insufficient for the desired conclusion, say so plainly.
- If a plan contradicts arithmetic, deadlines are unrealistic, or expectations are inflated, raise it during discussion, not after failure.
This is not a license for rudeness or arguing for the sake of arguing. It is a requirement for substance: arguments instead of blind agreement.

## Decision Log (`docs/DECISIONS.md`)

- Record every significant decision (architecture, technology choices, data models, user behavior, rejected alternatives) in the decision log immediately after the user makes it.
- Structure of the file — three sections:
  1. **Table of Contents** — table, one row per decision: `| D-<number> | <date> | <domain> | <short title> | <status> |`. Domains are a short fixed list established when creating the log (e.g.: architecture, data, rag-pipeline, gui, workflow).
  2. **Decisions** — active records following the format below.
  3. **Archive** — superseded records.
- Add new decisions in two places simultaneously: a row in the table of contents and an entry under "Decisions".
- Entry length: 4–6 lines, each field in 1–2 sentences without replaying the chat transcript.
- Entry format:

  ```markdown
  ## D-<number>. <Short Title> (<date>)
  **Question:** What was discussed.
  **Options:** Proposed alternatives with pros and cons.
  **Chosen:** What the user chose.
  **Why:** Reason for choice in the user's words.
  **Why it should work:** Technical or rational justification.
  **Affects:** What parts of the system are impacted.
  **Status:** Active / Superseded by D-<number>.
  ```

- Never delete or rewrite old entries. If a decision is reconsidered: create a new entry, mark the old one as "Superseded by D-<number>", move the old entry to "Archive" unchanged (except for status), and update its status in the table of contents.
- Before planning, designing, or refactoring an existing part of the system, read the table of contents in its entirety, and read detailed entries only for the affected domain and referenced decisions. Do not read the entire file. Ground choices in active decisions. If a new plan contradicts an entry, point it out explicitly and ask if the decision should be revisited.
- This rule applies inside skills (brainstorming, writing-plans): the log must be consulted before proposing options.
- If a decision represents a rule that must govern every session (e.g., "always keep web_gui and tkinter_gui synchronized"), record it in the agent rules as well. The log stores why it was decided; rules store what to do.
- When active decisions exceed 100 or multiple collaborators join, propose migrating to a `docs/decisions/` directory with one file per decision and an index file.

## Work Summary

Upon completing any task, stage, or plan, provide a structured summary. Never say "everything is done" without it. Structure:

1. **What was done** — completed tasks and functionality.
2. **Autonomous decisions made** — especially choices affecting architecture or behavior, with links to decision log entries if applicable.
3. **Where changes occurred** — files, modules, components modified.
4. **Why this implementation** — rationale behind the solution.
5. **Impact on neighboring components** — affected areas across the project (remember to verify synchronization between `web_gui_pageoracle` and `tkinter_gui`).
6. **Discovered issues** — newly surfaced bugs, limitations, or dependencies.
7. **Incomplete items** — everything deferred, along with the reason.
8. **Decisions required from user** — questions that cannot be resolved autonomously.
9. **Recommended next steps** — what to verify, fix, or improve next.
Items with nothing to report must remain with "None" rather than omitted: an empty item is also crucial information.
A stage is not considered complete until the user reviews the summary and confirms completion. If the user confirms without addressing controversial points, remind them once.

## No Hidden Technical Debt

Without explicit user approval, it is forbidden to:

- drop any functionality from the agreed plan;
- leave unfinished tasks without notice;
- defer work indefinitely ("later", "in a future iteration");
- introduce stubs, TODOs, or temporary workarounds without reporting them.
If something needs to be deferred, explain why and request a decision.
Two-wave cycle:

1. Primary wave of work according to the plan.
2. Planned tests and validations.
3. Separate report: what remains incomplete and what surfaced during validation.
For each item, the user decides: fix now (if related to current stage or trivial) / schedule a dedicated wave immediately after the current one / consciously defer with a mandatory tracker entry.
Deferring work is a product or technical decision belonging to the user, not the agent. Only recommend deferring non-blocking items with medium or lower priority that require disproportionate refactoring, backed by solid justification.

## Tasks and Deferred Debt — GitHub Projects (Project #3 pageoracle)

- Every consciously deferred task must be immediately logged to GitHub Projects (Project #3 `pageoracle`) via `gh` CLI (or GitHub MCP); the user never does this manually.
- Task contents: what was not completed, why it was deferred, who made the decision and when, dependencies, what needs to be done in the future, and link to the decision log entry if applicable.
- In the work summary, list links or references to created tasks.
- Before beginning a new stage, check open project tasks: evaluate whether any should be brought back into active work.

## Rules Accumulation (Section "Accumulated Rules" in AGENTS.md)

- After every user critique or correction, analyze: is this a one-off edit or does it reflect a general working principle?
- If it is general, formulate a broader rule and propose recording it to the user. Record only after user consent.
- Format: rule; **Why:** what feedback prompted it; **How to apply:** when it triggers; date.
- Rules govern agent behavior: how to discuss decisions, handle errors, honor agreements, report changes, when to request confirmation, and what anti-patterns to avoid.
- UI styling, padding, colors, fonts, and components belong in a design system, not here.
- Do not turn every single request into a global rule. If in doubt, ask if it is a one-off preference.
- Before adding a rule, check for existing related rules: update existing ones rather than creating duplicates. If a conflict arises, present it to the user.

## Checkpoints and External Memory

All critical project state must reside in files, not solely in the transient chat dialogue:

| Document | What it restores |
|---|---|
| Specification | How the project is designed |
| Decision Log | Why it was designed that way |
| Code Map | Where functionality is located |
| Plan | What needs to be done |
| Task Tracker | What was deferred and why |
| Checkpoint | Where the agent stopped |
| Rules | How the agent must operate |

Checkpoint location: `docs/CHECKPOINT.md` in the main project folder (not in a worktree branch copy). It is a single, continuously overwritten file that must not be committed to git (ensure it is listed in `.gitignore`). Create it whenever starting a multi-step task.
Update it: after each completed plan step, before launching subagents, before concluding a session, and whenever context becomes long.
Contents:

```markdown
# Checkpoint — <date, time>
**Stage:** ...
**Active Plan:** <path to plan file>
**Completed:** ...
**Incomplete:** ...
**Subagents:** active subagents, tasks, status.
**Next Action:** single concrete next action.
**Read Before Resuming:** list of documents.
```

When the user requests to resume from a checkpoint: read `docs/CHECKPOINT.md` and the listed documents, briefly summarize to the user where work paused, and only then proceed.
After each stage, synchronize documents — even if another skill or rule also mentions this:

- Specifications: reflect actual implementation (architecture in technical spec, user behavior in product spec);
- Decision log: record new decisions with rationale and consequences;
- Code map: reflect new files, modules, relocations, and major changes;
- Plan and task status: mark what is done, active, deferred with permission, new dependencies, and next wave scope.
In the work summary, include a line explicitly listing which documents were updated.

## Accumulated Rules
<!-- accumulated-rules:start -->
<!-- Add new behavioral rules here as they emerge from user feedback -->
<!-- accumulated-rules:end -->
<!-- workflow-addons:end -->
