"""Agent 1 — `file_reader_agent`.

Reads every file in `DOCS_DIR`, concatenates them into a single corpus, and
stores the raw result in shared state under `output_key="corpus"`. Agent 2
reads `{corpus}` via template substitution (see tutorial §"The Two Ways Agents
Talk").

The agent must NOT summarize or analyze — that's Agent 2's job. Its only
responsibility is to materialize the corpus faithfully.
"""

from __future__ import annotations

from google.adk.agents.llm_agent import LlmAgent

from backend.model_router import llm_for
from backend.tools.file_reader import read_all_docs


file_reader_agent = LlmAgent(
    model=llm_for("reader"),
    name="file_reader_agent",
    description=(
        "Reads every readable file in DOCS_DIR (txt/md/json/csv/html/pdf/docx) "
        "and stores the concatenated corpus in state under 'corpus'."
    ),
    tools=[read_all_docs],
    instruction=(
        "You are a document loader. The user has placed files in DOCS_DIR and "
        "asked Agent 2 to analyze them.\n\n"
        "Your ONLY job: call the `read_all_docs` tool exactly once, then stop. "
        "Do NOT summarize, paraphrase, analyze, or answer the user. Do NOT ask "
        "questions — a SequentialAgent cannot route a turn back to you.\n\n"
        "When `read_all_docs` returns, echo its `corpus` field back verbatim so "
        "the framework stores it under `output_key=\"corpus\"`. If `status` is "
        "'error', surface the error message verbatim instead."
    ),
    output_key="corpus",
)