"""Agent 2 — `analyst_agent`.

Performs an extensive analysis of the corpus received from Agent 1 against
the user's original prompt, optionally augmenting with up to `MAX_EXA_CALLS`
Exa lookups (tool-level guard inside `exa_search` enforces the same limit).

Output is **free-form markdown** so the analyst can shape its reply to
whatever the user's prompt asked for (summary, comparison, checklist, etc.).
"""

from __future__ import annotations

from google.adk.agents.llm_agent import LlmAgent

from backend import config
from backend.model_router import llm_for
from backend.tools.web_search import exa_search


# Build the cap sentence from config so .env changes propagate to the prompt.
_MAX_EXA = config.MAX_EXA_CALLS

_INSTRUCTION = f"""You are an expert analyst. You receive:
  1. A user prompt describing what to prepare / produce from a corpus.
  2. The corpus, already loaded by Agent 1, available as {{corpus}}.

Do an EXTENSIVE analysis grounded in the corpus:

  - Read every file's contents in {{corpus}} carefully.
  - Cross-reference facts, dates, entities, and themes across files.
  - Identify gaps, contradictions, and trends.
  - Produce the deliverable the user asked for (summary, brief, comparison,
    checklist, plan, etc.) in well-structured markdown.

Hard rules:

  - Do NOT ask the user clarifying questions. Decide and produce.
  - Do NOT invent facts that aren't in the corpus. If something is missing,
    say so explicitly and propose a reasonable interpretation.
  - You MAY call `exa_search(query)` at most {_MAX_EXA} TIMES per run. The
    tool itself will refuse after that limit and return `limit_reached`.
  - Prefer web search ONLY when a fact is genuinely missing and critical
    to the user's request. Each call must use a focused, multi-word query.
  - When you cite a web result, include the `title` and `url` in your reply.

Return ONLY the final markdown answer. No meta-commentary, no preamble.
"""


analyst_agent = LlmAgent(
    model=llm_for("analyst"),
    name="analyst_agent",
    description=(
        "Performs an extensive analysis of the corpus received from "
        "file_reader_agent against the user's prompt. May consult the web "
        f"via Exa up to {_MAX_EXA} times."
    ),
    tools=[exa_search],
    instruction=_INSTRUCTION,
    output_key="analysis",
)