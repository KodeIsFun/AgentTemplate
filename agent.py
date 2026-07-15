"""Root agent — the `adk run .` and `adk web .` commands load this file.

Pattern reused verbatim from §"Use Case 1 / 3. Compose the root agent" in the
tutorial: a `SequentialAgent` that runs the file reader, then the analyst,
passing the corpus forward via the `{corpus}` template in the analyst's
instruction.
"""

from __future__ import annotations

from google.adk.agents.sequential_agent import SequentialAgent

from backend.agents.analyzer import analyst_agent
from backend.agents.file_reader import file_reader_agent


root_agent = SequentialAgent(
    name="doc_analysis_pipeline",
    sub_agents=[file_reader_agent, analyst_agent],
    description=(
        "Reads every file in docs/, then performs an extensive analysis of "
        "the corpus against the user's original prompt. The analyst may call "
        "Exa at most 3 times if facts are missing."
    ),
)