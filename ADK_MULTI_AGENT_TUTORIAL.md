# Setting Up a Multi-Agent System with Google ADK (Python)

A practical, end-to-end guide to building multi-agent workflows with the
**Google Agent Development Kit (ADK)** in Python. It is framework-agnostic and
generic — no product-specific details.

This guide shows **two use cases** of the same machinery:

1. **Technical / LLM-driven pipeline** — a backend-style flow where one agent's
   structured output feeds the next, suitable for being called programmatically
   (e.g. by another service or an LLM orchestrator).
2. **Layman / content-creation pipeline** — a "topic → research → script"
   flow you can demo in a YouTube video or explain to a non-developer.

Both share the same ADK building blocks, so we cover the foundations once, then
apply them.

---

## Table of Contents

- [Core Concepts](#core-concepts)
- [Project Layout](#project-layout)
- [Installation & Environment](#installation--environment)
- [The Two Ways Agents Talk](#the-two-ways-agents-talk)
- [Use Case 1 — Technical LLM-Driven Pipeline](#use-case-1--technical-llm-driven-pipeline)
- [Use Case 2 — Layman Content Pipeline (YouTube-friendly)](#use-case-2--layman-content-pipeline-youtube-friendly)
- [Running Agents](#running-agents)
- [Structured Output & Validation](#structured-output--validation)
- [Using Non-Gemini Models (LiteLLM / OpenRouter)](#using-non-gemini-models-litellm--openrouter)
- [Common Pitfalls](#common-pitfalls)
- [When to Use What](#when-to-use-what)

---

## Core Concepts

| Concept | What it is |
|---|---|
| `LlmAgent` | A single agent backed by an LLM. Has a `model`, `name`, `instruction`, optional `tools`, and optional `output_schema`. |
| `SequentialAgent` | A parent agent that runs a list of `sub_agents` in order, one after another. |
| `ParallelAgent` | Runs sub-agents concurrently (when stages are independent). |
| `LoopAgent` | Runs sub-agents in a loop until a condition is met. |
| `output_key` | Tells an agent to store its final result in shared **state** under this key. |
| `{key}` template | In a later agent's `instruction`, `{key}` is replaced with whatever was stored under `output_key="key"`. |
| `tools` | Python functions (or MCP tools) the agent can call — web search, file read, API calls, etc. |
| `output_schema` | A Pydantic model that constrains the agent's output to structured JSON. |
| `LiteLlm` | ADK wrapper that routes to 100+ models (OpenAI, Anthropic, OpenRouter, etc.) via LiteLLM. |

**The golden rule of multi-agent design in ADK:** agents communicate by writing
to and reading from **shared state**. A `SequentialAgent` guarantees *order*;
`output_key` + `{key}` guarantees *data handoff*.

---

## Project Layout

A clean, generic multi-agent project looks like this:

```
my_agent/
├── agent.py                 # root_agent = SequentialAgent (the shim ADK loads)
├── .env                     # secrets (gitignored)
├── .env.example             # template
├── requirements.txt
└── backend/
    ├── __init__.py
    ├── config.py            # env loading + model-tier map
    ├── model_router.py      # LiteLlm factory
    ├── runner.py            # one-shot InMemoryRunner helper
    ├── schemas/
    │   └── *.py             # Pydantic output contracts
    ├── tools/
    │   └── *.py             # agent tools (search, fetch, etc.)
    └── agents/
        └── *.py             # one module per agent
```

ADK's CLI (`adk run .`) expects to find `root_agent` imported from `agent.py`
in the current directory.

---

## Installation & Environment

Create a virtual environment and install the framework:

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# Windows PowerShell: venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install google-adk requests python-dotenv
```

`requirements.txt` (pin real versions in your own project):

```
google-adk
litellm
requests
python-dotenv
fastapi
uvicorn[standard]
```

Create a `.env` file from the template:

```env
# Gemini (default if you use google.adk.models directly)
GOOGLE_API_KEY=...

# If routing through OpenRouter / LiteLLM instead of Gemini:
OPENROUTER_API_KEY=sk-or-v1-...

# Any external tool keys your agents need:
EXA_API_KEY=...

# Model routing (optional, read by config.py)
MODEL_LIGHT=openrouter/openai/gpt-4o-mini
MODEL_RESEARCH=openrouter/anthropic/claude-3.5-sonnet
MODEL_SCRIPT=openrouter/openai/gpt-4o
```

Load it at the top of your package:

```python
# backend/config.py
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
EXA_API_KEY = os.getenv("EXA_API_KEY", "")
```

> **Never commit `.env`.** Keep real keys out of `.env.example`.

---

## The Two Ways Agents Talk

### A) State handoff (the ADK-native way)

Agent 1 stores its result in state; Agent 2 reads it via `{key}`.

```python
from google.adk.agents.llm_agent import LlmAgent

first = LlmAgent(
    model="gemini-2.0-flash",
    name="researcher",
    instruction="Research the topic and write a brief.",
    output_key="research_brief",   # → state["research_brief"]
)

second = LlmAgent(
    model="gemini-2.0-flash",
    name="writer",
    instruction="Research brief:\n{research_brief}\n\nNow write the output.",  # reads it
    output_key="final_output",
)
```

### B) SequentialAgent wraps them

```python
from google.adk.agents.sequential_agent import SequentialAgent

root_agent = SequentialAgent(
    name="pipeline",
    sub_agents=[first, second],
    description="Runs first, then second, passing state forward.",
)
```

`SequentialAgent` runs `first` to completion, then `second`. The `{research_brief}`
substitution happens automatically because `first` set `output_key`.

**Important limitation:** a `SequentialAgent` runs each sub-agent *autonomously*
— it cannot pause a sub-agent to ask the user a question and resume. So every
agent instruction should forbid asking questions and force immediate action.

---

## Use Case 1 — Technical LLM-Driven Pipeline

**Goal:** a deterministic, programmatically-callable pipeline where one agent
produces a *structured* artifact that the next agent consumes. This is the
pattern you'd embed behind an API or call from another LLM orchestrator.

**Flow:** `Topic → Analyzer Agent → Summarizer Agent → structured JSON`

### 1. Define the output contracts (Pydantic)

```python
# backend/schemas/analysis.py
from pydantic import BaseModel, Field


class AnalysisFinding(BaseModel):
    angle: str = Field(..., description="The subtopic analyzed")
    key_facts: list[str] = Field(..., description="Concrete facts found")
    confidence: float = Field(..., description="0.0–1.0 confidence score")


class AnalysisReport(BaseModel):
    topic: str
    summary: str
    findings: list[AnalysisFinding] = Field(default_factory=list)


class Summary(BaseModel):
    headline: str
    bullets: list[str] = Field(default_factory=list)
    recommended_action: str
```

### 2. Build the agents

```python
# backend/agents/analyzer.py
from google.adk.agents.llm_agent import LlmAgent
from backend.schemas.analysis import AnalysisReport
from backend.model_router import llm_for   # optional LiteLLM routing

analyzer_agent = LlmAgent(
    model=llm_for("research"),
    name="analyzer_agent",
    description="Decomposes a topic into angles and analyzes each.",
    instruction=(
        "You are an analyst. Given a topic, decompose it into 3–6 distinct "
        "angles, gather concrete facts for each, and score your confidence. "
        "Return ONLY JSON matching the AnalysisReport schema."
    ),
    output_schema=AnalysisReport,
    output_key="analysis_report",
)
```

```python
# backend/agents/summarizer.py
from google.adk.agents.llm_agent import LlmAgent
from backend.schemas.analysis import Summary

summarizer_agent = LlmAgent(
    model=llm_for("script"),
    name="summarizer_agent",
    description="Turns an analysis report into a crisp actionable summary.",
    instruction=(
        "Analysis report:\n{analysis_report}\n\n"   # <-- reads state from analyzer
        "Produce a headline, 3–5 bullets, and one recommended action. "
        "Return ONLY JSON matching the Summary schema."
    ),
    output_schema=Summary,
    output_key="summary",
)
```

### 3. Compose the root agent

```python
# agent.py
from google.adk.agents.sequential_agent import SequentialAgent
from backend.agents.analyzer import analyzer_agent
from backend.agents.summarizer import summarizer_agent

root_agent = SequentialAgent(
    name="technical_pipeline",
    sub_agents=[analyzer_agent, summarizer_agent],
    description="Analyzes a topic then summarizes the findings.",
)
```

### 4. Run it one-shot (stateless, perfect for an API)

```python
# backend/runner.py
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part

async def run_agent_once(agent, user_message: str, app_name: str = "app"):
    runner = InMemoryRunner(agent=agent, app_name=app_name)
    session = await runner.session_service.create_session(
        app_name=app_name, user_id="svc"
    )
    message = Content(role="user", parts=[Part.from_text(text=user_message)])
    final_text = ""
    async for event in runner.run_async(
        user_id="svc", session_id=session.id, new_message=message
    ):
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if part.text:
                    final_text = part.text
    return final_text
```

```python
# call it
import asyncio
from agent import root_agent
from backend.runner import run_agent_once

result = asyncio.run(
    run_agent_once(root_agent, "the impact of open-source AI models in 2026")
)
print(result)   # final JSON (the summarizer's Summary)
```

**Why this is "technical":** it expects structured JSON (`output_schema`), it's
stateless per call, and the output is machine-consumable — exactly what another
service or an LLM orchestrator would want. You can also front each agent with
its own HTTP endpoint so a caller drives stages step by step and injects context
between them.

---

## Use Case 2 — Layman Content Pipeline (YouTube-friendly)

**Goal:** the same machinery, framed for a non-technical audience. We build a
**"Topic → Research → Video Script"** pipeline — easy to narrate on camera and
easy for a viewer to grasp: *"one AI researcher gathers facts, then a second AI
writer turns those facts into a script."*

### The mental model (say this on camera)

> "Think of two specialist assistants. Assistant #1 is a **researcher**: you
> give it a topic, it splits the topic into angles, searches the web for each,
> and writes a tidy brief. Assistant #2 is a **scriptwriter**: it reads that
> brief and writes a full video script. They work in a fixed order — research
> first, then write — and the brief is passed from one to the other
> automatically."

### The research agent (with a web-search tool)

```python
# backend/tools/web_search.py
import os, requests

def search_web(query: str) -> dict:
    """Search the web for current information."""
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        return {"status": "error", "error": "EXA_API_KEY missing"}
    try:
        resp = requests.post(
            "https://api.exa.ai/search",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "query": query,
                "numResults": 5,
                "contents": {"text": {"maxCharacters": 1000}},  # cap to avoid overflow
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "status": "success",
            "results": [
                {"title": r.get("title", ""), "url": r.get("url", ""),
                 "text": r.get("text", "")}
                for r in data.get("results", [])
            ],
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
```

```python
# backend/agents/research.py
from google.adk.agents.llm_agent import LlmAgent
from backend.tools.web_search import search_web

research_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="research_agent",
    description="Researches a topic via web search and writes a brief.",
    instruction="""You are a research analyst. Given a topic:
1. Decompose it into 3–6 distinct angles.
2. Call search_web ONCE per angle with a focused query.
3. Compile a brief with concrete facts and sources.
Never ask questions — decide the angles yourself and search.
Your output MUST be JSON: {"topic": str, "summary": str,
"angles": [{"angle": str, "key_facts": [str], "sources": [{"title": str, "url": str}]}]}""",
    tools=[search_web],
    output_key="research_brief",
)
```

### The scriptwriter agent (reads the brief)

```python
# backend/agents/script.py
from google.adk.agents.llm_agent import LlmAgent

script_writer_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="script_writer_agent",
    description="Writes a video script from a research brief.",
    instruction="""You are a video scriptwriter. Read the research brief below
and turn it into a complete script.

Research brief:
{research_brief}

Write: TITLE, HOOK, INTRO, BODY (3–5 sections with narration + visual cues),
OUTRO, CTA. Use only facts from the brief — never invent.
Return ONLY JSON: {"title": str, "hook": str, "intro": str,
"body": [{"heading": str, "narration": str, "visual_cues": [str]}],
"outro": str, "cta": str}""",
    output_key="video_script",
)
```

### Compose & run

```python
# agent.py
from google.adk.agents.sequential_agent import SequentialAgent
from backend.agents.research import research_agent
from backend.agents.script import script_writer_agent

root_agent = SequentialAgent(
    name="topic_to_video_script_agent",
    sub_agents=[research_agent, script_writer_agent],
    description="Researches a topic, then writes a video script from it.",
)
```

Run interactively (great for a screen recording):

```bash
adk run .
```

Then type a topic like `the rise of small open-source AI models` and watch the
two agents run in sequence, producing a script.

**Why this is "layman":** no structured schema enforcement is required for the
demo, the flow maps to a relatable story (researcher → writer), and `adk run .`
gives a live, narratable chat interface perfect for YouTube.

---

## Running Agents

**Interactive chat (great for demos):**
```bash
source venv/bin/activate
adk run .            # then type a topic; type exit to quit
```

**Single-shot from code:**
```python
import asyncio
from agent import root_agent
from backend.runner import run_agent_once

print(asyncio.run(run_agent_once(root_agent, "your topic here")))
```

**Verify the agent loads:**
```bash
python -c "from agent import root_agent; print([a.name for a in root_agent.sub_agents])"
# → ['research_agent', 'script_writer_agent']
```

**Web UI (ADK dev console):**
```bash
adk web .
```

---

## Structured Output & Validation

When you set `output_schema=SomePydanticModel`, ADK asks the model for JSON
that matches. Models occasionally return prose or a fenced code block, so
**validate and repair** at the boundary:

```python
import json
from pydantic import ValidationError

def parse_output(raw: str, schema):
    text = raw.strip()
    if text.startswith("```"):                  # strip ```json fences
        text = text[text.find("\n")+1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return schema.model_validate(json.loads(text.strip()))
```

If validation fails, retry once with a repair prompt ("Return ONLY valid JSON
matching the schema"). If it still fails, surface the raw text for debugging
rather than crashing.

> **Tip:** cap tool output size (`maxCharacters` in search, truncated fetches)
> so a few tool calls don't blow past the model's context window.

---

## Using Non-Gemini Models (LiteLLM / OpenRouter)

ADK's `LiteLlm` lets any agent run on 100+ models. Centralize the map:

```python
# backend/config.py
import os
from dotenv import load_dotenv
load_dotenv()

MODEL_TIERS = {
    "research": os.getenv("MODEL_RESEARCH", "openrouter/anthropic/claude-3.5-sonnet"),
    "script":   os.getenv("MODEL_SCRIPT",   "openrouter/openai/gpt-4o"),
    "light":    os.getenv("MODEL_LIGHT",    "openrouter/openai/gpt-4o-mini"),
}
```

```python
# backend/model_router.py
from google.adk.models.lite_llm import LiteLlm
from backend import config

def llm_for(tier: str) -> LiteLlm:
    return LiteLlm(model=config.MODEL_TIERS[tier])   # KeyError = fail fast
```

Then `model=llm_for("research")` in any agent. OpenRouter authenticates from
`OPENROUTER_API_KEY` in the environment — no auth code needed.

---

## Common Pitfalls

1. **Agents asking the user questions.** A `SequentialAgent` can't route a later
   turn back to a sub-agent. Forbid questions in every instruction; force
   immediate autonomous action.
2. **One agent doing the next agent's job.** Keep roles strict — the researcher
   must not write the script. This prevents "finishing early" with half-baked
   output.
3. **Hallucinated facts.** Give the writer *no* search tool and tell it to use
   only the brief. The researcher owns all factual gathering.
4. **Context overflow.** Cap tool output (`maxCharacters`, truncations). A few
   uncapped searches can exceed the 128k limit.
5. **Missing `output_key` / wrong `{key}` name.** The template substitution is a
   literal name match — typos silently pass empty strings.
6. **Forgetting to activate the venv.** `adk: command not found` almost always
   means the virtual environment isn't active.
7. **Committing secrets.** `.env` must be gitignored; only `.env.example` is
   tracked.
8. **Unvalidated JSON.** Always `model_validate` agent JSON at the boundary,
   with a repair retry.

---

## When to Use What

| Need | Use |
|---|---|
| Fixed, ordered stages (research → write) | `SequentialAgent` |
| Independent stages you can run at once | `ParallelAgent` |
| Repeat-until-good (e.g. refine) | `LoopAgent` |
| Dynamic routing / conditional next step | Graph-based workflow (ADK workflows) |
| Another service or LLM drives stages with review between them | Stateless per-endpoint agents + `InMemoryRunner` |
| Non-Gemini models | `LiteLlm` + model-tier map |
| Machine-consumable output | `output_schema` (Pydantic) + validation |
| Live, narratable demo | `adk run .` or `adk web .` |

---

## Minimal Working Example (copy-paste)

```python
# agent.py
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent

greeter = LlmAgent(
    model="gemini-2.0-flash",
    name="greeter",
    instruction="Greet the user and echo their topic back.",
    output_key="topic",
)
responder = LlmAgent(
    model="gemini-2.0-flash",
    name="responder",
    instruction="The user's topic is: {topic}\nWrite a one-line take on it.",
    output_key="reply",
)

root_agent = SequentialAgent(
    name="demo", sub_agents=[greeter, responder], description="Greet then respond."
)
```

```bash
pip install google-adk python-dotenv
export GOOGLE_API_KEY=...
adk run .
```

That's the entire multi-agent pattern: define agents, wire them with
`output_key` + `{key}`, wrap in a `SequentialAgent`, and run.
