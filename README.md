# Doc Analysis Pipeline (Google ADK)

A two-step Google ADK pipeline:

```
User prompt + files in docs/
        │
        ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│  file_reader_agent       │ →  │  analyst_agent           │
│  reads DOCS_DIR          │    │  reads {corpus}          │
│  (txt/md/json/csv/html/  │    │  + original user prompt  │
│   pdf/docx)              │    │  may call exa_search     │
│  output_key="corpus"     │    │  up to MAX_EXA_CALLS     │
└──────────────────────────┘    └──────────────────────────┘
                                            │
                                            ▼
                                   Free-form markdown
```

The agents communicate exclusively through shared state (see tutorial §"The
Two Ways Agents Talk" / `output_key` + `{key}` template substitution).

---

## Project Layout

```
.
├── agent.py                    # root_agent = SequentialAgent (ADK CLI shim)
├── requirements.txt
├── .env.example                # copy → .env and fill in real keys
├── docs/                       # ← Agent 1 reads everything here
│   └── sample.md
├── backend/
│   ├── config.py               # env + corpus/Exa caps + model tiers
│   ├── model_router.py         # LiteLlm factory
│   ├── runner.py               # run_agent_once() one-shot helper
│   ├── agents/
│   │   ├── file_reader.py      # Agent 1 — emits {corpus}
│   │   └── analyzer.py         # Agent 2 — emits {analysis} (markdown)
│   ├── tools/
│   │   ├── file_reader.py      # read_all_docs(): text + PDF + DOCX
│   │   ├── web_search.py       # exa_search() with MAX_EXA_CALLS guard
│   │   └── agentmail_tool.py   # AgentMail SDK wrappers (inboxes + messages)
│   └── schemas/                # (empty — Agent 2 returns free-form markdown)
└── README.md
```

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and fill in EXA_API_KEY (and either GOOGLE_API_KEY or OPENROUTER_API_KEY)
```

Key environment variables (see `.env.example` for the full list):

| Var                | Purpose                                                                |
|--------------------|------------------------------------------------------------------------|
| `DOCS_DIR`         | Folder Agent 1 walks. Default `./docs`.                                |
| `MAX_EXA_CALLS`    | Hard cap on Exa searches per process. Default `3`.                     |
| `MAX_CHARS_PER_FILE` | Truncation cap per file (default `10000`) to avoid context overflow. |
| `MAX_CORPUS_CHARS` | Truncation cap on the whole corpus (default `120000`).                 |
| `MODEL_READER`     | LiteLLM model string for Agent 1.                                      |
| `MODEL_ANALYST`    | LiteLLM model string for Agent 2.                                      ||| `AGENTMAIL_API_KEY` | API key for the [AgentMail](https://github.com/agentmail-to/agentmail-python) email tool. Leave blank to skip email tools entirely. |
|| `MAX_AGENTMAIL_CALLS` | Hard cap on AgentMail tool invocations per process (default `10`). |
|| `MAX_MESSAGES_PER_LIST` | Cap on the `limit` parameter for `list_messages` / `list_inboxes` (default `20`). |
|| `MAX_MESSAGE_PREVIEW_CHARS` | Truncation cap on returned message bodies (default `2000`). || `PROVIDER_API_BASE_<VENDOR>` | Optional custom `api_base` for any vendor (e.g. `PROVIDER_API_BASE_OPENAI`, `PROVIDER_API_BASE_GROQ`). |

---

## Providers (OpenAI-compatible & more)

Model strings go through LiteLLM, so any provider LiteLLM supports is a
drop-in. The plumbing lives in two files:

- `backend/config.py` — env loading, `PROVIDER_KEYS`, `provider_api_base()`,
  `provider_api_key()` resolvers.
- `backend/model_router.py` — `llm_for(tier)` factory that builds a
  `LiteLlm` with the right `api_base` and auth header for the tier's
  vendor.

### Three knobs control the entire provider layer

| Knob | Where it lives | Purpose |
|---|---|---|
| **Model tier** | `MODEL_TIERS` in `backend/config.py`, overridden via `MODEL_READER` / `MODEL_ANALYST` | Which model each agent uses (LiteLLM-compatible string). |
| **API key** | Vendor's canonical env var (`OPENAI_API_KEY`, `GROQ_API_KEY`, …) OR `PROVIDER_API_KEY_<VENDOR>` override | The credential. Resolution order below. |
| **API base** | `PROVIDER_API_BASE_<VENDOR>` (optional) | Custom endpoint for self-hosted or proxied requests. |

### API-key resolution order

`provider_api_key(model_string)` in `backend/config.py` checks, in order:

1. **`PROVIDER_API_KEY_<VENDOR>`** — explicit override. Use this when:
   - You proxy a known vendor through your own endpoint with a different key.
   - You use a vendor LiteLLM doesn't know about (e.g. `myproxy/llama-3`).
2. **Vendor's canonical env var** — what LiteLLM auto-picks (e.g. `OPENAI_API_KEY`, `GROQ_API_KEY`).
3. **Nothing → `key=None`** — caller decides whether to send no auth header (local Ollama with auth off).

The "vendor" is the prefix in the model string (everything before the first
`/`), uppercased.

### Header style

| Vendor style | Header sent |
|---|---|
| OpenAI-compatible (OpenAI, Groq, Together, Fireworks, Mistral, DeepSeek, xAI, Perplexity, Cohere, custom proxies, Ollama when a key is set) | `Authorization: Bearer <key>` |
| Azure | `api-key: <key>` |
| Local server, no key anywhere | **No auth header at all** (not even an empty `Bearer`) |

### `llm_for()` injection matrix

| Scenario | `api_base` set? | Key present? | What `llm_for()` does |
|---|---|---|---|
| Default hosted vendor (e.g. `openai/gpt-4o-mini` + `OPENAI_API_KEY`) | no | canonical env var | Returns `LiteLlm(model=...)`. LiteLLM handles auth. |
| Custom `api_base`, canonical key in env | yes | canonical env var | Returns `LiteLlm(model=..., api_base=..., extra_headers={'Authorization': 'Bearer ...'})`. |
| Override key in use | yes/no | `PROVIDER_API_KEY_<VENDOR>` | Same as above but the override key is injected; canonical env var is ignored. |
| `api_base` set, **no** key anywhere | yes | none | Returns `LiteLlm(model=..., api_base=...)` with **no** auth header. |
| Azure | yes | `AZURE_API_KEY` or `PROVIDER_API_KEY_AZURE` | Injects `api-key` header instead of `Authorization`. |

### Adding a new LiteLLM-supported vendor (recipe)

1. Add the vendor's canonical env-var name to the `PROVIDER_KEYS` tuple in
   `backend/config.py`.
2. If the vendor is unknown to LiteLLM or uses a different header, also
   extend `_LITELLM_DEFAULT_KEY_ENV` (auth resolver lookup) — leave it
   out and the user can always supply `PROVIDER_API_KEY_<VENDOR>`.
3. Set the model string in `MODEL_READER` / `MODEL_ANALYST`. Done.

### Adding an OpenAI-compatible custom vendor (no LiteLLM support needed)

```bash
# 1. Point the model at a custom vendor name
MODEL_READER=acme/llama-3
# 2. Tell us where it lives
PROVIDER_API_BASE_ACME=https://llm.acme.internal/v1
# 3. Tell us what key to use
PROVIDER_API_KEY_ACME=sk-acme-...
```

No code changes. `llm_for("reader")` will send
`Authorization: Bearer sk-acme-...` to `https://llm.acme.internal/v1`.

### Auth behavior cheat-sheet

| Scenario | What `llm_for()` injects |
|---|---|
| Vendor with default key (e.g. `groq/...` + `GROQ_API_KEY`) | LiteLLM auto-picks; no manual header |
| `PROVIDER_API_BASE_<VENDOR>` set + vendor's canonical key present | Sends `Authorization: Bearer <key>` (or `api-key: <key>` for Azure) |
| `PROVIDER_API_KEY_<VENDOR>` set | Sends header using that override key |
| `api_base` set, **no** key anywhere | **No auth header sent** (local Ollama / LM Studio with auth off) — no bogus empty `Bearer ` |
| Azure | Always uses `api-key: ...` header, never `Authorization` |

### Worked examples

```bash
# OpenAI direct
OPENAI_API_KEY=sk-...
MODEL_READER=openai/gpt-4o-mini

# Groq (key alone is enough — Groq uses its own endpoint)
GROQ_API_KEY=gsk_...
MODEL_READER=groq/llama-3.1-8b-instant

# Ollama running locally (needs api_base because LiteLLM doesn't know it)
MODEL_READER=ollama/llama3.1
PROVIDER_API_BASE_OLLAMA=http://localhost:11434

# OpenAI-compatible local server via the openai/ vendor
MODEL_READER=openai/llama-3.1-8b
PROVIDER_API_BASE_OPENAI=http://localhost:11434/v1

# Azure OpenAI
AZURE_API_KEY=...
AZURE_API_BASE=https://YOUR-RESOURCE.openai.azure.com
AZURE_API_VERSION=2024-08-01-preview
MODEL_READER=azure/YOUR-DEPLOYMENT-NAME

# Custom vendor LiteLLM has never heard of
MODEL_READER=acme/llama-3
PROVIDER_API_BASE_ACME=https://llm.acme.internal/v1
PROVIDER_API_KEY_ACME=sk-acme-...

# Local Ollama with auth disabled (no Authorization header sent)
MODEL_READER=ollama/llama3.1
PROVIDER_API_BASE_OLLAMA=http://localhost:11434
# (no key vars at all)
```

No code changes are needed to switch providers — just edit `.env` and
restart. Unknown tiers fail fast (`KeyError`) per the tutorial's
`llm_for()` design.

### Provider troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `401 Unauthorized` | Wrong key, or LiteLLM picked up a different env var than you expected | Set `PROVIDER_API_KEY_<VENDOR>` explicitly to override |
| `404 Not Found` on a hosted vendor | Model string wrong — LiteLLM doesn't know about that model on that vendor | Check the vendor's docs for the exact model id; `echo $MODEL_READER` |
| `ConnectError` to `api.openai.com` when you meant a local server | You forgot `PROVIDER_API_BASE_<VENDOR>` | Add it; e.g. `PROVIDER_API_BASE_OPENAI=http://localhost:11434/v1` |
| Empty `Authorization: Bearer ` header → server rejects | Local server has strict auth, you accidentally set `OPENAI_API_KEY=""` | Unset the env var, or set `PROVIDER_API_KEY_<VENDOR>` to a dummy value |
| `ProviderAuth` returns wrong key | Both canonical and override are set; we honor the override (intended) | Unset one of them |
| `KeyError` from `llm_for("foo")` | Unknown tier name | Add `"foo"` to `MODEL_TIERS` in `backend/config.py`, or fix the name |

---

## The `docs/` contract

Place any of the following into `docs/` (subfolders are walked recursively;
output is sorted for determinism):

- **Plain text** (UTF-8, latin-1 fallback): `.txt .md .rst .json .csv .tsv
  .html .htm .xml .py .js .ts .yml .yaml .ini .cfg .toml .log`
- **PDF**: `.pdf` (requires `pdfplumber`)
- **DOCX**: `.docx` (requires `python-docx`)

Each file is wrapped with `=== FILE: <path> ===` / `=== END FILE ===`
headers so the analyst can see boundaries. Per-file text is hard-truncated
to `MAX_CHARS_PER_FILE`, the whole corpus to `MAX_CORPUS_CHARS` — both are
env-configurable. If a file fails to parse (malformed PDF, encrypted DOCX,
etc.), the tool records the error inline and continues with the rest.

If `docs/` is empty, Agent 1 stores `[no documents found in <DOCS_DIR>]`
under `corpus` and the analyst continues gracefully.

---

## Exa cap behavior

Both layers enforce `MAX_EXA_CALLS` (default `3`) — the agent's instruction
explicitly says "at most N calls" **and** the `exa_search` tool itself
short-circuits to `{"status": "limit_reached", "remaining": 0}` once the
process-wide counter hits zero. The agent cannot reset the counter.

Per-call safety (see tutorial §"Common Pitfalls / Context overflow"):

- `numResults` clamped to `3`
- `text.maxCharacters` capped at `1500` per result

---

## Tools

The pipeline ships with three tool modules. Each follows the same recipe:
an LLM-friendly Python function + a process-wide counter + hard-coded
caps that the agent cannot bypass.

### `backend/tools/file_reader.py` — `read_all_docs()`

Walks `DOCS_DIR`, parses text/PDF/DOCX, returns a single concatenated
string with `=== FILE: <path> ===` headers. Per-file truncation
(`MAX_CHARS_PER_FILE`) and total-corpus truncation (`MAX_CORPUS_CHARS`)
are enforced in code.

### `backend/tools/web_search.py` — `exa_search(query)`

Calls the Exa `/search` endpoint. Counter is bumped *after* a successful
HTTP call; failed calls do **not** consume budget so the agent can retry.

### `backend/tools/agentmail_tool.py` — AgentMail SDK wrappers

Six thin wrappers over the [`agentmail`](https://github.com/agentmail-to/agentmail-python)
Python SDK. They are **opt-in**: if `AGENTMAIL_API_KEY` is unset, the
client is built lazily and raises only on first call (so importing
`agent.py` still succeeds when email isn't needed).

| Tool | Purpose | Key args |
|---|---|---|
| `create_inbox` | Provision a new AgentMail inbox | `username`, `domain`, `display_name` (all optional) |
| `list_inboxes` | List account inboxes | `limit` (clamped to `MAX_MESSAGES_PER_LIST`) |
| `send_message` | Send an email | `inbox_id`, `to`, `subject`, `text` or `html` |
| `list_messages` | Recent messages in an inbox | `inbox_id`, `limit` (clamped + body-truncated) |
| `get_message` | Fetch a single message | `inbox_id`, `message_id` (body-truncated) |
| `delete_inbox` | Permanently delete an inbox | `inbox_id` |

Caps in code (not just in the prompt):

- **Global budget** — every tool call checks `MAX_AGENTMAIL_CALLS` first;
  on exhaustion it returns `"[agentmail] Call budget exhausted ..."`
  without touching the network.
- **List clamping** — `limit` is always clamped to `MAX_MESSAGES_PER_LIST`.
- **Body truncation** — `text`/`html`/`body`/`preview` fields in returned
  messages are truncated to `MAX_MESSAGE_PREVIEW_CHARS` to keep token
  usage predictable.
- **Counter is process-wide** — same shape as Exa. Reset between runs
  via `agentmail_tool.reset_call_count()` (used by tests).

To attach these tools to an agent, import them and add to the
`tools=[...]` list:

```python
from backend.tools.agentmail_tool import (
    create_inbox, list_inboxes, send_message,
    list_messages, get_message, delete_inbox,
)

my_agent = LlmAgent(
    name="inbox_manager",
    model="openai/gpt-4o-mini",
    instruction="... at most MAX_AGENTMAIL_CALLS=10 calls total ...",
    tools=[create_inbox, list_inboxes, send_message,
           list_messages, get_message, delete_inbox],
)
```

---

## Running the pipeline

### Interactive (`adk run`)

```bash
source venv/bin/activate
adk run .
```

Then type a prompt like:

> Compare the policies in docs/policy-a.md and docs/policy-b.md and tell me
> which is more user-friendly.

Type `exit` to quit.

### Web UI

```bash
adk web .
```

### Single-shot from code

```python
import asyncio
from agent import root_agent
from backend.runner import run_agent_once

print(asyncio.run(
    run_agent_once(
        root_agent,
        "Summarize the three most important findings across all documents.",
    )
))
```

### Verify the agent loads

```bash
python -c "from agent import root_agent; print([a.name for a in root_agent.sub_agents])"
# → ['file_reader_agent', 'analyst_agent']
```

---

## Notes

- Each agent's instruction explicitly forbids asking the user questions — a
  `SequentialAgent` cannot route a turn back from Agent 2 to Agent 1, so
  every stage must decide and act autonomously (see tutorial §"Common
  Pitfalls").
- Agent 1 stores the corpus **verbatim** under `output_key="corpus"`; it
  does not summarize. Summarization is Agent 2's job.
- Agent 2 has no `output_schema` — the user prompt dictates the deliverable
  shape (summary, brief, checklist, comparison, etc.), so free-form
  markdown is the most flexible contract.
- The Exa counter is process-wide. If you re-run the pipeline from the same
  Python process, the limit persists. Start a fresh process to reset.

---

## Tests

A small pytest suite covers the auth resolver and the AgentMail tool
without any network calls:

```bash
.venv/bin/pytest tests/ -v
# → 22 passed
```

The `conftest.py` at the repo root adds the project root to `sys.path` so
`from backend import …` works inside tests regardless of where pytest is
invoked from.

Tests cover:

- Override wins over canonical env var.
- Canonical env var used when no override is set.
- Unknown vendor (not in LiteLLM's map) wired via override.
- No-key local server → `key=None` and no `Authorization` header sent.
- Azure uses `api-key` header instead of `Authorization: Bearer`.
- `llm_for()` end-to-end: mock `LiteLlm` and assert the right kwargs.
- AgentMail: budget enforcement, list clamping, body truncation,
  lazy client init, missing API key, SDK exception swallowed into a
  string, counter reset.

All AgentMail tests mock `_get_client` so they run without a real API
key or even the `agentmail` package installed (the SDK is imported
lazily inside `_get_client`).

---

## Project hygiene

### `.gitignore`

Grouped into commented sections (Python, venvs, test/cache, secrets, logs,
editor junk, ADK runtime, Puku runtime, distribution, misc):

- **Always ignored**: `.env`, `.env.*` (except `.env.example`),
  `__pycache__/`, `.venv/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`,
  `*.pyc`, build artifacts.
- **ADK runtime**: `.adk/` (the SQLite session DB created by `adk run`).
- **Puku runtime**: `.puku/` (the AI-runtime's plans and embeddings DB).
- **Editor**: `.vscode/`, `.idea/`, `*.swp`, `.DS_Store`, `Thumbs.db`.

To verify what would be committed:

```bash
git status --ignored    # see both tracked + ignored
git ls-files --others --exclude-standard   # see only untracked non-ignored
```

### Tracked vs ignored (at a glance)

| Path | Tracked? | Why |
|---|---|---|
| `README.md`, `agent.py`, `backend/**`, `docs/**`, `tests/**`, `requirements.txt`, `conftest.py`, `.env.example`, `ADK_MULTI_AGENT_TUTORIAL.md` | yes | Project source |
| `.env` | **no** | Real secrets |
| `.venv/`, `venv/`, `__pycache__/`, `.pytest_cache/` | **no** | Generated |
| `.adk/` | **no** | ADK session DB |
| `.puku/` | **no** | AI-runtime bookkeeping |

---

## Using this repo as a template

If you're cloning this for a new agent project, the recommended workflow is:

1. **Copy the layout verbatim** — `agent.py`, `backend/`, `docs/`, `tests/`,
   `conftest.py`, `.env.example`, `requirements.txt`, `.gitignore`.
2. **Decide your tiers first** — pick `reader` / `analyst` (or whatever
   roles you need) and their model strings in `.env`. The
   `provider_api_key` resolver means you can swap providers per tier
   without touching code.
3. **Keep secrets out of git** — `.env.example` is the only env file that
   should ever be tracked. `.env` is in `.gitignore`.
4. **Two layers of cost / loop control** — every external tool should
   have both (a) an LLM-visible instruction cap and (b) a hard runtime
   counter inside the tool itself. `exa_search` is the reference
   implementation; copy its pattern for new tools. `agentmail_tool.py`
   is the second worked example (with list clamping + body truncation
   on top of the global budget).
5. **Free-form markdown outputs** when the deliverable shape is dictated
   by the user's prompt (analyst / writer / brief). Use
   `output_schema=Pydantic` only when the downstream consumer is another
   service that needs a structured contract (see tutorial §"Use Case 1").
6. **Run `pytest tests/` before pushing** — the auth suite catches
   provider config regressions cheaply.