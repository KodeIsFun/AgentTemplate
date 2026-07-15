"""Environment loading and central configuration constants.

Reuses the same patterns as the tutorial's §"Installation & Environment" and
§"Using Non-Gemini Models (LiteLLM / OpenRouter)" sections.

Provider credentials are loaded generically so any LiteLLM-supported /
OpenAI-compatible provider can be added by dropping one env block into
`.env`. The agent code never has to change.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- API keys (kept as module attrs for backward compatibility) -----------
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
EXA_API_KEY: str = os.getenv("EXA_API_KEY", "")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
AGENTMAIL_API_KEY: str = os.getenv("AGENTMAIL_API_KEY", "")

# --- Generic provider credentials ----------------------------------------
# Any key here is automatically picked up by LiteLLM via its standard env
# var conventions (OPENAI_API_KEY, GROQ_API_KEY, TOGETHERAI_API_KEY,
# FIREWORKS_AI_API_KEY, MISTRAL_API_KEY, ANTHROPIC_API_KEY, ...).
# Add more as needed — `model_router.llm_for()` will pick them up by name.
PROVIDER_KEYS: dict[str, str] = {
    name: os.getenv(name)
    for name in (
        # OpenAI direct
        "OPENAI_API_KEY",
        "OPENAI_ORGANIZATION",
        "OPENAI_PROJECT",
        # Anthropic direct
        "ANTHROPIC_API_KEY",
        # Groq
        "GROQ_API_KEY",
        # Together AI
        "TOGETHERAI_API_KEY",
        "TOGETHER_API_KEY",
        # Fireworks AI
        "FIREWORKS_AI_API_KEY",
        "FIREWORKSAI_API_KEY",
        "FIREWORKS_API_KEY",
        # Mistral
        "MISTRAL_API_KEY",
        # DeepSeek (OpenAI-compatible)
        "DEEPSEEK_API_KEY",
        # xAI / Grok (OpenAI-compatible)
        "XAI_API_KEY",
        # Perplexity (OpenAI-compatible)
        "PERPLEXITY_API_KEY",
        # Cohere (LiteLLM supports it via chat)
        "COHERE_API_KEY",
        # Hugging Face Inference (OpenAI-compatible router)
        "HF_API_KEY",
        "HUGGINGFACE_API_KEY",
        # Azure OpenAI
        "AZURE_API_KEY",
        "AZURE_API_BASE",
        "AZURE_API_VERSION",
    )
    if os.getenv(name)
}


def provider_api_base(model_string: str) -> str | None:
    """Return the optional `api_base` for an OpenAI-compatible endpoint.

    Reads `PROVIDER_API_BASE_<VENDOR>` where VENDOR is the LiteLLM prefix in
    the model string (the chunk before `/`). Lets you point a tier at a
    self-hosted or regional endpoint without touching code, e.g.:

        MODEL_READER=openai/llama-3.1-8b
        PROVIDER_API_BASE_OPENAI=http://localhost:11434/v1   # Ollama
    """
    if "/" not in model_string:
        return None
    vendor = model_string.split("/", 1)[0].upper()
    return os.getenv(f"PROVIDER_API_BASE_{vendor}")


# Map of LiteLLM vendor prefix → the env var LiteLLM expects for its API key.
# Used by `provider_api_key()` to look up the canonical key, and by callers
# to detect "no auth needed" (when neither this nor PROVIDER_API_KEY_* is set).
_LITELLM_DEFAULT_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "together_ai": "TOGETHERAI_API_KEY",
    "fireworks_ai": "FIREWORKS_AI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
    "cohere": "COHERE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "azure": "AZURE_API_KEY",
}


class ProviderAuth:
    """Resolved auth for a model string.

    - `key`: the literal API key to send (None if no auth is needed).
    - `header_name`: header name to use when injecting manually
      (e.g. Azure uses `api-key`, OpenAI-compatible servers use
      `Authorization: Bearer ...`).
    - `env_var`: which env var the key was read from, for diagnostics.
    """

    __slots__ = ("key", "header_name", "env_var")

    def __init__(self, key: str | None, header_name: str, env_var: str | None):
        self.key = key
        self.header_name = header_name
        self.env_var = env_var

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        redacted = "***" if self.key else "None"
        return f"ProviderAuth(key={redacted}, header={self.header_name!r}, env={self.env_var})"


def provider_api_key(model_string: str) -> ProviderAuth:
    """Resolve the API key (and header style) for a model string.

    Resolution order:

      1. `PROVIDER_API_KEY_<VENDOR>` — explicit override, wins over everything.
         Used for custom vendors or when you want to point a known vendor at
         a different key than its canonical env var.
      2. `<vendor's canonical env var>` — what LiteLLM would auto-pick
         (e.g. `OPENAI_API_KEY`, `GROQ_API_KEY`).
      3. If neither is set, returns `key=None` so the caller can decide
         whether to send no auth (typical for local servers like Ollama)
         instead of an empty `Bearer ` header.

    Header style:
      - Azure uses `api-key`.
      - Everything else uses `Authorization`.
    """
    if "/" not in model_string:
        vendor, _, _ = model_string.partition("/")
    else:
        vendor = model_string.split("/", 1)[0]

    upper = vendor.upper()
    is_azure = upper == "AZURE"
    header_name = "api-key" if is_azure else "Authorization"

    # 1. Explicit override (also covers custom / unknown LiteLLM vendors).
    override = os.getenv(f"PROVIDER_API_KEY_{upper}")
    if override:
        return ProviderAuth(key=override, header_name=header_name, env_var=f"PROVIDER_API_KEY_{upper}")

    # 2. Vendor's canonical env var.
    canonical = _LITELLM_DEFAULT_KEY_ENV.get(vendor.lower())
    if canonical and os.getenv(canonical):
        return ProviderAuth(key=os.getenv(canonical), header_name=header_name, env_var=canonical)

    # 3. Nothing configured → no auth.
    return ProviderAuth(key=None, header_name=header_name, env_var=None)


# --- Folder + corpus safety caps ------------------------------------------
DOCS_DIR: Path = Path(os.getenv("DOCS_DIR", "./docs")).resolve()
MAX_EXA_CALLS: int = int(os.getenv("MAX_EXA_CALLS", "3"))
MAX_CHARS_PER_FILE: int = int(os.getenv("MAX_CHARS_PER_FILE", "10000"))
MAX_CORPUS_CHARS: int = int(os.getenv("MAX_CORPUS_CHARS", "120000"))

# --- AgentMail safety caps -----------------------------------------------
# Hard guard inside each tool function (mirrored in agent instruction text).
MAX_AGENTMAIL_CALLS: int = int(os.getenv("MAX_AGENTMAIL_CALLS", "10"))
MAX_MESSAGES_PER_LIST: int = int(os.getenv("MAX_MESSAGES_PER_LIST", "20"))
MAX_MESSAGE_PREVIEW_CHARS: int = int(os.getenv("MAX_MESSAGE_PREVIEW_CHARS", "2000"))

# --- Model routing --------------------------------------------------------
# Each tier is a LiteLLM-compatible model string. Override via .env.
# Examples:
#   openrouter/anthropic/claude-3.5-sonnet
#   openai/gpt-4o-mini
#   groq/llama-3.1-70b-versatile
#   together_ai/meta-llama/Llama-3-70b-chat-hf
#   fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct
#   ollama/llama3.1                      (with PROVIDER_API_BASE_OLLAMA)
#   openai/llama-3.1-8b                  (with PROVIDER_API_BASE_OPENAI)
MODEL_TIERS: dict[str, str] = {
    "reader": os.getenv("MODEL_READER", "openrouter/openai/gpt-4o-mini"),
    "analyst": os.getenv("MODEL_ANALYST", "openrouter/anthropic/claude-3.5-sonnet"),
}


def docs_dir() -> Path:
    """Return the (resolved) documents directory, creating it if missing."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    return DOCS_DIR