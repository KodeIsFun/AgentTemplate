"""LiteLLM model factory — same shape as the tutorial's `llm_for()` helper,
extended to honor an optional `api_base` and explicit `api_key` per provider
so any OpenAI-compatible endpoint can be wired without code changes.

Resolution rules (see `backend.config.provider_api_key`):

  - If `PROVIDER_API_BASE_<VENDOR>` is set, the request is sent to that URL.
  - If `PROVIDER_API_KEY_<VENDOR>` is set, that key is used (overrides the
    vendor's canonical env var).
  - Otherwise LiteLLM auto-picks the vendor's canonical env var
    (`OPENAI_API_KEY`, `GROQ_API_KEY`, …).
  - If neither is set (e.g. a local Ollama server with auth disabled), no
    `Authorization` header is sent at all, instead of an empty `Bearer `.
  - Azure uses the `api-key` header; every other OpenAI-compatible endpoint
    uses `Authorization: Bearer ...`.
"""

from __future__ import annotations

from google.adk.models.lite_llm import LiteLlm

from backend import config


def _auth_header_value(auth: config.ProviderAuth) -> str | None:
    """Return the header value to inject, or None to send no auth at all."""
    if not auth.key:
        return None
    if auth.header_name == "api-key":
        return auth.key
    # OpenAI-compatible (and OpenAI itself) → "Bearer <key>".
    return f"Bearer {auth.key}"


def llm_for(tier: str) -> LiteLlm:
    """Return a `LiteLlm` configured for the given tier. KeyError = fail fast.

    Args:
        tier: Key into `config.MODEL_TIERS`. Examples: "reader", "analyst".
    """
    model_string = config.MODEL_TIERS[tier]
    api_base = config.provider_api_base(model_string)
    auth = config.provider_api_key(model_string)
    header_value = _auth_header_value(auth)

    kwargs: dict = {"model": model_string}

    # Only override api_base when the user has explicitly pointed us at a
    # custom endpoint — otherwise LiteLLM uses the vendor's default.
    if api_base:
        kwargs["api_base"] = api_base

    # If the user explicitly set a key (via PROVIDER_API_KEY_<VENDOR>) or has
    # no env var at all (local server), we have to inject the header
    # ourselves. LiteLLM auto-picks canonical env vars on its own, so we
    # don't need to duplicate that work.
    if api_base or auth.env_var and auth.env_var.startswith("PROVIDER_API_KEY_"):
        if header_value is not None:
            kwargs["extra_headers"] = {auth.header_name: header_value}
        # If header_value is None and we have an api_base, we deliberately
        # send no auth header (local server with auth off).

    return LiteLlm(**kwargs)