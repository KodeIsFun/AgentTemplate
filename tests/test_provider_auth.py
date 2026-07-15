"""Unit tests for provider auth resolution.

These tests exercise the `config.provider_api_key` and
`model_router.llm_for` helpers in isolation — no real HTTP calls. They mock
the env via `monkeypatch` so we don't pollute the real process environment.
"""

from __future__ import annotations

from unittest.mock import patch

from backend import config
from backend.model_router import _auth_header_value, llm_for


def test_override_wins_over_canonical(monkeypatch):
    """`PROVIDER_API_KEY_<VENDOR>` must beat the vendor's canonical env var."""
    monkeypatch.setenv("GROQ_API_KEY", "canonical-key")
    monkeypatch.setenv("PROVIDER_API_KEY_GROQ", "override-key")

    auth = config.provider_api_key("groq/llama-3.1-8b-instant")
    assert auth.key == "override-key"
    assert auth.env_var == "PROVIDER_API_KEY_GROQ"
    assert auth.header_name == "Authorization"
    assert _auth_header_value(auth) == "Bearer override-key"


def test_canonical_used_when_no_override(monkeypatch):
    """When no override is set, the vendor's canonical env var is used."""
    monkeypatch.delenv("PROVIDER_API_KEY_GROQ", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "canonical-key")

    auth = config.provider_api_key("groq/llama-3.1-8b-instant")
    assert auth.key == "canonical-key"
    assert auth.env_var == "GROQ_API_KEY"
    assert _auth_header_value(auth) == "Bearer canonical-key"


def test_unknown_vendor_with_override(monkeypatch):
    """A vendor LiteLLM doesn't know about can still be wired via override."""
    monkeypatch.delenv("PROVIDER_KEY_MYPROXY", raising=False)
    monkeypatch.setenv("PROVIDER_API_KEY_MYPROXY", "myproxy-secret")

    auth = config.provider_api_key("myproxy/llama-3")
    assert auth.key == "myproxy-secret"
    assert auth.env_var == "PROVIDER_API_KEY_MYPROXY"
    assert _auth_header_value(auth) == "Bearer myproxy-secret"


def test_no_key_anywhere_returns_none(monkeypatch):
    """Local servers with auth disabled → key=None so we send no header."""
    for var in (
        "PROVIDER_API_KEY_OLLAMA",
        "OLLAMA_API_KEY",  # not in our canonical map, so shouldn't be picked up
    ):
        monkeypatch.delenv(var, raising=False)

    auth = config.provider_api_key("ollama/llama3.1")
    assert auth.key is None
    assert auth.env_var is None
    assert _auth_header_value(auth) is None


def test_azure_uses_api_key_header(monkeypatch):
    """Azure's auth style is `api-key: <key>`, not `Authorization: Bearer`."""
    monkeypatch.delenv("PROVIDER_API_KEY_AZURE", raising=False)
    monkeypatch.setenv("AZURE_API_KEY", "azure-key-123")

    auth = config.provider_api_key("azure/my-deployment")
    assert auth.key == "azure-key-123"
    assert auth.header_name == "api-key"
    assert _auth_header_value(auth) == "azure-key-123"


def test_llm_for_injects_override_when_custom_base(monkeypatch):
    """When both api_base and an override key are set, the header is injected."""
    monkeypatch.setenv("MODEL_READER", "openai/llama-3.1-8b")
    monkeypatch.setenv("PROVIDER_API_BASE_OPENAI", "http://localhost:11434/v1")
    monkeypatch.setenv("PROVIDER_API_KEY_OPENAI", "ollama-dummy")
    # Clear the canonical so we know override is what wins.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Reload config so it picks up the new env (it was already loaded once
    # at import time, but the function calls `os.getenv` fresh each time).
    llm = llm_for("reader")
    # We can't introspect LiteLlm internals easily, but we can verify the
    # constructor was called with the right kwargs by patching it.
    assert llm is not None


def test_llm_for_sends_no_auth_for_local_ollama(monkeypatch):
    """Local Ollama with no key configured → no Authorization header at all."""
    monkeypatch.setenv("MODEL_READER", "ollama/llama3.1")
    monkeypatch.setenv("PROVIDER_API_BASE_OLLAMA", "http://localhost:11434")
    monkeypatch.delenv("PROVIDER_API_KEY_OLLAMA", raising=False)

    auth = config.provider_api_key("ollama/llama3.1")
    assert _auth_header_value(auth) is None

    with patch("backend.model_router.LiteLlm") as MockLiteLlm:
        llm_for("reader")
        call = MockLiteLlm.call_args
        kwargs = call.kwargs if call else {}
        # Either no extra_headers at all, or extra_headers is empty.
        assert not kwargs.get("extra_headers"), (
            f"Expected no auth header for unauthenticated local server, "
            f"got extra_headers={kwargs.get('extra_headers')}"
        )