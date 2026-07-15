"""AgentMail Python SDK tool wrappers.

Why this module exists
----------------------
The AgentMail SDK ships a `Client` class. We expose six small, LLM-friendly
tool functions on top of it so an agent can call them as plain Python
functions from its `tools=[...]` list.

Design rules (mirrors `web_search.py`)
-------------------------------------
1. **Hard caps live in code, not just the prompt.** Every tool that hits the
   network enforces the relevant cap from `backend.config` and refuses to
   do more work when it's exceeded. The model's instruction text *also*
   says to stop — defense in depth.
2. **Module-level counter**, so multiple tool calls in the same process
   share the budget (matches `exa_search`).
3. **No exceptions leak raw stack traces.** Tool functions return strings
   that are safe to hand back to an LLM.
4. **Lazy client construction.** We don't hit the network at import time;
   the client is built on first use so unit tests can patch it.
"""

from __future__ import annotations

import json
import threading
from typing import Any

from ..config import (
    AGENTMAIL_API_KEY,
    MAX_AGENTMAIL_CALLS,
    MAX_MESSAGES_PER_LIST,
    MAX_MESSAGE_PREVIEW_CHARS,
)

# --- Module-level rate limiter --------------------------------------------
# Mirrors backend/tools/web_search.py so both tools share the same idiom.
_call_lock = threading.Lock()
_call_count: int = 0


def _bump_call_count() -> int:
    """Increment and return the new call count. Thread-safe."""
    global _call_count
    with _call_lock:
        _call_count += 1
        return _call_count


def reset_call_count() -> None:
    """Reset the counter. Useful between test runs or pipeline runs."""
    global _call_count
    with _call_lock:
        _call_count = 0


def get_call_count() -> int:
    """Read the current call count (mostly for tests + diagnostics)."""
    with _call_lock:
        return _call_count


def _check_budget() -> str | None:
    """Return an error string if the global call budget is exhausted.

    Returns `None` if it's still safe to proceed.
    """
    with _call_lock:
        if _call_count >= MAX_AGENTMAIL_CALLS:
            return (
                f"[agentmail] Call budget exhausted "
                f"({_call_count}/{MAX_AGENTMAIL_CALLS}). "
                "Stop using AgentMail tools and answer with what you already have."
            )
    return None


def _truncate(text: str | None, limit: int = MAX_MESSAGE_PREVIEW_CHARS) -> str:
    """Truncate a string and append a marker if it was cut.

    Returns an empty string for `None` so tool outputs are always strings.
    """
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


# --- Lazy SDK client ------------------------------------------------------
_client_lock = threading.Lock()
_client: Any | None = None


def _get_client() -> Any:
    """Build or return a cached `AgentMail` client.

    Imported lazily so that:
      * importing this module never hits the network,
      * unit tests can monkey-patch `agentmail_tool._get_client` to return
        a fake client,
      * `AGENTMAIL_API_KEY` missing only fails at call time, not import time.
    """
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            if not AGENTMAIL_API_KEY:
                raise RuntimeError(
                    "AGENTMAIL_API_KEY is not set. "
                    "Add it to your .env (see .env.example)."
                )
            # Imported here so a missing package only fails when the tool
            # is actually used, not when the module loads.
            from agentmail import AgentMail

            _client = AgentMail(api_key=AGENTMAIL_API_KEY)
    return _client


def _to_json(obj: Any) -> str:
    """Serialize an SDK response (model, dict, list) to a JSON string.

    SDK methods return Pydantic models. `model_dump()` is the standard
    converter; we fall back to `__dict__` and `str()` if the object
    doesn't expose it (e.g. in tests with a mock).
    """
    if obj is None:
        return ""
    if hasattr(obj, "model_dump"):
        try:
            return json.dumps(obj.model_dump(), default=str, indent=2)
        except Exception:  # pragma: no cover - defensive
            pass
    if isinstance(obj, (dict, list)):
        return json.dumps(obj, default=str, indent=2)
    if hasattr(obj, "__dict__"):
        return json.dumps(obj.__dict__, default=str, indent=2)
    return str(obj)


# --- Tool functions exposed to the LLM -----------------------------------

def create_inbox(
    username: str | None = None,
    domain: str | None = None,
    display_name: str | None = None,
) -> str:
    """Create a new AgentMail inbox.

    Parameters
    ----------
    username : str | None
        Optional username (without `@domain`). The SDK (or AgentMail
        service) will reject invalid characters.
    domain : str | None
        Optional domain. If both `username` and `domain` are omitted, the
        service picks one for you.
    display_name : str | None
        Friendly label shown to recipients.

    Returns
    -------
    str
        JSON string with the created inbox (id, address, etc.) or a
        short error message that is safe to feed back to the LLM.
    """
    err = _check_budget()
    if err:
        return err

    kwargs: dict[str, Any] = {}
    if username:
        kwargs["username"] = username
    if domain:
        kwargs["domain"] = domain
    if display_name:
        kwargs["display_name"] = display_name

    try:
        client = _get_client()
        # SDK exposes: client.inboxes.create(...)
        inbox = client.inboxes.create(**kwargs)
    except Exception as e:
        return f"[agentmail] create_inbox failed: {e}"

    _bump_call_count()
    return _to_json(inbox)


def list_inboxes(limit: int = 25) -> str:
    """List inboxes in this AgentMail account.

    `limit` is clamped to `MAX_MESSAGES_PER_LIST` so a runaway agent
    cannot request thousands of pages.
    """
    err = _check_budget()
    if err:
        return err

    safe_limit = max(1, min(int(limit), MAX_MESSAGES_PER_LIST))
    try:
        client = _get_client()
        # SDK exposes: client.inboxes.list(limit=...)
        page = client.inboxes.list(limit=safe_limit)
    except Exception as e:
        return f"[agentmail] list_inboxes failed: {e}"

    _bump_call_count()

    # `page` is usually a paginated response with `.inboxes` and
    # `.count`. We try to be tolerant of either shape.
    if hasattr(page, "model_dump"):
        return _to_json(page)
    inboxes = getattr(page, "inboxes", page)
    return _to_json(inboxes)


def send_message(
    inbox_id: str,
    to: str,
    subject: str,
    text: str | None = None,
    html: str | None = None,
) -> str:
    """Send an email from `inbox_id` to `to`.

    Either `text` or `html` must be provided.
    """
    err = _check_budget()
    if err:
        return err

    if not (text or html):
        return "[agentmail] send_message requires text or html body."
    if not inbox_id or not to or not subject:
        return "[agentmail] send_message requires inbox_id, to, and subject."

    try:
        client = _get_client()
        # SDK exposes: client.inboxes.messages.send(inbox_id=..., ...)
        result = client.inboxes.messages.send(
            inbox_id=inbox_id,
            to=to,
            subject=subject,
            text=text,
            html=html,
        )
    except Exception as e:
        return f"[agentmail] send_message failed: {e}"

    _bump_call_count()
    return _to_json(result)


def list_messages(
    inbox_id: str,
    limit: int = 10,
) -> str:
    """List recent messages in `inbox_id`.

    `limit` is clamped to `MAX_MESSAGES_PER_LIST`. Returned message bodies
    are truncated to `MAX_MESSAGE_PREVIEW_CHARS` to keep token usage sane.
    """
    err = _check_budget()
    if err:
        return err
    if not inbox_id:
        return "[agentmail] list_messages requires inbox_id."

    safe_limit = max(1, min(int(limit), MAX_MESSAGES_PER_LIST))
    try:
        client = _get_client()
        page = client.inboxes.messages.list(inbox_id=inbox_id, limit=safe_limit)
    except Exception as e:
        return f"[agentmail] list_messages failed: {e}"

    _bump_call_count()

    # Best-effort truncation of any nested body fields.
    messages = getattr(page, "messages", page)
    if hasattr(messages, "__iter__"):
        truncated: list[Any] = []
        for m in messages:
            if hasattr(m, "model_dump"):
                dump = m.model_dump()
            elif isinstance(m, dict):
                dump = m
            else:
                truncated.append(m)
                continue
            for k in ("text", "html", "body", "preview"):
                if k in dump and isinstance(dump[k], str):
                    dump[k] = _truncate(dump[k])
            truncated.append(dump)
        return _to_json(truncated)
    return _to_json(page)


def get_message(inbox_id: str, message_id: str) -> str:
    """Fetch a single message by id.

    Body fields are truncated to `MAX_MESSAGE_PREVIEW_CHARS`.
    """
    err = _check_budget()
    if err:
        return err
    if not inbox_id or not message_id:
        return "[agentmail] get_message requires inbox_id and message_id."

    try:
        client = _get_client()
        msg = client.inboxes.messages.get(inbox_id=inbox_id, message_id=message_id)
    except Exception as e:
        return f"[agentmail] get_message failed: {e}"

    _bump_call_count()

    if hasattr(msg, "model_dump"):
        dump = msg.model_dump()
        for k in ("text", "html", "body", "preview"):
            if k in dump and isinstance(dump[k], str):
                dump[k] = _truncate(dump[k])
        return _to_json(dump)
    return _to_json(msg)


def delete_inbox(inbox_id: str) -> str:
    """Delete an inbox permanently. Use with care."""
    err = _check_budget()
    if err:
        return err
    if not inbox_id:
        return "[agentmail] delete_inbox requires inbox_id."

    try:
        client = _get_client()
        client.inboxes.delete(inbox_id=inbox_id)
    except Exception as e:
        return f"[agentmail] delete_inbox failed: {e}"

    _bump_call_count()
    return f"[agentmail] inbox {inbox_id} deleted."


# Public re-exports for tests + integration.
__all__ = [
    "create_inbox",
    "list_inboxes",
    "send_message",
    "list_messages",
    "get_message",
    "delete_inbox",
    "reset_call_count",
    "get_call_count",
]