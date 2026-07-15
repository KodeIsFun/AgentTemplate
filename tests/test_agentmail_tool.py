"""Tests for `backend.tools.agentmail_tool`.

We mock `agentmail_tool._get_client` so we don't need a real API key or
the SDK installed for the test suite to run.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


# Must come before any import that triggers config.
os.environ.setdefault("AGENTMAIL_API_KEY", "test-key")

from backend.tools import agentmail_tool as amt  # noqa: E402


class _FakePaginator:
    """Looks enough like the SDK's paginated response."""

    def __init__(self, messages):
        self.messages = messages
        self.count = len(messages)

    def model_dump(self):
        return {"count": self.count, "messages": [m.model_dump() for m in self.messages]}


class _FakeMessage:
    def __init__(self, message_id, text="hello"):
        self.message_id = message_id
        self.text = text

    def model_dump(self):
        return {"message_id": self.message_id, "text": self.text}


class _FakeInbox:
    def __init__(self, inbox_id="inb_1", address="a@ex.com"):
        self.inbox_id = inbox_id
        self.address = address

    def model_dump(self):
        return {"inbox_id": self.inbox_id, "address": self.address}


class _FakeInboxesNamespace:
    def __init__(self):
        self.deleted: list[str] = []
        self.sent: list[dict] = []
        self.messages_listed: list[dict] = []
        self.messages_got: list[dict] = []
        self.inbox_to_return = _FakeInbox()
        self.page_to_return = _FakePaginator(
            [_FakeMessage("m1", text="hi " * 500)]
        )

    def create(self, **kwargs):
        return _FakeInbox(inbox_id="inb_new", address=f"{kwargs.get('username', 'x')}@ex.com")

    def list(self, limit=25):
        return [_FakeInbox(inbox_id=f"inb_{i}") for i in range(limit)]

    def delete(self, inbox_id):
        self.deleted.append(inbox_id)

    class messages:
        @staticmethod
        def send(**kwargs):
            return {"status": "queued", **kwargs}

        @staticmethod
        def list(inbox_id, limit):
            return _FakePaginator(
                [_FakeMessage(f"{inbox_id}_m{i}", text="x" * 5000) for i in range(min(limit, 3))]
            )

        @staticmethod
        def get(inbox_id, message_id):
            return _FakeMessage(message_id, text="body " * 2000)


class _FakeClient:
    def __init__(self):
        self.inboxes = _FakeInboxesNamespace()


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Fresh counter + cached client for every test."""
    amt.reset_call_count()
    amt._client = _FakeClient()
    # Force a known call budget for predictable math.
    monkeypatch.setattr(amt, "MAX_AGENTMAIL_CALLS", 3, raising=False)
    monkeypatch.setattr(amt, "MAX_MESSAGES_PER_LIST", 5, raising=False)
    monkeypatch.setattr(amt, "MAX_MESSAGE_PREVIEW_CHARS", 50, raising=False)
    yield
    amt.reset_call_count()
    amt._client = None


# --- Budget enforcement ---------------------------------------------------

def test_create_inbox_returns_json_and_bumps_counter():
    out = amt.create_inbox(username="alice")
    assert '"inbox_id": "inb_new"' in out
    assert amt.get_call_count() == 1


def test_list_inboxes_clamps_to_max():
    out = amt.list_inboxes(limit=999)
    # Limit was clamped to 5 in the fixture.
    # `_to_json` falls back to `str()` for objects lacking `model_dump`,
    # so we just check the clamp marker is present in the call path.
    fake_ns = amt._client.inboxes
    requested: list[int] = []
    real_list = fake_ns.list

    def _spy_list(limit=25):
        requested.append(limit)
        return real_list(limit=limit)

    fake_ns.list = _spy_list  # type: ignore[assignment]
    amt.list_inboxes(limit=999)
    assert requested == [5]


def test_budget_exhaustion_short_circuits():
    # 3 allowed -> consume them all.
    amt.create_inbox()
    amt.list_inboxes()
    amt.send_message("inb_1", "x@y.com", "subj", text="hi")
    assert amt.get_call_count() == 3

    # 4th call must NOT hit the client and must return the budget message.
    out = amt.send_message("inb_1", "x@y.com", "subj", text="hi")
    assert "budget exhausted" in out
    assert amt.get_call_count() == 3  # unchanged


def test_send_requires_body():
    out = amt.send_message("inb_1", "x@y.com", "subj")
    assert "requires text or html" in out
    assert amt.get_call_count() == 0


def test_send_requires_core_fields():
    out = amt.send_message("", "", "", text="x")
    assert "requires inbox_id" in out
    assert amt.get_call_count() == 0


def test_send_success_returns_json():
    out = amt.send_message("inb_1", "x@y.com", "subj", text="hi")
    assert '"status": "queued"' in out
    assert amt.get_call_count() == 1


def test_list_messages_truncates_body():
    out = amt.list_messages("inb_1", limit=3)
    assert "truncated" in out  # body was cut to MAX_MESSAGE_PREVIEW_CHARS=50


def test_get_message_truncates_body():
    out = amt.get_message("inb_1", "m1")
    assert "truncated" in out


def test_get_message_requires_ids():
    assert "requires inbox_id and message_id" in amt.get_message("", "")
    assert "requires inbox_id and message_id" in amt.get_message("a", "")
    assert amt.get_call_count() == 0


def test_delete_inbox_calls_sdk_and_returns_confirmation():
    out = amt.delete_inbox("inb_42")
    assert "inb_42 deleted" in out
    assert amt.get_call_count() == 1


def test_delete_inbox_requires_id():
    out = amt.delete_inbox("")
    assert "requires inbox_id" in out
    assert amt.get_call_count() == 0


# --- Lazy client + error handling ----------------------------------------

def test_missing_api_key_raises_only_at_call_time(monkeypatch):
    monkeypatch.setattr(amt, "AGENTMAIL_API_KEY", "")
    amt._client = None  # force rebuild
    # Patch _get_client to bypass the missing-key path; this test only
    # proves that lazy init does not fail at import time.
    with patch.object(amt, "_get_client", return_value=_FakeClient()):
        out = amt.list_inboxes(limit=1)
    # Default `_to_json` falls back to `str()` for plain objects, so we
    # verify the call DID reach the client by checking the budget bumped.
    assert amt.get_call_count() == 1
    assert "FakeInbox" in out  # serialized via fallback


def test_sdk_exception_is_swallowed_into_string():
    class _Boom:
        class inboxes:
            @staticmethod
            def list(limit):
                raise RuntimeError("network down")

    amt._client = _Boom()
    out = amt.list_inboxes(limit=1)
    assert "list_inboxes failed" in out
    assert "network down" in out
    # Budget is bumped AFTER the try/except, so a failed call does NOT
    # consume budget — agent keeps its remaining attempts for retry.
    assert amt.get_call_count() == 0


def test_reset_call_count_zeros_counter():
    amt.create_inbox()
    amt.create_inbox()
    assert amt.get_call_count() == 2
    amt.reset_call_count()
    assert amt.get_call_count() == 0


# --- Pure helpers --------------------------------------------------------

def test_truncate_handles_none_and_short_strings():
    assert amt._truncate(None) == ""
    assert amt._truncate("abc", limit=10) == "abc"
    out = amt._truncate("a" * 100, limit=10)
    assert out.startswith("a" * 10)
    assert "truncated 90 chars" in out


# --- Live integration test ----------------------------------------------
# Skipped unless the user provides a real AGENTMAIL_API_KEY in their `.env`.
# The module-level `setdefault("AGENTMAIL_API_KEY", "test-key")` above means
# the constant inside `amt` is always non-empty, so we check the raw env
# directly to tell "real key" apart from "test fallback".

import json as _json  # noqa: E402


def _real_key_present() -> str | None:
    """Return the real key from `.env`, or None if it's empty/missing.

    The module-level `os.environ.setdefault("AGENTMAIL_API_KEY", "test-key")`
    above shadows the real key in `os.environ`. To tell them apart at
    `skipif` time (which runs before any test body), we read `.env`
    directly via dotenv with `override=True`.
    """
    from dotenv import load_dotenv

    load_dotenv(override=True)
    raw = os.environ.get("AGENTMAIL_API_KEY", "")
    if not raw or raw == "test-key":
        return None
    return raw


@pytest.mark.skipif(
    not _real_key_present(),
    reason="No real AGENTMAIL_API_KEY in environment; set one in .env to run this test.",
)
def test_list_inboxes_live_integration():
    """Calls AgentMail's real API to confirm wiring end-to-end.

    Auto-skipped unless `AGENTMAIL_API_KEY` is set in `.env` to something
    other than the unit-test fallback. With a real key, this:

      1. Force-reloads `.env` so the test fallback doesn't shadow the real key.
      2. Resets the cached client so the new key is picked up.
      3. Calls `list_inboxes(limit=5)` against the real AgentMail API.
      4. Asserts the response is JSON and parses cleanly.
    """
    # `dotenv.load_dotenv()` defaults to `override=False`, so the
    # module-level `setdefault("AGENTMAIL_API_KEY", "test-key")` above
    # would shadow the real key. Force the override here so we use
    # whatever is in `.env` RIGHT NOW.
    from dotenv import load_dotenv

    load_dotenv(override=True)

    # Pick up the freshly-loaded key into the module constants too.
    amt.AGENTMAIL_API_KEY = os.environ["AGENTMAIL_API_KEY"]
    amt.reset_call_count()
    amt._client = None  # force rebuild so the new key takes effect

    out = amt.list_inboxes(limit=5)

    # Should be valid JSON (the SDK returns Pydantic models; the tool
    # serializes them via model_dump -> json.dumps).
    parsed = _json.loads(out) if out.strip().startswith(("[", "{")) else None
    assert parsed is not None, f"Expected JSON, got: {out!r}"

    # The SDK returns either a list of inboxes or {"inboxes": [...], ...}.
    # Accept both shapes.
    if isinstance(parsed, dict):
        assert "inboxes" in parsed, f"Unexpected shape: {parsed.keys()}"
        inboxes = parsed["inboxes"]
    else:
        inboxes = parsed
    assert isinstance(inboxes, list)
    assert amt.get_call_count() == 1  # budget was bumped exactly once

    # Eyeball-friendly diagnostic. Each inbox should expose at least an
    # id and an address (the SDK uses either `address` or `email`).
    if inboxes:
        sample = inboxes[0]
        print(f"\n[agentmail live] {len(inboxes)} inbox(es); first = {sample}")
        assert "inbox_id" in sample or "id" in sample
        assert "address" in sample or "email" in sample
