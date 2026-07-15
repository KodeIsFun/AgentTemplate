"""`exa_search` — Exa-backed web search with a hard per-process call cap.

Two layers of protection against an unlimited loop:

1. **Tool-level guard** — this module tracks call count in a global counter
   and short-circuits to `{"status": "limit_reached"}` once `MAX_EXA_CALLS`
   is exhausted. The counter is process-wide; the analyst cannot reset it.
2. **Agent-level guard** — the analyst's `instruction` text explicitly caps
   itself to `MAX_EXA_CALLS` calls (the LLM sees the same number).

Per the tutorial's §"Common Pitfalls": output is capped (`numResults=3`,
`maxCharacters=1500`) so the analyst can't accidentally exceed the model's
context window.
"""

from __future__ import annotations

import os

import requests

from backend import config


# Module-level counter. Each call decrements; once 0 the tool refuses to call.
_CALLS_REMAINING: int = int(os.getenv("MAX_EXA_CALLS", "3"))


def _remaining() -> int:
    """Read-only accessor — handy for debugging or tests."""
    return _CALLS_REMAINING


def exa_search(query: str, num_results: int = 3) -> dict[str, str | int | list]:
    """Search the web via Exa. Caps at `MAX_EXA_CALLS` per process.

    Args:
        query: Focused, multi-word search string.
        num_results: 1..10; clamped to 3 here to stay within the model's
            context window.

    Returns:
        A dict with one of:
          - `{"status": "success", "results": [...], "remaining": int}`
          - `{"status": "limit_reached", "remaining": 0}`
          - `{"status": "error", "error": str}`
    """
    global _CALLS_REMAINING

    if _CALLS_REMAINING <= 0:
        return {"status": "limit_reached", "remaining": 0}

    api_key = os.getenv("EXA_API_KEY", "")
    if not api_key:
        return {
            "status": "error",
            "error": "EXA_API_KEY missing — set it in .env to enable web search",
        }

    # Clamp to keep context window safe.
    n = max(1, min(int(num_results or 3), 3))

    try:
        resp = requests.post(
            "https://api.exa.ai/search",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "numResults": n,
                "contents": {"text": {"maxCharacters": 1500}},
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return {"status": "error", "error": str(exc)}

    # Only count successful (or at least attempted) calls toward the cap.
    _CALLS_REMAINING -= 1

    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "text": r.get("text", ""),
        }
        for r in data.get("results", [])
    ]
    return {
        "status": "success",
        "remaining": _CALLS_REMAINING,
        "results": results,
    }