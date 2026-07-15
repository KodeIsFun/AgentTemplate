"""One-shot `InMemoryRunner` helper — lifted from §"Use Case 1 / 4. Run it
one-shot (stateless, perfect for an API)" in the tutorial.

Use this to call the pipeline programmatically without spinning up the ADK
CLI. Each call returns the final markdown text produced by the analyst.
"""

from __future__ import annotations

import asyncio

from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part


async def run_agent_once(
    agent,
    user_message: str,
    app_name: str = "doc_analysis_app",
) -> str:
    """Run `agent` once against `user_message` and return the final text."""
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


if __name__ == "__main__":
    # Tiny smoke test: prints the pipeline output for a trivial prompt.
    from agent import root_agent

    sample = "Summarize the documents in one short paragraph."
    result = asyncio.run(run_agent_once(root_agent, sample))
    print(result)