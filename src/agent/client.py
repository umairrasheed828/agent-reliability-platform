from dataclasses import dataclass

import httpx

from src.config import settings


@dataclass(frozen=True)
class AgentRun:
    """One captured agent run: what the judge scores + the trajectory to audit."""

    question: str
    output: str  # the brief
    context: str  # verified_notes the writer used (faithfulness source)
    path: list[str]  # trajectory node order
    checks: dict[str, bool]  # deterministic reliability checks


def run_agent(question: str, timeout: float = 60.0) -> AgentRun:
    """Drive the P2 agent service over HTTP and capture its run for scoring."""
    resp = httpx.post(
        f"{settings.agent_url}/ask",
        json={"question": question},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return AgentRun(
        question=question,
        output=data["brief"],
        context=data.get("context", ""),  # P2's additive observability field
        path=data["path"],
        checks=data["checks"],
    )
