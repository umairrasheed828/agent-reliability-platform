from unittest.mock import patch

from src.agent.client import AgentRun, run_agent


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_run_agent_parses_response() -> None:
    payload = {
        "brief": "RAG retrieves documents and grounds the answer.",
        "context": "RAG fetches documents and conditions generation on them.",
        "path": [
            "supervisor",
            "researcher",
            "supervisor",
            "verifier",
            "supervisor",
            "writer",
            "supervisor",
        ],
        "checks": {"verified_before_wrote": True, "brief_present": True},
    }
    with patch("src.agent.client.httpx.post", return_value=_FakeResponse(payload)):
        run = run_agent("What is RAG?")

    assert isinstance(run, AgentRun)
    assert run.output.startswith("RAG retrieves")
    assert run.context.startswith("RAG fetches")
    assert run.checks["verified_before_wrote"] is True
    assert "writer" in run.path
