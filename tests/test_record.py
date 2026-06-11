from judgekit import Judgment, Sample

from src.agent.client import AgentRun
from src.scoring.record import ScoredRun, score_agent_run


class _FakeJudge:
    def score(self, sample: Sample) -> Judgment:
        return Judgment(scores={"faithfulness": 5, "relevance": 4}, rationale="ok")


def test_score_agent_run_builds_record() -> None:
    run = AgentRun(
        question="What is RAG?",
        output="RAG retrieves documents and grounds the answer.",
        context="RAG fetches documents and conditions generation.",
        path=["supervisor", "writer"],
        checks={"verified_before_wrote": True},
    )
    record = score_agent_run(run, _FakeJudge(), cohort="baseline")

    assert isinstance(record, ScoredRun)
    assert record.faithfulness == 5
    assert record.relevance == 4
    assert record.cohort == "baseline"
    assert record.judge_model  # filled from settings
    assert record.timestamp.endswith("+00:00")  # UTC, timezone-aware
