from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from judgekit import Judge

from src.agent.client import AgentRun, run_agent
from src.config import settings
from src.scoring.judge import score_run


@dataclass(frozen=True)
class ScoredRun:
    """The linchpin record: one scored agent run, fixed in time."""

    timestamp: str
    question: str
    output: str
    context: str
    path: list[str]
    checks: dict[str, bool]
    faithfulness: int
    relevance: int
    judge_model: str
    cohort: str
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def score_agent_run(run: AgentRun, judge: Judge, cohort: str = "live") -> ScoredRun:
    """Score a captured AgentRun into a timestamped ScoredRun record."""
    judgment = score_run(judge, run.question, run.output, run.context)
    return ScoredRun(
        timestamp=_now(),
        question=run.question,
        output=run.output,
        context=run.context,
        path=run.path,
        checks=run.checks,
        faithfulness=judgment.scores["faithfulness"],
        relevance=judgment.scores["relevance"],
        judge_model=settings.judge_model,
        cohort=cohort,
        rationale=judgment.rationale,
    )


def evaluate_question(question: str, judge: Judge, cohort: str = "live") -> ScoredRun:
    """End-to-end: drive the agent over HTTP, then score the run."""
    run = run_agent(question)
    return score_agent_run(run, judge, cohort)
