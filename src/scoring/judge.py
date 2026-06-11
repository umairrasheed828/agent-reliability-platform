from judgekit import Axis, Judge, Judgment, LLMJudge, Sample, openai_complete

from src.config import settings

FAITHFULNESS = Axis(
    name="faithfulness",
    description=(
        "Is every claim in the OUTPUT grounded in the CONTEXT? "
        "5: every claim is supported by the context. "
        "3: mostly grounded, one minor unsupported detail. "
        "1: a clearly fabricated or contradicted claim. "
        "Judge ONLY grounding here, never topical fit."
    ),
)

RELEVANCE = Axis(
    name="relevance",
    description=(
        "Does the OUTPUT address the QUESTION that was asked? "
        "5: directly and fully answers it. "
        "3: partially, or a narrower version. "
        "1: answers a different question or is off-topic. "
        "Judge ONLY topical fit. A wrong claim is a FAITHFULNESS "
        "problem, never a relevance one."
    ),
)

AXES = [FAITHFULNESS, RELEVANCE]


def build_judge() -> LLMJudge:
    """The live production judge: gpt-4o-mini behind judgekit's LLMJudge."""
    return LLMJudge(AXES, openai_complete(model=settings.judge_model))


def score_run(judge: Judge, question: str, output: str, context: str) -> Judgment:
    """Score one agent run on the calibrated faithfulness/relevance rubric."""
    sample = Sample(input=question, output=output, context=context)
    return judge.score(sample)
