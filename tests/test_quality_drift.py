from src.drift.quality import quality_drift
from src.scoring.record import ScoredRun


def _run(faith: int, rel: int) -> ScoredRun:
    return ScoredRun(
        timestamp="t",
        question="q",
        output="o",
        context="c",
        path=[],
        checks={},
        faithfulness=faith,
        relevance=rel,
        judge_model="m",
        cohort="x",
        rationale="",
    )


def test_no_regression_when_stable() -> None:
    base = [_run(5, 5), _run(4, 5)]
    recent = [_run(5, 5), _run(4, 5)]
    assert not quality_drift(base, recent).any_regression


def test_detects_faithfulness_regression() -> None:
    base = [_run(5, 5), _run(5, 5)]  # mean 5.0
    recent = [_run(3, 5), _run(4, 5)]  # mean 3.5 -> delta -1.5
    report = quality_drift(base, recent, threshold=0.5)
    faith = next(a for a in report.axes if a.axis == "faithfulness")
    assert faith.delta == -1.5
    assert faith.regressed
    assert report.any_regression
