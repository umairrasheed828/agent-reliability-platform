from src.drift.distribution import distribution_drift
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


def test_no_shift_when_identical() -> None:
    base = [_run(5, 5)] * 10
    recent = [_run(5, 5)] * 10
    assert not distribution_drift(base, recent).any_shift


def test_detects_distribution_shift() -> None:
    base = [_run(5, 5)] * 10  # all faithfulness = 5
    recent = [_run(3, 5)] * 10  # all faithfulness = 3
    report = distribution_drift(base, recent, threshold=0.25)
    faith = next(a for a in report.axes if a.axis == "faithfulness")
    assert faith.psi > 0.25
    assert faith.shifted
    assert report.any_shift
