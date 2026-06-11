from pathlib import Path

from src.drift.engine import detect_drift
from src.scoring.record import ScoredRun
from src.store.metrics_store import init_store, insert_run, recent_runs, runs_by_cohort


def _run(faith: int, cohort: str = "baseline", ts: str = "t") -> ScoredRun:
    return ScoredRun(
        timestamp=ts,
        question="q",
        output="o",
        context="c",
        path=[],
        checks={},
        faithfulness=faith,
        relevance=5,
        judge_model="m",
        cohort=cohort,
        rationale="",
    )


def test_runs_by_cohort_filters(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    init_store(db)
    insert_run(_run(5, "baseline"), db)
    insert_run(_run(5, "baseline"), db)
    insert_run(_run(3, "live"), db)
    assert len(runs_by_cohort("baseline", db)) == 2
    assert len(runs_by_cohort("live", db)) == 1


def test_recent_runs_chronological(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    init_store(db)
    insert_run(_run(5, "x", "2026-06-11T10:00:00+00:00"), db)
    insert_run(_run(4, "x", "2026-06-11T11:00:00+00:00"), db)
    insert_run(_run(3, "x", "2026-06-11T12:00:00+00:00"), db)
    assert [r.faithfulness for r in recent_runs(2, db)] == [4, 3]


def test_detect_drift_alerts_on_regression() -> None:
    report = detect_drift([_run(5), _run(5)], [_run(2), _run(3)])
    assert report.alert
    assert report.baseline_n == 2 and report.recent_n == 2


def test_detect_drift_stable() -> None:
    assert not detect_drift([_run(5), _run(4)], [_run(5), _run(4)]).alert
