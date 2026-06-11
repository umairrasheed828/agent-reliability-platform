from pathlib import Path

from src.scoring.record import ScoredRun
from src.store.metrics_store import all_runs, count_runs, init_store, insert_run


def _make_run(cohort: str = "live") -> ScoredRun:
    return ScoredRun(
        timestamp="2026-06-11T12:00:00+00:00",
        question="What is RAG?",
        output="RAG retrieves documents.",
        context="RAG fetches documents.",
        path=["supervisor", "writer"],
        checks={"verified_before_wrote": True},
        faithfulness=5,
        relevance=4,
        judge_model="gpt-4o-mini",
        cohort=cohort,
        rationale="ok",
    )


def test_insert_and_read_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "metrics.db"
    init_store(db)
    insert_run(_make_run(), db)
    insert_run(_make_run(cohort="baseline"), db)

    assert count_runs(db) == 2
    runs = all_runs(db)
    assert runs[0].path == ["supervisor", "writer"]  # JSON -> list survives
    assert runs[0].checks["verified_before_wrote"] is True  # JSON -> dict survives
    assert {r.cohort for r in runs} == {"live", "baseline"}
