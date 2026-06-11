from pathlib import Path
from unittest.mock import patch

from judgekit import Judgment, Sample

from src.runner import load_probes, run_probes
from src.scoring.record import ScoredRun
from src.store.metrics_store import count_runs


class _FakeJudge:
    def score(self, sample: Sample) -> Judgment:
        return Judgment(scores={"faithfulness": 5, "relevance": 5}, rationale="ok")


def _canned(question: str, judge: object, cohort: str = "live") -> ScoredRun:
    return ScoredRun(
        timestamp="2026-06-11T12:00:00+00:00",
        question=question,
        output="o",
        context="c",
        path=["supervisor", "writer"],
        checks={"x": True},
        faithfulness=5,
        relevance=5,
        judge_model="gpt-4o-mini",
        cohort=cohort,
        rationale="ok",
    )


def test_load_probes(tmp_path: Path) -> None:
    f = tmp_path / "probes.jsonl"
    f.write_text('{"question": "q1"}\n{"question": "q2"}\n', encoding="utf-8")
    assert load_probes(f) == ["q1", "q2"]


def test_run_probes_persists(tmp_path: Path) -> None:
    db = tmp_path / "metrics.db"
    with patch("src.runner.evaluate_question", side_effect=_canned):
        records = run_probes(
            _FakeJudge(), cohort="baseline", probes=["q1", "q2", "q3"], db_path=db
        )

    assert len(records) == 3
    assert count_runs(db) == 3
