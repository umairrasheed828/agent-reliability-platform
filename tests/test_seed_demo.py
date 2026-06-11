from pathlib import Path

from scripts.seed_demo import seed
from src.drift.engine import detect_drift
from src.store.metrics_store import count_runs, runs_by_cohort


def test_seed_creates_intentional_drift(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    seed(db_path=db)
    assert count_runs(db) == 80
    report = detect_drift(runs_by_cohort("baseline", db), runs_by_cohort("live", db))
    assert report.alert  # baseline healthy, live drifted -> the alert must fire
