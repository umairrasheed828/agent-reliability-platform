"""Seed the metrics store with DEMO data: a healthy baseline cohort and a
deliberately-drifted 'live' cohort. Dev/demo only -- production data comes from
run_probes() against the live P2 agent. The drift is intentional so the
dashboard and drift engine can be demonstrated offline and reproducibly.
"""

import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.runner import load_probes
from src.scoring.record import ScoredRun
from src.store.metrics_store import DB_PATH, init_store, insert_run

from src.scoring.calibration import CalibrationVerdict
from src.store.metrics_store import (
    init_calibration,
    insert_calibration,
)

_PATH = [
    "supervisor",
    "researcher",
    "supervisor",
    "verifier",
    "supervisor",
    "writer",
    "supervisor",
]
_CHECKS = {"verified_before_wrote": True, "brief_present": True}


def _record(
    question: str, faith: int, rel: int, cohort: str, ts: datetime
) -> ScoredRun:
    return ScoredRun(
        timestamp=ts.isoformat(),
        question=question,
        output="(demo brief)",
        context="(demo context)",
        path=_PATH,
        checks=_CHECKS,
        faithfulness=faith,
        relevance=rel,
        judge_model="gpt-4o-mini",
        cohort=cohort,
        rationale="(demo seed)",
    )


def seed_calibration(db_path: Path = DB_PATH) -> None:
    """Seed one demo calibration verdict (mirrors P4's real benchmark)."""
    init_calibration(db_path)
    insert_calibration(
        CalibrationVerdict(
            timestamp=datetime.now(timezone.utc).isoformat(),
            n=24,
            faithfulness_mae=0.46,
            relevance_mae=0.20,
            faithfulness_kappa=0.55,
            relevance_kappa=0.58,
            aligned=True,
        ),
        db_path,
    )


def seed(db_path: Path = DB_PATH, seed_value: int = 0) -> None:
    rng = random.Random(seed_value)
    questions = load_probes()
    init_store(db_path)
    start = datetime.now(timezone.utc) - timedelta(hours=2)

    for i in range(40):  # baseline: healthy
        q = questions[i % len(questions)]
        insert_run(
            _record(
                q,
                rng.choice([5, 5, 5, 4]),
                rng.choice([5, 5, 4]),
                "baseline",
                start + timedelta(minutes=i),
            ),
            db_path,
        )

    for i in range(40):  # recent: faithfulness deliberately drifted down
        q = questions[i % len(questions)]
        insert_run(
            _record(
                q,
                rng.choice([3, 3, 4, 2]),
                rng.choice([5, 5, 4]),
                "live",
                start + timedelta(hours=1, minutes=i),
            ),
            db_path,
        )

    seed_calibration(db_path)


if __name__ == "__main__":
    seed()
    print("Seeded 40 baseline + 40 drifted 'live' runs into the metrics store.")
