from pathlib import Path

from src.scoring.calibration import CalibrationVerdict
from src.store.metrics_store import (
    init_calibration,
    insert_calibration,
    latest_calibration,
)


def test_calibration_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    init_calibration(db)
    assert latest_calibration(db) is None
    insert_calibration(
        CalibrationVerdict("t", 24, 0.46, 0.20, 0.55, 0.58, aligned=True), db
    )
    v = latest_calibration(db)
    assert v is not None
    assert v.faithfulness_mae == 0.46
    assert v.aligned is True
