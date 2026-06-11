import json
import sqlite3
from pathlib import Path

from src.scoring.record import ScoredRun
from src.scoring.calibration import CalibrationVerdict

DB_PATH = Path("data/metrics.db")


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_store(db_path: Path = DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    TEXT    NOT NULL,
                question     TEXT    NOT NULL,
                output       TEXT    NOT NULL,
                context      TEXT    NOT NULL,
                path         TEXT    NOT NULL,
                checks       TEXT    NOT NULL,
                faithfulness INTEGER NOT NULL,
                relevance    INTEGER NOT NULL,
                judge_model  TEXT    NOT NULL,
                cohort       TEXT    NOT NULL,
                rationale    TEXT    NOT NULL
            )
            """
        )
        conn.commit()


def insert_run(run: ScoredRun, db_path: Path = DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO runs (
                timestamp, question, output, context, path, checks,
                faithfulness, relevance, judge_model, cohort, rationale
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.timestamp,
                run.question,
                run.output,
                run.context,
                json.dumps(run.path),
                json.dumps(run.checks),
                run.faithfulness,
                run.relevance,
                run.judge_model,
                run.cohort,
                run.rationale,
            ),
        )
        conn.commit()


def _row_to_run(row: sqlite3.Row) -> ScoredRun:
    return ScoredRun(
        timestamp=row["timestamp"],
        question=row["question"],
        output=row["output"],
        context=row["context"],
        path=json.loads(row["path"]),
        checks=json.loads(row["checks"]),
        faithfulness=row["faithfulness"],
        relevance=row["relevance"],
        judge_model=row["judge_model"],
        cohort=row["cohort"],
        rationale=row["rationale"],
    )


def all_runs(db_path: Path = DB_PATH) -> list[ScoredRun]:
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM runs ORDER BY timestamp").fetchall()
    return [_row_to_run(r) for r in rows]


def count_runs(db_path: Path = DB_PATH) -> int:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM runs").fetchone()
    return int(row["n"])


def runs_by_cohort(cohort: str, db_path: Path = DB_PATH) -> list[ScoredRun]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM runs WHERE cohort = ? ORDER BY timestamp", (cohort,)
        ).fetchall()
    return [_row_to_run(r) for r in rows]


def recent_runs(n: int, db_path: Path = DB_PATH) -> list[ScoredRun]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?", (n,)
        ).fetchall()
    return [_row_to_run(r) for r in reversed(rows)]  # back to chronological order


def init_calibration(db_path: Path = DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS calibration (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp          TEXT NOT NULL,
                n                  INTEGER NOT NULL,
                faithfulness_mae   REAL NOT NULL,
                relevance_mae      REAL NOT NULL,
                faithfulness_kappa REAL NOT NULL,
                relevance_kappa    REAL NOT NULL,
                aligned            INTEGER NOT NULL
            )
            """
        )
        conn.commit()


def insert_calibration(v: CalibrationVerdict, db_path: Path = DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO calibration (timestamp, n, faithfulness_mae, relevance_mae,
                faithfulness_kappa, relevance_kappa, aligned)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                v.timestamp,
                v.n,
                v.faithfulness_mae,
                v.relevance_mae,
                v.faithfulness_kappa,
                v.relevance_kappa,
                int(v.aligned),
            ),
        )
        conn.commit()


def latest_calibration(db_path: Path = DB_PATH) -> CalibrationVerdict | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM calibration ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return None
    return CalibrationVerdict(
        timestamp=row["timestamp"],
        n=row["n"],
        faithfulness_mae=row["faithfulness_mae"],
        relevance_mae=row["relevance_mae"],
        faithfulness_kappa=row["faithfulness_kappa"],
        relevance_kappa=row["relevance_kappa"],
        aligned=bool(row["aligned"]),
    )
