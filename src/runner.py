import json
from pathlib import Path

from judgekit import Judge

from src.scoring.record import ScoredRun, evaluate_question
from src.store.metrics_store import DB_PATH, init_store, insert_run

PROBES_FILE = Path("eval/probes.jsonl")


def load_probes(path: Path = PROBES_FILE) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["question"] for line in lines if line.strip()]


def run_probes(
    judge: Judge,
    cohort: str = "live",
    probes: list[str] | None = None,
    db_path: Path = DB_PATH,
) -> list[ScoredRun]:
    """Drive every probe question through the loop and persist each scored run."""
    init_store(db_path)
    questions = probes if probes is not None else load_probes()
    records: list[ScoredRun] = []
    for q in questions:
        record = evaluate_question(q, judge, cohort=cohort)
        insert_run(record, db_path)
        records.append(record)
    return records
