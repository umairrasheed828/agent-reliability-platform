"""Offline cross-check: re-score recent REAL runs with the P4 anchor and record
the live-vs-anchor calibration verdict. Requires the `gpu` extra + a GPU.

Run locally:  uv run --extra gpu python -m scripts.run_crosscheck
"""

from judgekit import Sample

from src.scoring.anchor import FineTunedJudge
from src.scoring.calibration import compare_judges
from src.store.metrics_store import init_calibration, insert_calibration, recent_runs


def main(n: int = 20) -> None:
    real = [r for r in recent_runs(n) if not r.output.startswith("(demo")]
    if not real:
        print("No real runs to cross-check. Run probes against the live agent first.")
        return

    anchor = FineTunedJudge()
    live_f, anchor_f, live_r, anchor_r = [], [], [], []
    for r in real:
        j = anchor.score(Sample(input=r.question, output=r.output, context=r.context))
        live_f.append(r.faithfulness)
        anchor_f.append(j.scores["faithfulness"])
        live_r.append(r.relevance)
        anchor_r.append(j.scores["relevance"])

    verdict = compare_judges(live_f, anchor_f, live_r, anchor_r)
    init_calibration()
    insert_calibration(verdict)
    print(verdict.summary())


if __name__ == "__main__":
    main()
