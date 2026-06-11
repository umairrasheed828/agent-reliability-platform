from dataclasses import dataclass
from datetime import datetime, timezone

from judgekit import cohens_kappa


def _mae(a: list[int], b: list[int]) -> float:
    if not a:
        return 0.0
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


@dataclass(frozen=True)
class CalibrationVerdict:
    timestamp: str
    n: int
    faithfulness_mae: float
    relevance_mae: float
    faithfulness_kappa: float
    relevance_kappa: float
    aligned: bool  # live judge tracks the calibrated anchor within tolerance

    def summary(self) -> str:
        status = "ALIGNED" if self.aligned else "DIVERGED"
        return (
            f"[{status}] live judge vs P4 anchor (n={self.n})\n"
            f"  faithfulness: MAE={self.faithfulness_mae:.2f} "
            f"kappa={self.faithfulness_kappa:.2f}\n"
            f"  relevance:    MAE={self.relevance_mae:.2f} "
            f"kappa={self.relevance_kappa:.2f}"
        )


def compare_judges(
    live_faithfulness: list[int],
    anchor_faithfulness: list[int],
    live_relevance: list[int],
    anchor_relevance: list[int],
    mae_tolerance: float = 0.6,
) -> CalibrationVerdict:
    """Agreement between the live (cheap) judge and the P4 calibrated anchor."""
    f_mae = _mae(live_faithfulness, anchor_faithfulness)
    r_mae = _mae(live_relevance, anchor_relevance)
    return CalibrationVerdict(
        timestamp=datetime.now(timezone.utc).isoformat(),
        n=len(live_faithfulness),
        faithfulness_mae=f_mae,
        relevance_mae=r_mae,
        faithfulness_kappa=cohens_kappa(live_faithfulness, anchor_faithfulness),
        relevance_kappa=cohens_kappa(live_relevance, anchor_relevance),
        aligned=f_mae <= mae_tolerance and r_mae <= mae_tolerance,
    )
