from dataclasses import dataclass

from src.drift.distribution import DistributionDriftReport, distribution_drift
from src.drift.quality import QualityDriftReport, quality_drift
from src.scoring.record import ScoredRun


@dataclass(frozen=True)
class DriftReport:
    baseline_n: int
    recent_n: int
    quality: QualityDriftReport
    distribution: DistributionDriftReport

    @property
    def alert(self) -> bool:
        return self.quality.any_regression or self.distribution.any_shift

    def summary(self) -> str:
        status = "ALERT" if self.alert else "stable"
        return (
            f"[{status}] baseline n={self.baseline_n} recent n={self.recent_n}\n"
            f"  quality:      {self.quality.summary()}\n"
            f"  distribution: {self.distribution.summary()}"
        )


def detect_drift(
    baseline: list[ScoredRun],
    recent: list[ScoredRun],
    quality_threshold: float = 0.5,
    psi_threshold: float = 0.25,
) -> DriftReport:
    return DriftReport(
        baseline_n=len(baseline),
        recent_n=len(recent),
        quality=quality_drift(baseline, recent, quality_threshold),
        distribution=distribution_drift(baseline, recent, psi_threshold),
    )
