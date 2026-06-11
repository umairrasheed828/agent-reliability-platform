from dataclasses import dataclass
from statistics import mean

from src.scoring.record import ScoredRun

AXES = ("faithfulness", "relevance")


@dataclass(frozen=True)
class AxisDrift:
    axis: str
    baseline_mean: float
    recent_mean: float
    delta: float  # recent - baseline (negative = quality dropped)
    regressed: bool  # delta <= -threshold


@dataclass(frozen=True)
class QualityDriftReport:
    axes: list[AxisDrift]

    @property
    def any_regression(self) -> bool:
        return any(a.regressed for a in self.axes)

    def summary(self) -> str:
        parts = []
        for a in self.axes:
            flag = "  [REGRESSION]" if a.regressed else ""
            parts.append(
                f"{a.axis}: {a.baseline_mean:.2f} -> {a.recent_mean:.2f} "
                f"(d {a.delta:+.2f}){flag}"
            )
        return " | ".join(parts)


def _axis_mean(runs: list[ScoredRun], axis: str) -> float:
    vals = [getattr(r, axis) for r in runs]
    return float(mean(vals)) if vals else 0.0


def quality_drift(
    baseline: list[ScoredRun],
    recent: list[ScoredRun],
    threshold: float = 0.5,
) -> QualityDriftReport:
    drifts = []
    for axis in AXES:
        b = _axis_mean(baseline, axis)
        r = _axis_mean(recent, axis)
        delta = r - b
        drifts.append(AxisDrift(axis, b, r, delta, regressed=delta <= -threshold))
    return QualityDriftReport(axes=drifts)
