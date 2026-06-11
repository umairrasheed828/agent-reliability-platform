import math
from collections import Counter
from dataclasses import dataclass

from src.scoring.record import ScoredRun

AXES = ("faithfulness", "relevance")
SCORE_BINS = (1, 2, 3, 4, 5)
_EPS = 1e-6


@dataclass(frozen=True)
class AxisDistributionDrift:
    axis: str
    psi: float
    shifted: bool  # psi >= threshold


@dataclass(frozen=True)
class DistributionDriftReport:
    axes: list[AxisDistributionDrift]

    @property
    def any_shift(self) -> bool:
        return any(a.shifted for a in self.axes)

    def summary(self) -> str:
        parts = []
        for a in self.axes:
            flag = "  [SHIFT]" if a.shifted else ""
            parts.append(f"{a.axis}: PSI={a.psi:.3f}{flag}")
        return " | ".join(parts)


def _distribution(runs: list[ScoredRun], axis: str) -> dict[int, float]:
    counts = Counter(getattr(r, axis) for r in runs)
    n = len(runs)
    if n == 0:
        return {b: 0.0 for b in SCORE_BINS}
    return {b: counts.get(b, 0) / n for b in SCORE_BINS}


def _psi(baseline_dist: dict[int, float], recent_dist: dict[int, float]) -> float:
    psi = 0.0
    for b in SCORE_BINS:
        p = max(baseline_dist[b], _EPS)
        q = max(recent_dist[b], _EPS)
        psi += (q - p) * math.log(q / p)
    return psi


def distribution_drift(
    baseline: list[ScoredRun],
    recent: list[ScoredRun],
    threshold: float = 0.25,
) -> DistributionDriftReport:
    out = []
    for axis in AXES:
        b = _distribution(baseline, axis)
        r = _distribution(recent, axis)
        psi = _psi(b, r)
        out.append(AxisDistributionDrift(axis, psi, shifted=psi >= threshold))
    return DistributionDriftReport(axes=out)
