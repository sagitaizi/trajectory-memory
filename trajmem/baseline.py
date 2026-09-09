"""Non-SNN comparators. Same TrajectoryMemory interface as model.py."""
from __future__ import annotations


class PeriodicKalman:
    """Kalman filter with a periodic-motion state model."""

    def fit(self, tracks) -> None:
        raise NotImplementedError

    def observe(self, x: float, y: float) -> None:
        raise NotImplementedError

    def predict(self, horizon_s: float) -> tuple[float, float]:
        raise NotImplementedError

    def deviation_score(self) -> float:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class HarmonicFit:
    """Fourier / harmonic fit to the observed path."""

    def fit(self, tracks) -> None:
        raise NotImplementedError

    def observe(self, x: float, y: float) -> None:
        raise NotImplementedError

    def predict(self, horizon_s: float) -> tuple[float, float]:
        raise NotImplementedError

    def deviation_score(self) -> float:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
