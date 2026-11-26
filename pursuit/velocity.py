"""Velocity from positions: the change per step, and the light smoothing the memory's input gets."""
from __future__ import annotations

import numpy as np

SMOOTHING_S = 0.02


class Rate:
    """Change per second between consecutive samples; NaN when either is unseen."""

    def __init__(self, dt_s: float):
        self.dt_s, self.last = dt_s, np.full(2, np.nan)

    def step(self, x) -> np.ndarray:
        rate, self.last = (np.asarray(x) - self.last) / self.dt_s, np.array(x, dtype=float)
        return rate


class Smooth:
    """First-order smoothing. NaN in gives NaN out and leaves the state as it was."""

    def __init__(self, dt_s: float, tau_s: float = SMOOTHING_S):
        self.alpha, self.state = min(1.0, dt_s / tau_s), None

    def step(self, x) -> np.ndarray:
        x = np.array(x, dtype=float)
        if not np.isfinite(x).all():
            return np.full_like(x, np.nan)
        self.state = x if self.state is None else self.state + self.alpha * (x - self.state)
        return self.state
