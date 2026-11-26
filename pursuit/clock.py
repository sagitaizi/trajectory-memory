"""The clock memory: spiking clocks that fall into step with the target, and a learned map from
a clock's phase to the target's velocity, scaled by the clock's swing.

Everything it is shown is one visual delay old, so it pairs late velocity with late phase, and
`velocity(ahead_s)` reads the map that far ahead. Adapted from the MCSoC memory
(`trajmem.snn_phasemap`).
"""
from __future__ import annotations

import numpy as np
import torch

from trajmem.lmu import SpikingLmu, build_dynamics
from trajmem.phasemap import _Axis
from trajmem.snn_phasemap import lif_rates

from .oscillator import PERIODS_S, TWO_PI, OscillatorBank


class ClockMemory:
    """PursuitMemory. `map` (clocks, 2, cells) is learned by PES; `slow` is the slowly updated
    copy surprise is judged against, None until the map has settled."""

    def __init__(self, dt_s: float, periods_s=PERIODS_S, damping_cycles: float | None = 1.0, n_clock: int = 3000,
                 n_ring: int = 400, eta: float = 0.02, eta_start: float = 0.3, eta_tau_s: float = 2.0,
                 score_tau_s: float = 0.2, slow_tau_s: float = 30.0,
                 time_limit_s: float = 12.0, lag_s: float = 0.0, seed: int = 1):
        self.dt_s, self.time_limit_s = dt_s, time_limit_s
        self.eta, self.eta_start, self.eta_tau_s = eta, eta_start, eta_tau_s
        self.score_alpha, self.slow_tau_s = min(1.0, dt_s / score_tau_s), slow_tau_s
        self.lag_s = lag_s                              # lags in the chain other than the visual delay
        self.clocks = OscillatorBank(dt_s, periods_s, damping_cycles, n=n_clock, seed=seed)
        ring = build_dynamics(n_ring, 2, 1.2, lambda s: [0.0, 0.0], in_dims=2, tau_syn=0.01, seed=seed + 1)
        self.ring, self.n_ring = SpikingLmu(ring, dt_s), n_ring
        self.reset()

    def reset(self) -> None:
        B = len(self.clocks.periods_s)
        self.t, self.learned_s, self.slow_age_s = 0.0, 0.0, 0.0
        self.clocks.reset()
        self.axis, self.ring_state = _Axis(self.dt_s), self.ring.init_state(B)
        self.map, self.slow = torch.zeros(B, 2, self.n_ring), None
        self.err, self.score = np.full(B, np.nan), np.full(B, np.nan)
        self.best, self.t_chosen, self.err_history = None, None, []

    def observe(self, position, velocity) -> None:
        self.t += self.dt_s
        self.clocks.step(self.axis.drive(np.asarray(position, dtype=float)))
        swing = torch.tensor(self.clocks.swing, dtype=torch.float32)
        phase = torch.tensor(self.clocks.phase, dtype=torch.float32)
        with torch.no_grad():
            act, self.ring_state = self.ring.step(torch.stack([phase.cos(), phase.sin()], 1), self.ring_state)
            cells = act / 100.0
            if np.isfinite(velocity).all():
                self._learn(torch.tensor(velocity, dtype=torch.float32), cells * swing[:, None],
                            power=(cells * cells).sum(1, keepdim=True) + 1e-3)
        self._choose()

    def _learn(self, target, cells, power) -> None:
        err = target - torch.einsum("bcn,bn->bc", self.map, cells)
        eta = max(self.eta, self.eta_start * np.exp(-self.learned_s / self.eta_tau_s))
        self.map += eta * (err / power)[:, :, None] * cells[:, None, :]
        self.learned_s += self.dt_s
        self.err = self._smoothed(self.err, err)
        if self.slow is not None:
            self.slow_age_s += self.dt_s
            self.slow += self.dt_s / min(self.slow_tau_s, max(1.0, self.slow_age_s)) * (self.map - self.slow)
            self.score = self._smoothed(self.score, target - torch.einsum("bcn,bn->bc", self.slow, cells))

    def _smoothed(self, level, err) -> np.ndarray:
        e = err.norm(dim=1).numpy()
        return np.where(np.isnan(level), e, level + self.score_alpha * (e - level))

    def _choose(self) -> None:
        """The clock whose map predicts best, until the slow copy is saved."""
        if not np.isfinite(self.err).any() or self.slow is not None:
            return
        self.best = int(np.nanargmin(self.err))
        self.t_chosen = self.t if self.t_chosen is None else self.t_chosen
        if self._settled():
            self.slow = self.map.clone()

    def _settled(self) -> bool:
        """At least three cycles on a chosen clock and its error no longer falling, or the time limit."""
        lap, since = self.period(), self.t - self.t_chosen
        if since < 3.0 * lap:
            return False
        if since >= min(6.0 * lap, self.time_limit_s):
            return True
        self.err_history = [(t, e) for t, e in self.err_history + [(self.t, self.err[self.best])] if t >= self.t - lap]
        return len(self.err_history) > 1 and self.err_history[-1][1] > 0.9 * self.err_history[0][1]

    def velocity(self, ahead_s: float) -> np.ndarray:
        if self.best is None:
            return np.full(2, np.nan)
        i, w = self.best, self.ring.w
        xy = self.clocks.ahead(ahead_s + self.lag_s)[i]
        swing = float(np.hypot(*xy))
        tuning = lif_rates(w.scaled_encoders @ torch.tensor(xy / max(swing, 1e-6), dtype=torch.float32) + w.bias)
        return swing * (self.map[i] @ (tuning / 100.0)).numpy()

    def surprise(self) -> float:
        """How far the velocity is from what the slow copy expected, smoothed; NaN until it is saved."""
        return np.nan if self.best is None else float(self.score[self.best])

    def period(self) -> float:
        return np.nan if self.best is None else float(TWO_PI / self.clocks.omega[self.best])
