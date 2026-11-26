"""The clock memory: spiking clocks that fall into step with the target, and a learned map from
a clock's phase to the target's velocity, scaled by the clock's swing.

Everything it is shown is one visual delay old, so it pairs late velocity with late phase, and
`velocity(ahead_s)` reads the map that far ahead. Each clock is judged by how well its read-ahead
one delay earlier matches what then arrives; the best is chosen, and it moves the eye only while
it does better than assuming the target kept its speed. Until then the reaction alone moves the
eye. Adapted from the MCSoC memory
(`trajmem.snn_phasemap`).
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch

from trajmem.lmu import SpikingLmu, build_dynamics
from trajmem.phasemap import _Axis
from trajmem.snn_phasemap import lif_rates

from .eye import VISUAL_DELAY_S
from .oscillator import PERIODS_S, TWO_PI, OscillatorBank


class ClockMemory:
    """PursuitMemory. `map` (clocks, 2, cells) is learned by PES; `slow` is the slowly updated
    copy surprise is judged against, None until the map has settled."""

    def __init__(self, dt_s: float, periods_s=PERIODS_S, damping_cycles: float | None = 1.0, n_clock: int = 3000,
                 n_ring: int = 400, eta: float = 0.02, eta_start: float = 0.3, eta_tau_s: float = 2.0,
                 score_tau_s: float = 0.2, slow_tau_s: float = 30.0,
                 time_limit_s: float = 12.0, lag_s: float = 0.0, check_ahead_s: float = VISUAL_DELAY_S,
                 check_tau_s: float = 1.0, seed: int = 1):
        self.dt_s, self.time_limit_s = dt_s, time_limit_s
        self.eta, self.eta_start, self.eta_tau_s = eta, eta_start, eta_tau_s
        self.score_alpha, self.slow_tau_s = min(1.0, dt_s / score_tau_s), slow_tau_s
        self.lag_s = lag_s                              # lags in the chain other than the visual delay
        self.check_ahead_s, self.check_steps = check_ahead_s, max(1, int(round(check_ahead_s / dt_s)))
        self.check_alpha = min(1.0, dt_s / check_tau_s)
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
        self.score = np.full(B, np.nan)
        self.best, self.t_chosen, self.err_history = None, None, []
        self.predicted, self.held = np.full(B, np.nan), np.nan
        self.made = deque()                             # (each clock's read-ahead, velocity seen) per step

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
        self._check(np.asarray(velocity, dtype=float))
        self._choose()

    def _check(self, velocity) -> None:
        """How well each clock read one delay ahead, and how well keeping the speed then seen did."""
        if len(self.made) == self.check_steps:
            ahead, seen = self.made.popleft()
            if np.isfinite(velocity).all() and np.isfinite(seen).all():
                a = self.check_alpha
                self.predicted = self._smoothed(self.predicted, torch.tensor(velocity - ahead, dtype=torch.float32), a)
                self.held = float(self._smoothed(np.array([self.held]),
                                                 torch.tensor(velocity - seen, dtype=torch.float32)[None], a)[0])
        self.made.append((self._ahead(self.check_ahead_s), velocity))

    def _learn(self, target, cells, power) -> None:
        err = target - torch.einsum("bcn,bn->bc", self.map, cells)
        eta = max(self.eta, self.eta_start * np.exp(-self.learned_s / self.eta_tau_s))
        self.map += eta * (err / power)[:, :, None] * cells[:, None, :]
        self.learned_s += self.dt_s
        if self.slow is not None:
            self.slow_age_s += self.dt_s
            self.slow += self.dt_s / min(self.slow_tau_s, max(1.0, self.slow_age_s)) * (self.map - self.slow)
            self.score = self._smoothed(self.score, target - torch.einsum("bcn,bn->bc", self.slow, cells))

    def _smoothed(self, level, err, alpha=None) -> np.ndarray:
        e = err.norm(dim=1).numpy()
        return np.where(np.isnan(level), e, level + (self.score_alpha if alpha is None else alpha) * (e - level))

    def _choose(self) -> None:
        """The clock that reads ahead best; the slow copy of every map is saved once it has settled."""
        if not np.isfinite(self.predicted).any():
            return
        self.best = int(np.nanargmin(self.predicted))
        if self.slow is not None:
            return
        if self.t_chosen is None and self._predicts():
            self.t_chosen = self.t
        if self.t_chosen is not None and self._settled():
            self.slow = self.map.clone()

    def _settled(self) -> bool:
        """At least three cycles since the memory first moved the eye and its error no longer falling,
        or the time limit."""
        lap, since = self.period(), self.t - self.t_chosen
        if since < 3.0 * lap:
            return False
        if since >= min(6.0 * lap, self.time_limit_s):
            return True
        self.err_history = [(t, e) for t, e in self.err_history + [(self.t, self.predicted[self.best])] if t >= self.t - lap]
        return len(self.err_history) > 1 and self.err_history[-1][1] > 0.9 * self.err_history[0][1]

    def velocity(self, ahead_s: float) -> np.ndarray:
        if not self._predicts():
            return np.full(2, np.nan)
        return self._ahead(ahead_s)[self.best]

    def _predicts(self) -> bool:
        """The chosen clock reads ahead better than keeping the speed would."""
        return self.best is not None and bool(self.predicted[self.best] < self.held)

    def _ahead(self, ahead_s: float) -> np.ndarray:
        """(clocks, 2): each clock's map read where the clock will be `ahead_s` from now."""
        w = self.ring.w
        xy = self.clocks.ahead(ahead_s + self.lag_s)
        swing = np.hypot(*xy.T)
        unit = torch.tensor(xy / np.maximum(swing, 1e-6)[:, None], dtype=torch.float32)
        with torch.no_grad():
            tuning = lif_rates(unit @ w.scaled_encoders.T + w.bias)
            return swing[:, None] * torch.einsum("bcn,bn->bc", self.map, tuning / 100.0).numpy()

    def surprise(self) -> float:
        """How far the velocity is from what the slow copy expected, smoothed; NaN until it is saved."""
        return np.nan if self.best is None else float(self.score[self.best])

    def period(self) -> float:
        return np.nan if self.best is None else float(TWO_PI / self.clocks.omega[self.best])
