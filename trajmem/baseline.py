"""Non-SNN comparators. Same TrajectoryMemory interface as model.py.

Both take one position per step of `dt_s`, estimate the period from the first
`warmup_s` of observations and keep it, then predict ahead and score surprise. The
warm-up must cover at least about a cycle; real clips run up to 4 s periods.
A NaN observation (target unseen) is skipped. Positions are in normalised image
coordinates, like every Clip's ground truth.
"""
from __future__ import annotations

import numpy as np

from .trajectories import search_period


class _Periodic:
    """Shared plumbing: the observation buffer, the warm-up, the surprise smoothing."""

    def __init__(self, dt_s: float, warmup_s: float = 5.0, n_harmonics: int = 2,
                 score_tau_s: float = 0.1):
        self.dt_s, self.warmup_s, self.n_harmonics = dt_s, warmup_s, n_harmonics
        self.score_alpha = dt_s / score_tau_s
        self.reset()

    def reset(self) -> None:
        self.t_now = -self.dt_s
        self.times, self.xys = [], []                     # every observation so far
        self.period_s = None
        self.last_known = (np.nan, np.nan)
        self.score = 0.0

    def fit(self, tracks) -> None:
        """Nothing to pretrain: these baselines learn each clip from its own warm-up."""

    def deviation_score(self) -> float:
        return float(self.score)

    def observe(self, x: float, y: float) -> None:
        self.t_now += self.dt_s
        if not (np.isfinite(x) and np.isfinite(y)):
            return
        self.times.append(self.t_now)
        self.xys.append((float(x), float(y)))
        self.last_known = (float(x), float(y))
        if self.period_s is None:
            if self.t_now >= self.warmup_s and len(self.times) >= 8:
                self.period_s = search_period(np.array(self.times), np.array(self.xys))
                self._start(np.array(self.times), np.array(self.xys))
            return
        self._step(self.t_now, np.array([x, y], dtype=float))

    def _smooth(self, surprise: float) -> None:
        self.score += self.score_alpha * (surprise - self.score)

    def _design(self, t):
        """[1, cos k w t, sin k w t ...] rows for the harmonic model at times t."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        w = 2 * np.pi / self.period_s
        cols = [np.ones_like(t)]
        for k in range(1, self.n_harmonics + 1):
            cols += [np.cos(k * w * t), np.sin(k * w * t)]
        return np.column_stack(cols)


class HarmonicFit(_Periodic):
    """Least-squares harmonics over a sliding window, extended forward in time.

    Surprise is the distance between each new observation and what the previous
    fit said it would be, relative to the fit's own residual spread.
    """

    def __init__(self, dt_s: float, warmup_s: float = 5.0, n_harmonics: int = 2,
                 window_periods: float = 3.0, score_tau_s: float = 0.1):
        self.window_periods = window_periods
        super().__init__(dt_s, warmup_s, n_harmonics, score_tau_s)

    def reset(self) -> None:
        super().reset()
        self.coef, self.scale = None, None

    def _start(self, t, xy) -> None:
        self._refit(t, xy)

    def _refit(self, t, xy) -> None:
        design = self._design(t)
        self.coef, *_ = np.linalg.lstsq(design, xy, rcond=None)
        resid = np.hypot(*(design @ self.coef - xy).T)
        self.scale = max(float(np.median(resid)), 1e-4)

    def _step(self, t_now, xy) -> None:
        expected = self._design(t_now) @ self.coef
        self._smooth(float(np.hypot(*(expected[0] - xy))) / self.scale)
        keep = max(0, len(self.times) - int(self.window_periods * self.period_s / self.dt_s))
        self._refit(np.array(self.times[keep:]), np.array(self.xys[keep:]))

    def predict(self, horizon_s: float) -> tuple[float, float]:
        if self.coef is None:
            return self.last_known
        x, y = (self._design(self.t_now + horizon_s) @ self.coef)[0]
        return float(x), float(y)


class PeriodicKalman(_Periodic):
    """Kalman filter whose state is the harmonic decomposition of each coordinate:
    [mean, (c_k, d_k) per harmonic], each pair rotating at k w. Prediction to a
    horizon is the same rotation applied further. Surprise is the normalised
    innovation (chi-squared with two degrees of freedom when the model holds).
    """

    def __init__(self, dt_s: float, warmup_s: float = 5.0, n_harmonics: int = 2,
                 process_noise: float = 1e-6, measurement_noise: float = 1e-4,
                 score_tau_s: float = 0.1):
        self.q, self.r = process_noise, measurement_noise
        super().__init__(dt_s, warmup_s, n_harmonics, score_tau_s)

    def reset(self) -> None:
        super().reset()
        self.state = None                                # (n, 2): one column per coordinate
        self.cov = None
        self.t_state = None

    @property
    def _h(self) -> np.ndarray:
        h = np.zeros(1 + 2 * self.n_harmonics)
        h[0] = 1.0
        h[1::2] = 1.0                                    # p = mean + sum of the cosines
        return h

    def _transition(self, dt: float) -> np.ndarray:
        n = 1 + 2 * self.n_harmonics
        f = np.eye(n)
        w = 2 * np.pi / self.period_s
        for k in range(1, self.n_harmonics + 1):
            c, s = np.cos(k * w * dt), np.sin(k * w * dt)
            i = 2 * k - 1
            f[i:i + 2, i:i + 2] = [[c, s], [-s, c]]
        return f

    def _start(self, t, xy) -> None:
        """Seed the state from a least-squares fit of the warm-up, phased at its end."""
        coef, *_ = np.linalg.lstsq(self._design(t - t[-1]), xy, rcond=None)
        self.state, self.t_state = coef, t[-1]
        n = len(coef)
        self.cov = np.eye(n) * self.r
        self.scale = 1.0

    def _step(self, t_now, xy) -> None:
        f = self._transition(t_now - self.t_state)
        state = f @ self.state
        cov = f @ self.cov @ f.T + np.eye(len(state)) * self.q
        h = self._h
        innovation = xy - h @ state
        s = h @ cov @ h + self.r
        gain = cov @ h / s
        self.state = state + np.outer(gain, innovation)
        self.cov = cov - np.outer(gain, h @ cov)
        self.t_state = t_now
        self._smooth(float(innovation @ innovation / s))

    def predict(self, horizon_s: float) -> tuple[float, float]:
        if self.state is None:
            return self.last_known
        ahead = self._transition(self.t_now + horizon_s - self.t_state) @ self.state
        x, y = self._h @ ahead
        return float(x), float(y)
