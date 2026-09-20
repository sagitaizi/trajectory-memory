"""Clock-and-map memory, non-spiking reference: a phase advancing at a rate, a map from
phase to position learned online by a delta rule, and a locking rule that nudges phase
and rate from the mismatch (an adaptive-frequency oscillator, Righetti & Ijspeert 2006).
Several clocks start at different rates; the one with the smallest smoothed mismatch is
read. The spiking version is built against this.
"""
from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


class _Clock:
    def __init__(self, omega: float, n_bins: int, width: float, dt_s: float, eta: float, k_phase: float,
                 k_rate: float, score_tau_s: float):
        self.phi, self.omega, self.n, self.width, self.dt = 0.0, omega, n_bins, width, dt_s
        self.eta, self.k_phase, self.k_rate = eta, k_phase, k_rate
        self.map = np.zeros((n_bins, 2))
        self.conf = np.zeros(n_bins)                   # how much each bin has learned
        self.err = np.nan                              # smoothed mismatch, normalised units
        self.alpha = min(1.0, dt_s / score_tau_s)
        self.centres = np.arange(n_bins) * TWO_PI / n_bins

    def _weights(self, phi: float) -> np.ndarray:
        d = np.angle(np.exp(1j * (self.centres - phi)))          # signed distance on the ring
        return np.exp(-0.5 * (d / self.width) ** 2)

    def read(self, phi) -> np.ndarray:
        """Position at phase(s) `phi`: confidence-weighted average of the nearby bins;
        NaN where the map is still empty there."""
        phi = np.atleast_1d(np.asarray(phi, dtype=float))
        w = np.exp(-0.5 * (np.angle(np.exp(1j * (self.centres[None] - phi[:, None]))) / self.width) ** 2) * self.conf
        s = w.sum(1)
        out = (w @ self.map) / np.maximum(s, 1e-9)[:, None]
        out[s < 0.05] = np.nan
        return out

    def step(self, obs: np.ndarray, drive: float) -> None:
        """`drive`: the motion's main-axis signal, zero-mean, unit-ish amplitude. Locking is
        the adaptive-frequency-oscillator rule (Righetti, Buchli & Ijspeert 2006): the input
        pulls the phase and the rate through F sin(phi), independent of the map."""
        if np.isfinite(drive):
            pull = drive * np.sin(self.phi)
            self.phi = (self.phi + (self.omega - self.k_phase * pull) * self.dt) % TWO_PI
            self.omega = float(np.clip(self.omega - self.k_rate * pull * self.dt, 0.3, 20.0))
        else:
            self.phi = (self.phi + self.omega * self.dt) % TWO_PI
        if not np.isfinite(obs).all():
            return
        p = self.read(self.phi)[0]
        if np.isfinite(p).all():
            e = float(np.hypot(*(obs - p)))
            self.err = e if np.isnan(self.err) else self.err + self.alpha * (e - self.err)
        g = self._weights(self.phi)
        rate = np.maximum(self.eta, g / (self.conf + 1.0))       # running mean at first, then a steady step
        self.map += (rate * g)[:, None] * (obs - self.map)
        self.conf += g


class _Axis:
    """The motion's main axis, online: a slow running mean and covariance; `drive` is the
    observation projected on the leading eigenvector, divided by its running amplitude."""

    def __init__(self, dt_s: float, tau_s: float = 3.0):
        self.a = min(1.0, dt_s / tau_s)
        self.mean, self.cov, self.n = np.zeros(2), np.zeros((2, 2)), 0

    def drive(self, obs: np.ndarray) -> float:
        if not np.isfinite(obs).all():
            return np.nan
        self.n += 1
        a = max(self.a, 1.0 / self.n)
        self.mean += a * (obs - self.mean)
        d = obs - self.mean
        self.cov += a * (np.outer(d, d) - self.cov)
        if self.n < 20:
            return np.nan
        vals, vecs = np.linalg.eigh(self.cov)
        u, var = vecs[:, -1], max(vals[-1], 1e-8)
        return float(d @ u) / np.sqrt(2.0 * var)              # unit amplitude for a sinusoid


class PhaseMap:
    """TrajectoryMemory: clocks race, the best is read (`period`, `path_points`, `predict`)."""

    def __init__(self, dt_s: float, periods_s=(0.8, 1.3, 2.0, 3.0, 4.0), n_bins: int = 64, width_bins: float = 1.5,
                 eta: float = 0.05, k_phase: float = 2.0, k_rate: float = 4.0, score_tau_s: float = 0.2,
                 settle_s: float = 3.0, resolution=(640, 480)):
        self.dt_s, self.periods_s, self.n_bins = dt_s, tuple(periods_s), n_bins
        self.width = width_bins * TWO_PI / n_bins
        self.eta, self.k_phase, self.k_rate, self.score_tau_s = eta, k_phase, k_rate, score_tau_s
        self.settle_s, self.resolution = settle_s, np.asarray(resolution, dtype=float)
        self.reset()

    def fit(self, tracks) -> None:
        """Nothing to pretrain: the map is learned in the clip."""

    def reset(self) -> None:
        self.clocks = [_Clock(TWO_PI / T, self.n_bins, self.width, self.dt_s, self.eta, self.k_phase, self.k_rate,
                              self.score_tau_s) for T in self.periods_s]
        self.t = 0.0
        self.best = None
        self.last = np.array([np.nan, np.nan])
        self.axis = _Axis(self.dt_s)

    def observe(self, x: float, y: float) -> None:
        obs = np.array([x, y], dtype=float)
        self.t += self.dt_s
        drive = self.axis.drive(obs)
        for c in self.clocks:
            c.step(obs, drive)
        if np.isfinite(obs).all():
            self.last = obs
        errs = np.array([c.err for c in self.clocks])
        if self.t >= self.settle_s and np.isfinite(errs).any():
            self.best = int(np.nanargmin(errs))

    @property
    def _clock(self):
        return None if self.best is None else self.clocks[self.best]

    def predict(self, horizon_s: float) -> tuple[float, float]:
        c = self._clock
        if c is None:
            return tuple(self.last)
        p = c.read(c.phi + c.omega * horizon_s)[0]
        return tuple(p) if np.isfinite(p).all() else tuple(self.last)

    def deviation_score(self) -> float:
        c = self._clock
        return 0.0 if c is None or np.isnan(c.err) else float(c.err * self.resolution[0])

    def period(self) -> float:
        c = self._clock
        return np.nan if c is None else float(TWO_PI / c.omega)

    def path_points(self, fractions):
        c = self._clock
        return None if c is None else c.read(np.asarray(fractions, dtype=float) * TWO_PI)
