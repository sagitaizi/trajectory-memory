"""Clock-and-map memory, non-spiking reference: a phase advancing at a rate, a map from
phase to position learned online by a delta rule, and a locking rule that nudges phase
and rate from the mismatch (an adaptive-frequency oscillator, Righetti & Ijspeert 2006).
Several clocks start at different rates; the one with the smallest smoothed mismatch is
read. The spiking version is built against this.
"""
from __future__ import annotations

from collections import deque

import numpy as np

TWO_PI = 2.0 * np.pi


class _Clock:
    def __init__(self, omega: float, n_bins: int, width: float, dt_s: float, eta: float, k_phase: float,
                 k_rate: float, score_tau_s: float, resid_tau_s: float, k_scale: float, slow_tau_s: float):
        self.phi, self.omega, self.n, self.width, self.dt = 0.0, omega, n_bins, width, dt_s
        self.slow_tau_s, self.snap_age = slow_tau_s, 0.0
        self.k_scale = k_scale
        self.alpha_r = min(1.0, dt_s / resid_tau_s)
        self.eta, self.k_phase, self.k_rate = eta, k_phase, k_rate
        self.map = np.zeros((n_bins, 2))
        self.conf = np.zeros(n_bins)                   # how much each bin has learned
        self.err = np.nan                              # smoothed mismatch, normalised units
        self.resid = np.zeros(2)                       # smoothed (obs - map), carried into predictions
        self.pull = 0.0                                # smoothed signed pull on the clock: sustained = off-pattern
        self.snapshot = None                           # (map, conf) once locked: the remembered path, for deviation
        self.err_snap = np.nan
        self.alpha = min(1.0, dt_s / score_tau_s)
        self.centres = np.arange(n_bins) * TWO_PI / n_bins

    def _weights(self, phi: float) -> np.ndarray:
        d = np.angle(np.exp(1j * (self.centres - phi)))          # signed distance on the ring
        return np.exp(-0.5 * (d / self.width) ** 2)

    def read(self, phi, snapshot: bool = False) -> np.ndarray:
        """Position at phase(s) `phi`: confidence-weighted average of the nearby bins;
        NaN where the map is still empty there. `snapshot`: the remembered path instead."""
        table, conf = self.snapshot if snapshot and self.snapshot is not None else (self.map, self.conf)
        phi = np.atleast_1d(np.asarray(phi, dtype=float))
        w = np.exp(-0.5 * (np.angle(np.exp(1j * (self.centres[None] - phi[:, None]))) / self.width) ** 2) * conf
        s = w.sum(1)
        out = (w @ table) / np.maximum(s, 1e-9)[:, None]
        out[s < 0.05] = np.nan
        return out

    def score(self, obs: np.ndarray) -> None:
        """Mismatch against the remembered path, from the raw observation (before the gate:
        a break is exactly what the gate would refuse to believe)."""
        if self.snapshot is None or not np.isfinite(obs).all():
            return
        es = float(np.hypot(*(obs - self.read(self.phi, snapshot=True)[0])))
        self.err_snap = es if np.isnan(self.err_snap) else self.err_snap + self.alpha * (es - self.err_snap)

    def step(self, obs: np.ndarray, drive: float) -> None:
        """`drive`: the motion's main-axis signal, zero-mean, unit-ish amplitude. Locking is
        the adaptive-frequency-oscillator rule (Righetti, Buchli & Ijspeert 2006): the input
        pulls the phase and the rate through F sin(phi), independent of the map."""
        if np.isfinite(drive):
            pull = float(np.clip(drive, -1.5, 1.5)) * np.sin(self.phi) * self.omega   # in the clock's own units
            self.pull += self.alpha * (pull / self.omega - self.pull)
            self.phi = (self.phi + (self.omega - self.k_phase * pull) * self.dt) % TWO_PI
            self.omega = float(np.clip(self.omega - self.k_rate * pull * self.dt, TWO_PI / 6.0, TWO_PI / 0.4))
        else:
            self.phi = (self.phi + self.omega * self.dt) % TWO_PI
        if not np.isfinite(obs).all():
            return
        p = self.read(self.phi)[0]
        if np.isfinite(p).all():
            e = float(np.hypot(*(obs - p)))
            self.err = e if np.isnan(self.err) else self.err + self.alpha * (e - self.err)
            self.resid += self.alpha_r * (obs - p - self.resid)
            if self.k_scale > 0 and self.conf.sum() > 2 * self.n:
                centre = (self.map * self.conf[:, None]).sum(0) / self.conf.sum()
                radial = p - centre
                s = float((obs - p) @ radial) / (float(radial @ radial) + 1e-6)
                self.map = centre + (1.0 + self.k_scale * s * self.dt) * (self.map - centre)
        g = self._weights(self.phi)
        rate = np.maximum(self.eta, g / (self.conf + 1.0))       # running mean at first, then a steady step
        self.map += (rate * g)[:, None] * (obs - self.map)
        self.conf += g
        if self.snapshot is not None:                            # the slow map consolidates toward the working one,
            self.snap_age += self.dt                             # quickly while young, slowly once old
            a = self.dt / min(self.slow_tau_s, max(1.0, self.snap_age))
            table, conf = self.snapshot
            table += a * (self.map - table)
            conf += a * (self.conf - conf)


class _Axis:
    """The motion's main axis, online: a slow running mean and covariance; `drive` is the
    observation projected on the leading eigenvector, divided by its running amplitude."""

    def __init__(self, dt_s: float, tau_s: float = 3.0):
        self.a = min(1.0, dt_s / tau_s)
        self.dt = dt_s
        self.mean, self.cov, self.n = np.zeros(2), np.zeros((2, 2)), 0
        self.sign, self.crossings, self.t = 0, [], 0.0

    def period_zc(self) -> float:
        """Period from the drive's upward zero crossings (with hysteresis), NaN before two."""
        if len(self.crossings) < 3:
            return np.nan
        gaps = np.diff(self.crossings[-6:])
        return float(np.median(gaps)) if gaps.std() < 0.15 * gaps.mean() else np.nan     # only when steady

    def drive(self, obs: np.ndarray) -> float:
        self.t += self.dt                                     # clip time, whether or not the target was seen
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
        if not hasattr(self, "u") or float(u @ self.u) < 0:  # keep the axis's sign stable
            u = -u if hasattr(self, "u") else u
        self.u = u
        f = float(d @ u) / np.sqrt(2.0 * var)              # unit amplitude for a sinusoid
        if f > 0.3 and self.sign <= 0:
            if self.sign < 0:
                self.crossings.append(self.t)
            self.sign = 1
        elif f < -0.3 and self.sign >= 0:
            self.sign = -1
        return f


class PhaseMap:
    """TrajectoryMemory: clocks race, the best is read (`period`, `path_points`, `predict`)."""

    def __init__(self, dt_s: float, periods_s=(0.8, 1.3, 2.0, 3.0, 4.0), n_bins: int = 64, width_bins: float = 1.5,
                 eta: float = 0.05, k_phase: float = 0.3, k_rate: float = 0.6, score_tau_s: float = 0.2,
                 settle_s: float = 3.0, resid_tau_s: float = 0.02, resid_decay_s: float = 2.0,
                 prefer_slow: float = 0.0, k_scale: float = 1.0, elect: str = "zero_crossings", pull_weight: float = 0.0,
                 snapshot_laps: float = 2.0, smooth_frac: float = 0.0, smooth_min_s: float = 0.01,
                 gate_k: float = 2.0, gate_floor_px: float = 10.0, slow_tau_s: float = 30.0, resolution=(640, 480)):
        self.dt_s, self.periods_s, self.n_bins = dt_s, tuple(periods_s), n_bins
        self.width = width_bins * TWO_PI / n_bins
        self.eta, self.k_phase, self.k_rate, self.score_tau_s = eta, k_phase, k_rate, score_tau_s
        self.settle_s, self.resolution = settle_s, np.asarray(resolution, dtype=float)
        self.resid_tau_s, self.resid_decay_s, self.prefer_slow = resid_tau_s, resid_decay_s, prefer_slow
        self.k_scale, self.elect, self.pull_weight, self.snapshot_laps = k_scale, elect, pull_weight, snapshot_laps
        self.smooth_frac, self.smooth_min_s, self.gate_k, self.gate_floor_px = smooth_frac, smooth_min_s, gate_k, gate_floor_px
        self.slow_tau_s = slow_tau_s
        self.reset()

    def fit(self, tracks) -> None:
        """Nothing to pretrain: the map is learned in the clip."""

    def reset(self) -> None:
        self.clocks = [_Clock(TWO_PI / T, self.n_bins, self.width, self.dt_s, self.eta, self.k_phase, self.k_rate,
                              self.score_tau_s, self.resid_tau_s, self.k_scale, self.slow_tau_s) for T in self.periods_s]
        self.t = 0.0
        self.best = None
        self.t_elected = None
        self.last = np.array([np.nan, np.nan])
        self.axis = _Axis(self.dt_s)
        self.filtered = np.array([np.nan, np.nan])       # the input stage: gated, period-scaled smoothing
        self._err_hist = []
        self.rejected = 0
        self.gaps = deque(maxlen=200)                     # observation-vs-map gaps over the last second, all samples
        self.accepted = True

    def _input_stage(self, obs: np.ndarray) -> np.ndarray:
        """Smooth the observation over a fixed fraction of the locked period, and once
        locked discount an observation the map does not expect (a centroid flip to the
        string): the memory's expectation gates its input."""
        unseen = np.array([np.nan, np.nan])
        if not np.isfinite(obs).all():
            return unseen
        c = self._clock
        tau = self.smooth_min_s if c is None else max(self.smooth_min_s, self.smooth_frac * TWO_PI / c.omega)
        if c is not None:
            expected = c.read(c.phi + c.omega * tau)[0]          # the map is learned from the lagged input
            if np.isfinite(expected).all():
                gap = float(np.hypot(*(obs - expected)))
                self.gaps.append(gap)
                if gap > self.gate_k * float(np.median(self.gaps)) + self.gate_floor_px / self.resolution[0]:
                    self.rejected += 1
                    self.accepted = False
                    return unseen
        self.accepted = True
        a = min(1.0, self.dt_s / tau)
        self.filtered = obs if not np.isfinite(self.filtered).all() else self.filtered + a * (obs - self.filtered)
        return self.filtered

    def observe(self, x: float, y: float) -> None:
        raw = np.array([x, y], dtype=float)
        self.t += self.dt_s
        obs = self._input_stage(raw)
        drive = self.axis.drive(obs)
        for c in self.clocks:
            c.step(obs, drive)
        if self._clock is not None:
            self._clock.score(raw)
        if np.isfinite(obs).all():
            self.last = obs
        errs = np.array([c.err for c in self.clocks])
        if self.best is not None and self.clocks[self.best].snapshot is not None:
            return                                                   # committed: the memory is this clock
        if self.t >= self.settle_s and np.isfinite(errs).any():
            good = np.flatnonzero(errs <= np.nanmin(errs) * (1.0 + self.prefer_slow))
            t_zc = self.axis.period_zc()
            if self.elect == "zero_crossings" and np.isfinite(t_zc):
                good = np.flatnonzero(errs <= np.nanmin(errs) * 1.5)
                self.best = int(min(good, key=lambda i: abs(np.log(TWO_PI / self.clocks[i].omega / t_zc))))
            else:
                self.best = int(min(good, key=lambda i: self.clocks[i].omega))
            if self.t_elected is None:
                self.t_elected = self.t
            c = self.clocks[self.best]
            if c.snapshot is None and self.snapshot_laps > 0 and self._settled(c):
                c.snapshot = (c.map.copy(), c.conf.copy())          # the remembered path

    def _settled(self, c) -> bool:
        """Snapshot time: at least `snapshot_laps` since election and the mismatch no longer
        falling by more than 10 % per lap (or six laps, whichever first)."""
        lap = TWO_PI / c.omega
        since = self.t - self.t_elected
        if since < max(3.0, self.snapshot_laps) * lap:
            return False
        if since >= min(6 * lap, 12.0):
            return True
        hist = self._err_hist
        hist.append((self.t, c.err))
        self._err_hist = [(t, e) for t, e in hist if t >= self.t - lap]
        return len(self._err_hist) > 1 and self._err_hist[-1][1] > 0.9 * self._err_hist[0][1]

    @property
    def _clock(self):
        return None if self.best is None else self.clocks[self.best]

    def predict(self, horizon_s: float) -> tuple[float, float]:
        c = self._clock
        if c is None:
            return tuple(self.last)
        lag = max(self.smooth_min_s, self.smooth_frac * TWO_PI / c.omega)
        p = c.read(c.phi + c.omega * (horizon_s + lag))[0] + c.resid * np.exp(-horizon_s / self.resid_decay_s)
        return tuple(p) if np.isfinite(p).all() else tuple(self.last)

    def deviation_score(self) -> float:
        c = self._clock
        if c is None or np.isnan(c.err):
            return 0.0
        err = c.err_snap if np.isfinite(c.err_snap) else c.err
        return float(err * self.resolution[0] + self.pull_weight * abs(c.pull))

    def period(self) -> float:
        c = self._clock
        return np.nan if c is None else float(TWO_PI / c.omega)

    def path_points(self, fractions):
        c = self._clock
        return None if c is None else c.read(np.asarray(fractions, dtype=float) * TWO_PI)
