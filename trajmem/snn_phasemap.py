"""Clock-and-map memory in spiking neurons, the counterpart of `phasemap.PhaseMap`.

Each clock is a 4-D LIF population (Nengo-built, torch-run) whose recurrent wiring is the
adaptive-frequency-oscillator dynamics (Righetti, Buchli & Ijspeert 2006): state (x, y)
on the unit circle = the phase, a rate, and the drive; the pull F sin(phi) and the rate
adaptation are products the population computes. The clocks share one wiring and run as
one batch, differing in their starting rate. Each clock feeds a ring of phase cells (a
2-D LIF population with unit-circle encoders); the map is the ring -> position weights,
learned in the clip by the PES rule (the delta rule on decoders). Prediction ahead reads
the map at the ring's tuning curves for the rotated phase. Election, residual carry,
scale adaptation and the deviation snapshot are as in the prototype.
"""
from __future__ import annotations

import numpy as np
import torch

from .lmu import SpikingLmu, build_dynamics
from .phasemap import TWO_PI, _Axis

# the rate lives in the population as w in [-1, 1] over periods 4.5-0.7 s; the drive is halved
OMEGA_LO, OMEGA_HI = TWO_PI / 4.5, TWO_PI / 0.7
OMEGA_MID, OMEGA_SPAN = (OMEGA_LO + OMEGA_HI) / 2, (OMEGA_HI - OMEGA_LO) / 2
F_SCALE = 0.5


def omega_to_w(omega: float) -> float:
    return (omega - OMEGA_MID) / OMEGA_SPAN


def w_to_omega(w) -> float:
    return OMEGA_MID + OMEGA_SPAN * float(np.clip(w, -1.0, 1.0))
TAU_RC, TAU_REF = 0.02, 0.002


def clock_dynamics(tau: float, gamma: float, k_phase: float):
    """NEF recurrent function for one clock: tau * f(s) + s on the phase (x, y); the rate w
    and the drive F are inputs. The rate itself is a slow modulatory scalar outside the
    population (a neural integrator would drift), updated from the decoded F * y."""
    def f(s):
        x, y, w, Fs = s
        omega = OMEGA_MID + OMEGA_SPAN * w
        F = Fs / F_SCALE
        r2 = x * x + y * y
        pull = 1.0 - k_phase * F * y
        return [tau * (gamma * (1 - r2) * x - omega * pull * y) + x,
                tau * (gamma * (1 - r2) * y + omega * pull * x) + y, 0.0, 0.0]
    return f


def lif_rates(j: torch.Tensor) -> torch.Tensor:
    """Steady firing rate (Hz) of Nengo's LIF for input current j."""
    j = j.clamp(min=1.0 + 1e-6)
    return 1.0 / (TAU_REF - TAU_RC * torch.log1p(-1.0 / j)) * (j > 1.0 + 1e-6)


class SpikingPhaseMap:
    """TrajectoryMemory. Parameters mirror `PhaseMap` where the mechanism is the same."""

    def __init__(self, dt_s: float, periods_s=(0.8, 1.3, 2.0, 3.0, 4.0), n_clock: int = 6000, n_ring: int = 400,
                 gamma: float = 2.0, k_phase: float = 0.3, k_rate: float = 0.6, eta: float = 0.02,
                 eta_start: float = 0.3, eta_tau_s: float = 2.0, score_tau_s: float = 0.2, settle_s: float = 3.0,
                 resid_tau_s: float = 0.02, resid_decay_s: float = 2.0, k_scale: float = 1.0,
                 elect: str = "zero_crossings", snapshot_laps: float = 2.0, kick_s: float = 0.1,
                 tau_syn_s: float = 0.02, tau_ring_s: float = 0.01, resolution=(640, 480), seed: int = 1, device=None):
        self.dt_s, self.periods_s, self.n_clock, self.n_ring = dt_s, tuple(periods_s), n_clock, n_ring
        self.gamma, self.k_phase, self.k_rate = gamma, k_phase, k_rate
        self.eta, self.eta_start, self.eta_tau_s, self.score_tau_s, self.settle_s = eta, eta_start, eta_tau_s, score_tau_s, settle_s
        self.resid_tau_s, self.resid_decay_s, self.k_scale = resid_tau_s, resid_decay_s, k_scale
        self.elect, self.snapshot_laps, self.kick_s, self.tau_syn_s, self.tau_ring_s = elect, snapshot_laps, kick_s, tau_syn_s, tau_ring_s
        self.resolution, self.seed = np.asarray(resolution, dtype=float), seed
        self.device = torch.device(device or "cpu")
        torch.set_num_threads(min(4, torch.get_num_threads()))
        wc = build_dynamics(n_clock, 4, 1.7, clock_dynamics(tau_syn_s, gamma, k_phase), in_dims=4,
                            tau_syn=tau_syn_s, seed=seed)      # inputs: kick x, kick y, rate w, drive F
        self.clock = SpikingLmu(wc, dt_s, device=self.device)
        wr = build_dynamics(n_ring, 2, 1.2, lambda s: [0.0, 0.0], in_dims=2, tau_syn=tau_ring_s, seed=seed + 1)
        self.ring = SpikingLmu(wr, dt_s, device=self.device)
        self.B = len(self.periods_s)
        self.alpha = min(1.0, dt_s / score_tau_s)
        self.alpha_r = min(1.0, dt_s / resid_tau_s)
        self.reset()

    def fit(self, tracks) -> None:
        """Nothing to pretrain: the map is learned in the clip."""

    def reset(self) -> None:
        B = self.B
        self.t = 0.0
        self.axis = _Axis(self.dt_s)
        self.st_clock, self.st_ring = self.clock.init_state(B), self.ring.init_state(B)
        self.w_map = torch.zeros(B, 2, self.n_ring, device=self.device)
        self.snapshot = None
        self.centre = torch.full((B, 2), float("nan"), device=self.device)   # running centre of the map's output
        self.gain = np.ones(B)                                                  # output scale about that centre
        self.ring_act = torch.zeros(B, self.n_ring, device=self.device)
        self.xyw = np.zeros((B, 3))
        self.omega = np.array([TWO_PI / T for T in self.periods_s])
        self.err = np.full(B, np.nan)
        self.err_snap = np.full(B, np.nan)
        self.resid = np.zeros((B, 2))
        self.best, self.t_elected, self._err_hist = None, None, []
        self.last = np.array([np.nan, np.nan])
        self.n_learned = 0

    # -- one step --

    def observe(self, x: float, y: float) -> None:
        obs = np.array([x, y], dtype=float)
        self.t += self.dt_s
        drive = self.axis.drive(obs)
        kick = self.t < self.kick_s
        F = 0.0 if not np.isfinite(drive) else float(np.clip(drive, -1.5, 1.5))
        u = torch.zeros(self.B, 4, device=self.device)
        u[:, 0] = 1.0 if kick else 0.0
        u[:, 2] = torch.tensor([omega_to_w(o) for o in self.omega], dtype=torch.float32, device=self.device)
        u[:, 3] = F_SCALE * F
        with torch.no_grad():
            act, self.st_clock = self.clock.step(u, self.st_clock)
            s = self.clock.decode(act).reshape(self.B, 4)
            self.xyw = s[:, :3].cpu().numpy()
            y = np.clip(self.xyw[:, 1], -1.2, 1.2)
            self.omega = np.clip(self.omega - self.k_rate * self.omega * F * y * self.dt_s, OMEGA_LO, OMEGA_HI)
            xy = s[:, :2] / s[:, :2].norm(dim=1, keepdim=True).clamp(min=1e-3)
            self.ring_act, self.st_ring = self.ring.step(xy, self.st_ring)
            raw = torch.einsum("bcn,bn->bc", self.w_map, self.ring_act / 100.0)
            self.centre = torch.where(torch.isfinite(self.centre), self.centre + 0.002 * (raw - self.centre), raw)
            gain = torch.tensor(self.gain, dtype=torch.float32, device=self.device)[:, None]
            pred = self.centre + gain * (raw - self.centre)
            if np.isfinite(obs).all():
                target = torch.tensor(obs, dtype=torch.float32, device=self.device)
                err = target[None] - pred                                                  # (B, 2)
                learned = self.n_learned * self.dt_s
                eta = max(self.eta, self.eta_start * np.exp(-learned / self.eta_tau_s))
                a = self.ring_act / 100.0
                power = (a * a).sum(1, keepdim=True) + 1e-3                                # normalised LMS
                self.w_map += eta * ((target[None] - raw) / power)[:, :, None] * a[:, None, :]
                self.n_learned += 1
                e = err.norm(dim=1).cpu().numpy()
                self.err = np.where(np.isnan(self.err), e, self.err + self.alpha * (e - self.err))
                self.resid += self.alpha_r * (err.cpu().numpy() - self.resid)
                if self.k_scale > 0 and learned > 2.0:
                    radial = pred - self.centre
                    sc = ((err * radial).sum(1) / ((radial * radial).sum(1) + 1e-6)).cpu().numpy()
                    self.gain = np.clip(self.gain * (1.0 + self.k_scale * sc * self.dt_s), 0.5, 1.5)
                if self.snapshot is not None:
                    ps = torch.einsum("bcn,bn->bc", self.snapshot, self.ring_act / 100.0)
                    es = (target[None] - ps).norm(dim=1).cpu().numpy()
                    self.err_snap = np.where(np.isnan(self.err_snap), es, self.err_snap + self.alpha * (es - self.err_snap))
                self.last = obs
        self._elect()

    def _elect(self) -> None:
        if self.t < self.settle_s or not np.isfinite(self.err).any() or self.snapshot is not None:
            return
        t_zc = self.axis.period_zc()
        if self.elect == "zero_crossings" and np.isfinite(t_zc):
            self.best = int(min(range(self.B), key=lambda i: abs(np.log(TWO_PI / self._omega(i) / t_zc))))
        else:
            self.best = int(np.nanargmin(self.err))
        if self.t_elected is None:
            self.t_elected = self.t
        if self.snapshot is None and self.snapshot_laps > 0 and self._settled():
            self.snapshot = self.w_map.clone()

    def _settled(self) -> bool:
        lap = TWO_PI / self._omega(self.best)
        since = self.t - self.t_elected
        if since < max(3.0, self.snapshot_laps) * lap:
            return False
        if since >= min(6 * lap, 12.0):
            return True
        hist = getattr(self, "_err_hist", [])
        hist.append((self.t, self.err[self.best]))
        self._err_hist = [(t, e) for t, e in hist if t >= self.t - lap]
        return len(self._err_hist) > 1 and self._err_hist[-1][1] > 0.9 * self._err_hist[0][1]

    def _omega(self, i: int) -> float:
        return float(self.omega[i])

    def _read(self, i: int, phase_offset: float, snapshot: bool = False) -> np.ndarray:
        """The map at the clock's phase + `phase_offset`, via the ring's tuning curves."""
        x, y = self.xyw[i, :2]
        phi = np.arctan2(y, x) + phase_offset
        xy = torch.tensor([np.cos(phi), np.sin(phi)], dtype=torch.float32, device=self.device)
        w = self.ring.w
        rates = lif_rates(w.scaled_encoders @ xy + w.bias)
        table = self.snapshot if snapshot else self.w_map
        raw = table[i] @ (rates / 100.0)
        c = self.centre[i]
        return (c + self.gain[i] * (raw - c)).cpu().numpy() if bool(torch.isfinite(c).all()) else raw.cpu().numpy()

    # -- interface --

    def predict(self, horizon_s: float) -> tuple[float, float]:
        if self.best is None:
            return tuple(self.last)
        i = self.best
        p = self._read(i, self._omega(i) * horizon_s) + self.resid[i] * np.exp(-horizon_s / self.resid_decay_s)
        return tuple(p) if np.isfinite(p).all() else tuple(self.last)

    def deviation_score(self) -> float:
        if self.best is None or np.isnan(self.err[self.best]):
            return 0.0
        e = self.err_snap[self.best] if np.isfinite(self.err_snap[self.best]) else self.err[self.best]
        return float(e * self.resolution[0])

    def period(self) -> float:
        return np.nan if self.best is None else float(TWO_PI / self._omega(self.best))

    def path_points(self, fractions):
        if self.best is None:
            return None
        x, y = self.xyw[self.best, :2]
        phi0 = np.arctan2(y, x)
        return np.array([self._read(self.best, f * TWO_PI - phi0) for f in np.asarray(fractions, dtype=float)])
