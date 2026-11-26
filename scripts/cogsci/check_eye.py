"""Is the eye loop sound? The reaction alone, and with a perfect memory that gives the true
velocity, on the development pendulum runs and clean sinusoids."""
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import _thesis_path  # noqa: E402,F401
from pursuit.loop import follow  # noqa: E402
from pursuit.measures import gain_lag  # noqa: E402
from trajmem.data import read_labels  # noqa: E402
from trajmem.trajectories import search_period  # noqa: E402

RES, DT, SETTLE_S = np.array([640.0, 480.0]), 0.005, 3.0
RUNS = [("small_01", "small_01", 0.0), ("wide_02", "wide_02__steady", 6.0)]
SIN_PERIODS = [0.8, 1.5, 2.5, 4.0]


class TrueVelocity:
    """Knows the true velocity, and that what it sees is one delay old."""

    def __init__(self, true_velocity, delay_s):
        self.true_velocity, self.delay_s = true_velocity, delay_s

    def reset(self):
        self.n = 0

    def observe(self, position, velocity):
        self.n += 1

    def velocity(self, ahead_s):
        return self.true_velocity(self.n * DT - self.delay_s + ahead_s)

    def surprise(self):
        return np.nan

    def period(self):
        return np.nan


def real_run(clip, positions, start_s):
    marks = np.array([m for m in read_labels(ROOT / "corpus/real/pendulum" / clip) if np.isfinite(m).all()])
    t, xy = marks[:, 0] - start_s, marks[:, 1:] * RES
    txy = np.load(ROOT / f"runs/paper/development/positions/{positions}__snn.npz")["txy"].copy()
    txy[:, 1:] *= RES
    return txy[(txy[:, 0] > t[0]) & (txy[:, 0] < t[-1])], CubicSpline(t, xy, axis=0).derivative(), search_period(t, xy)


def sinusoid(period_s, duration_s=20.0):
    w, t = 2 * np.pi / period_s, np.arange(0, duration_s, DT) + DT / 2
    txy = np.column_stack([t, 320 + 200 * np.sin(w * t), np.full_like(t, 240.0)])
    return txy, lambda s: np.stack([200 * w * np.cos(w * np.asarray(s)), 0 * np.asarray(s)], axis=-1), period_s


def report(name, txy, velocity, period_s):
    for delay_s in (0.08, 0.10, 0.12, 0.14):
        for label, memory in (("reaction alone", None), ("perfect memory", TrueVelocity(velocity, delay_s))):
            t0 = txy[0, 0] - DT / 2
            if memory is not None:
                memory.true_velocity = lambda s, v=velocity: v(s + t0)
            tr = follow(memory, txy, DT, delay_s=delay_s)
            ok = tr.t - tr.t[0] > SETTLE_S
            true = np.atleast_2d(velocity(tr.t))[ok]
            g, lag = gain_lag(tr.t[ok], tr.eye_velocity[ok], true, period_s)
            rms = np.sqrt(np.mean(np.sum((tr.eye_velocity[ok] - true) ** 2, 1)) / np.mean(np.sum(true ** 2, 1)))
            print(f"{name:14s} {period_s:6.2f} {delay_s * 1e3:6.0f}  {label:15s} {g:5.2f} {lag * 1e3:7.0f} {rms:8.2f}")


if __name__ == "__main__":
    print(f"{'run':14s} {'T s':>6s} {'delay':>6s}  {'':15s} {'gain':>5s} {'lag ms':>7s} {'err/v':>8s}")
    for clip, positions, start in RUNS:
        report(clip, *real_run(clip, positions, start))
    for T in SIN_PERIODS:
        report("sinusoid", *sinusoid(T))
