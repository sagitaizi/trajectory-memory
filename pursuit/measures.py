"""What the eye did, in the units pursuit studies use."""
from __future__ import annotations

import numpy as np

from trajmem.metrics import RatchetAlarm

ALARM_K = 10.0         # the smallest with no false alarm on the steady development runs (scripts/cogsci/tune.py)


def _fundamental(t, v, period_s: float) -> complex:
    """The complex amplitude of `v` at the period, with an offset taken out."""
    w = 2.0 * np.pi * t / period_s
    c, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(t), np.cos(w), np.sin(w)]), v, rcond=None)
    return complex(c[1], c[2])


def gain_lag(t, eye_velocity, target_velocity, period_s: float) -> tuple[float, float]:
    """Gain and lag (s, positive = the eye is behind) of eye velocity against target
    velocity along the target's main direction, at the motion's period."""
    ok = np.isfinite(eye_velocity).all(axis=1) & np.isfinite(target_velocity).all(axis=1)
    t, eye, target = np.asarray(t)[ok], eye_velocity[ok], target_velocity[ok]
    _, vecs = np.linalg.eigh(np.cov(target.T))
    ratio = _fundamental(t, eye @ vecs[:, -1], period_s) / _fundamental(t, target @ vecs[:, -1], period_s)
    return abs(ratio), float(np.angle(ratio)) * period_s / (2.0 * np.pi)


def alarms(t, surprise, dt_s: float, k: float = ALARM_K, floor: float = 0.0) -> list[float]:
    """When each alarm starts, s: the MCSoC ratcheting rule on the surprise (NaN = none yet)."""
    rule = RatchetAlarm(k=k, floor_px=floor, dt_s=dt_s)
    flags = np.array([rule.update(s, x) for s, x in zip(surprise, t)])
    return [float(x) for x in np.asarray(t)[np.diff(flags.astype(int), prepend=0) == 1]]
