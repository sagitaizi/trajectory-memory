"""Harness: walk a clip through a memory, keep the per-step trace, score it.

`evaluate_clip` is the one loop every method goes through -- baselines now, the
SNN once it exists -- so the comparison is on identical inputs. The corpus-wide
pretrain -> freeze -> evaluate run is added with Phase C.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Clip
from .frontend import to_position
from .metrics import deviation_roc, lock_on_time, prediction_error


@dataclass
class Trace:
    """What a memory did on one clip, step by step. Positions are normalised."""
    t: np.ndarray                  # window centres, s
    obs: np.ndarray                # (N, 2) what the frontend saw; NaN = target unseen
    pred: np.ndarray               # (N, 2) prediction for t + horizon
    gt_ahead: np.ndarray           # (N, 2) truth at t + horizon; NaN where unknown
    score: np.ndarray              # (N,) deviation score
    horizon_s: float
    window_us: int


def make_memory(name: str, dt_s: float, **params):
    from .baseline import HarmonicFit, PeriodicKalman

    kinds = {"kalman": PeriodicKalman, "harmonic": HarmonicFit}
    if name not in kinds:
        raise ValueError(f"unknown memory {name!r}; one of {sorted(kinds)}")
    return kinds[name](dt_s=dt_s, **params)


def evaluate_clip(memory, clip: Clip, window_us: int, horizon_s: float) -> Trace:
    memory.reset()
    rows = []
    for t, x, y in to_position(clip, window_us):
        memory.observe(x, y)
        rows.append((t, x, y, *memory.predict(horizon_s), memory.deviation_score()))
    a = np.array(rows, dtype=float)
    ahead = a[:, 0] + horizon_s
    gt = np.full((len(a), 2), np.nan)
    inside = ahead <= clip.duration_us / 1e6
    if clip.gt is not None and inside.any():
        gt[inside] = np.atleast_2d(clip.gt(ahead[inside]))
    return Trace(t=a[:, 0], obs=a[:, 1:3], pred=a[:, 3:5], gt_ahead=gt, score=a[:, 5],
                 horizon_s=horizon_s, window_us=window_us)


def score_trace(trace: Trace, clip: Clip, tol_px: float, settle_s: float = 0.0,
                threshold: float | None = None) -> dict:
    """Prediction error (px, on steps before any break and after `settle_s`), lock-on
    time (s, from the clip's start), and the deviation-detection numbers."""
    scale = np.array(clip.meta["resolution"], dtype=float)
    err_all = prediction_error(trace.pred * scale, trace.gt_ahead * scale)["errors"]
    t_break = min(clip.deviation_times) if clip.deviation_times else np.inf
    steady = (trace.t >= settle_s) & (trace.t + trace.horizon_s < t_break)
    err = prediction_error(trace.pred[steady] * scale, trace.gt_ahead[steady] * scale)
    err.pop("errors")
    before_break = trace.t + trace.horizon_s < t_break
    return {
        "error_px": err,
        "lock_on_s": lock_on_time(err_all[before_break], tol_px, trace.window_us / 1e6),
        "deviation": deviation_roc(trace.score, trace.t, clip.deviation_times, threshold=threshold),
        "n_steps": int(len(trace.t)),
        "unseen_fraction": float(np.isnan(trace.obs[:, 0]).mean()),
    }
