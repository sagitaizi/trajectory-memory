"""Prediction error, lock-on time, deviation-detection ROC + latency.
Definitions follow materials/02-methods/metrics.md."""
from __future__ import annotations

import numpy as np


def prediction_error(pred, gt) -> dict:
    """Distance between prediction and truth per step; median / IQR / mean over the
    steps where both are known. Units are whatever the inputs are in."""
    pred, gt = np.asarray(pred, dtype=float), np.asarray(gt, dtype=float)
    errors = np.hypot(pred[:, 0] - gt[:, 0], pred[:, 1] - gt[:, 1])
    e = errors[np.isfinite(errors)]
    if len(e) == 0:
        return {"errors": errors, "median": np.nan, "iqr": np.nan, "mean": np.nan, "n": 0}
    q1, q3 = np.percentile(e, [25, 75])
    return {"errors": errors, "median": float(np.median(e)), "iqr": float(q3 - q1),
            "mean": float(e.mean()), "n": int(len(e))}


def lock_on_time(errors, tol: float, dt: float) -> float:
    """Time of the first step after which the error stays under `tol` for the rest of
    the clip; inf if it never does. Unknown (NaN) steps are skipped."""
    errors = np.asarray(errors, dtype=float)
    known = np.flatnonzero(np.isfinite(errors))
    if len(known) == 0:
        return np.inf
    over = known[errors[known] >= tol]
    if len(over) == 0:
        return float(known[0] * dt)
    after = known[known > over[-1]]
    return float(after[0] * dt) if len(after) else np.inf


def deviation_roc(scores, times, deviation_times, threshold: float | None = None,
                  hold_n: int = 3) -> dict:
    """Score the deviation signal as a detector.

    Positives are steps at or after the first break, negatives the steps before it
    (all steps, on a clip without a break). `auc` is threshold-free. At the operating
    `threshold` (default: the 99th percentile of the negatives), a flag is `hold_n`
    consecutive steps above it: `latency_s` is the first flag after the break,
    `fp_per_min` the flags raised on negative steps.
    """
    s, t = np.asarray(scores, dtype=float), np.asarray(times, dtype=float)
    t_break = min(deviation_times) if len(deviation_times) else np.inf
    positive = t >= t_break
    if threshold is None:
        threshold = float(np.nanpercentile(s[~positive], 99)) if (~positive).any() else np.inf

    flags = _sustained(s > threshold, hold_n)
    step = float(np.median(np.diff(t))) if len(t) > 1 else np.nan
    neg_minutes = (~positive).sum() * step / 60
    out = {"threshold": threshold, "auc": np.nan, "latency_s": np.nan,
           "fp_per_min": float(_rising_edges(flags & ~positive) / neg_minutes) if neg_minutes else np.nan}
    if positive.any() and (~positive).any():
        out["auc"] = _auc(s[positive], s[~positive])
        hit = np.flatnonzero(flags & positive)
        out["latency_s"] = float(t[hit[0]] - t_break) if len(hit) else np.inf
    return out


def _sustained(above, n: int) -> np.ndarray:
    """True where `above` has held for `n` consecutive steps ending here."""
    if n <= 1:
        return above
    run = np.convolve(above.astype(int), np.ones(n, dtype=int), mode="full")[: len(above)]
    return run >= n


def _rising_edges(flags) -> int:
    return int(np.count_nonzero(np.diff(flags.astype(int), prepend=0) == 1))


def _auc(pos, neg) -> float:
    """Mann-Whitney AUC: P(score_pos > score_neg), ties half."""
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    order = np.concatenate([pos, neg]).argsort(kind="stable")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks over ties
    values = np.concatenate([pos, neg])
    _, inv, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
