"""Displacement error (ADE/FDE), path-shape error, lock-on time, deviation-detection
ROC + latency. Definitions follow materials/02-methods/metrics.md."""
from __future__ import annotations

from collections import deque

import numpy as np

SUB_SHIFTS = 8                     # phase-alignment resolution of path_shape_error, in fractions of a point


def displacement_error(pred, gt) -> dict:
    """Distance between prediction and truth per step; median / IQR / mean over the
    steps where both are known. Units are whatever the inputs are in.

    At one horizon h this is the trajectory-prediction literature's FDE(h), pooled over
    the clip's steps rather than over a set of trajectories; `ade_fde` combines the
    horizons."""
    pred, gt = np.asarray(pred, dtype=float), np.asarray(gt, dtype=float)
    errors = np.hypot(pred[:, 0] - gt[:, 0], pred[:, 1] - gt[:, 1])
    e = errors[np.isfinite(errors)]
    if len(e) == 0:
        return {"errors": errors, "median": np.nan, "iqr": np.nan, "mean": np.nan, "n": 0}
    q1, q3 = np.percentile(e, [25, 75])
    return {"errors": errors, "median": float(np.median(e)), "iqr": float(q3 - q1),
            "mean": float(e.mean()), "n": int(len(e))}


def ade_fde(by_horizon: dict) -> dict:
    """ADE and FDE from one `displacement_error` per horizon, in the trajectory-prediction
    convention: FDE is the error at the longest horizon, ADE the mean over every horizon
    up to it. Our horizons are the future instants we predict, so their mean is the
    average displacement over the predicted stretch."""
    hs = sorted(h for h, e in by_horizon.items() if np.isfinite(e))
    if not hs:
        return {"ade": np.nan, "fde": np.nan}
    return {"ade": float(np.mean([by_horizon[h] for h in hs])), "fde": float(by_horizon[hs[-1]])}


def path_shape_error(cycle, reference) -> float:
    """How far a remembered cycle is from the true one: mean point distance over the
    cycle at the best circular shift, so a phase slip is not counted at every point.
    Both are (M, 2) sampled uniformly in phase; direction matters (a path run backwards
    is another path). NaN if the cycle has unknown points."""
    cycle, reference = np.asarray(cycle, dtype=float), np.asarray(reference, dtype=float)
    if cycle.shape != reference.shape:
        raise ValueError(f"cycle {cycle.shape} and reference {reference.shape} must match")
    if not np.isfinite(cycle).all():
        return np.nan
    m = len(cycle)
    idx = (np.arange(m)[None, :] + np.arange(m)[:, None]) % m                # (shift, point)
    frac = np.arange(SUB_SHIFTS) / SUB_SHIFTS
    # shifts finer than one point, by interpolating along the cycle
    shifted = (1 - frac)[:, None, None, None] * cycle[idx] + frac[:, None, None, None] * cycle[(idx + 1) % m]
    d = np.hypot(*(shifted - reference[None, None]).transpose(3, 0, 1, 2))   # (frac, shift, point)
    return float(d.mean(axis=2).min())


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


# Spreads above the settled score, calibrated per input on the development clips: the
# smallest k that misses no break, and among those the fewest false alarms. The classical
# centroid affords a k with neither; through the spiking localiser a deviation is less
# separable from quiet running, and every k costs one or the other.
RATCHET_K_BY_INPUT = {"centroid": 25.0, "frame_centroid": 25.0, "snn": 16.0}
RATCHET_K_BY_CHECKPOINT = {"pend_ft": 30.0}    # spiking localisers calibrated on their own, by file stem
RATCHET_K = 25.0                   # spreads above the settled score: the smallest of 3..30 that raised
                                   # no false alarm on the development clips (breaks still caught in 0.23 s)


def k_for_input(name: str | None, checkpoint=None) -> float:
    """The alarm constant calibrated for that position input: its checkpoint's own if it has
    one (RATCHET_K_BY_CHECKPOINT), else the input's (RATCHET_K_BY_INPUT)."""
    from pathlib import Path

    if checkpoint is not None and Path(checkpoint).stem in RATCHET_K_BY_CHECKPOINT:
        return RATCHET_K_BY_CHECKPOINT[Path(checkpoint).stem]
    return RATCHET_K_BY_INPUT.get(name or "centroid", RATCHET_K)


def ratchet_threshold(scores, times, k: float = RATCHET_K, window_s: float = 2.0,
                      warmup_s: float = 5.0, floor_px: float = 5.0) -> np.ndarray:
    """The alarm bar per step: the lowest `median + k x spread` of the score the memory
    has managed so far, over a trailing `window_s`.

    Causal -- each step sees only the past -- and one-way: the bar tightens while the
    memory settles on the path and never loosens again, so a deviation cannot raise the
    bar meant to catch it, and a slow drift away from the path is still caught. Before
    `warmup_s` there is no bar (inf): the memory has not learned the path yet.
    """
    s, t = np.asarray(scores, dtype=float), np.asarray(times, dtype=float)
    out = np.full(len(s), np.inf)
    if len(s) == 0:
        return out
    step = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    n_win = max(4, int(round(window_s / step)))
    bar = np.inf
    for i in range(len(s)):
        if t[i] >= warmup_s and i >= n_win:
            w = s[i - n_win + 1:i + 1]
            w = w[np.isfinite(w)]
            if len(w) >= 4:
                med = float(np.median(w))
                mad = float(np.median(np.abs(w - med))) * 1.4826
                bar = min(bar, max(med + k * mad, floor_px))
        out[i] = bar
    return out


class RatchetAlarm:
    """`ratchet_threshold` one step at a time, for a memory that is running live.

    `update(score, t)` returns True while the deviation is flagged: the score has stood
    above the bar for `hold_n` steps. Same rule and defaults as the batch function, so a
    live run and a scored one agree.
    """

    def __init__(self, k: float = RATCHET_K, window_s: float = 2.0, warmup_s: float = 5.0,
                 floor_px: float = 5.0, hold_n: int = 3, dt_s: float = 0.005):
        self.k, self.warmup_s, self.floor_px, self.hold_n = k, warmup_s, floor_px, hold_n
        self.window = deque(maxlen=max(4, int(round(window_s / dt_s))))
        self.bar, self.above = np.inf, 0

    def update(self, score: float, t: float) -> bool:
        if np.isfinite(score):
            self.window.append(float(score))
        if t >= self.warmup_s and len(self.window) == self.window.maxlen:
            w = np.array(self.window)
            med = float(np.median(w))
            mad = float(np.median(np.abs(w - med))) * 1.4826
            self.bar = min(self.bar, max(med + self.k * mad, self.floor_px))
        self.above = self.above + 1 if (np.isfinite(score) and score > self.bar) else 0
        return self.above >= self.hold_n


def deviation_roc(scores, times, deviation_times, threshold=None,
                  hold_n: int = 3) -> dict:
    """Score the deviation signal as a detector.

    Positives are steps at or after the first break, negatives the steps before it
    (all steps, on a clip without a break). `auc` is threshold-free. At the operating
    `threshold` -- a number, a per-step array, a rule (scores, times) -> per-step array,
    or "ratchet" for `ratchet_threshold`;
    the default, the 99th percentile of the negatives, needs the labels and so is a
    reference rather than a detector -- a flag is `hold_n`
    consecutive steps above it: `latency_s` is the first flag after the break,
    `fp_per_min` the flags raised on negative steps.
    """
    s, t = np.asarray(scores, dtype=float), np.asarray(times, dtype=float)
    t_break = min(deviation_times) if len(deviation_times) else np.inf
    positive = t >= t_break
    if isinstance(threshold, str):
        if threshold != "ratchet":
            raise ValueError(f"unknown threshold rule {threshold!r}")
        threshold = ratchet_threshold(s, t)
    elif callable(threshold):
        threshold = threshold(s, t)
    elif threshold is None:
        threshold = float(np.nanpercentile(s[~positive], 99)) if (~positive).any() else np.inf

    flags = _sustained(s > threshold, hold_n)
    step = float(np.median(np.diff(t))) if len(t) > 1 else np.nan
    neg_minutes = (~positive).sum() * step / 60
    out = {"threshold": float(np.min(threshold)) if np.ndim(threshold) else float(threshold),
           "auc": np.nan, "latency_s": np.nan,
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
