"""Prediction error, lock-on time, deviation-detection ROC + latency."""
from __future__ import annotations


def prediction_error(pred, gt):
    """Mean/median distance between predicted and true position at the horizon."""
    raise NotImplementedError


def lock_on_time(errors, tol: float, dt: float):
    """Time until error stays below tol."""
    raise NotImplementedError


def deviation_roc(scores, times, deviation_times):
    """ROC over a threshold sweep + median detection latency."""
    raise NotImplementedError
