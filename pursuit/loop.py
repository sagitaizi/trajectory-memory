"""The closed eye loop: target positions in, gaze out, with the memory and the reaction between."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .eye import REACTION_GAIN, VISUAL_DELAY_S, Delay, Eye, Reaction
from .velocity import SMOOTHING_S, Rate, Smooth


@runtime_checkable
class PursuitMemory(Protocol):
    def reset(self) -> None: ...
    def observe(self, position, velocity) -> None: ...   # the rebuilt target, one delay old; NaN = unseen
    def velocity(self, ahead_s: float): ...              # eye velocity for `ahead_s` after what was last seen; NaN = none yet
    def surprise(self) -> float: ...                     # how far what was last seen is from what was expected; NaN = no expectation yet
    def period(self) -> float: ...                       # the remembered period, s; NaN = none yet


@dataclass
class PursuitTrace:
    """One run, step by step, in the units of the positions (pixels, seconds)."""
    t: np.ndarray                  # step centres, s
    target: np.ndarray             # (N, 2) where the target was, hidden or not; NaN = localiser saw nothing
    target_velocity: np.ndarray    # (N, 2) its raw rate of change
    gaze: np.ndarray               # (N, 2)
    eye_velocity: np.ndarray       # (N, 2) = memory + reaction
    memory_velocity: np.ndarray    # (N, 2)
    reaction_velocity: np.ndarray  # (N, 2)
    surprise: np.ndarray           # (N,) the memory's; NaN = none
    period: np.ndarray             # (N,) the memory's, s; NaN = none
    delay_s: float
    blank: tuple | None = None     # (t0, t1) where the target was hidden


def follow(memory, txy, dt_s: float, delay_s: float = VISUAL_DELAY_S, reaction_gain: float = REACTION_GAIN,
           smoothing_s: float = SMOOTHING_S, blank=None) -> PursuitTrace:
    """Follow the target positions `txy` (N rows of t, x, y, one per `dt_s`) with the eye.
    `memory` None is the reaction alone. `blank` = (t0, t1) hides the target then."""
    txy = np.asarray(txy, dtype=float)
    seen = txy[:, 1:].copy()
    if blank is not None:
        seen[(txy[:, 0] >= blank[0]) & (txy[:, 0] < blank[1])] = np.nan
    if memory is not None:
        memory.reset()
    visible = seen[np.isfinite(seen).all(axis=1)]
    eye = Eye(dt_s, visible[0] if len(visible) else np.zeros(2))
    late_retina, late_gaze = Delay(delay_s, dt_s), Delay(delay_s, dt_s)
    slip_rate, gaze_rate = Rate(dt_s), Rate(dt_s)
    smooth, reaction = Smooth(dt_s, smoothing_s), Reaction(dt_s, reaction_gain)
    gaze, drives, reactions, surprise, period = [], [], [], [], []
    for p in seen:
        gaze.append(eye.gaze)
        retina, gaze_copy = late_retina.step(p - eye.gaze), late_gaze.step(eye.gaze)
        slip, eye_copy = slip_rate.step(retina), gaze_rate.step(gaze_copy)
        drive = np.zeros(2)
        if memory is not None:
            memory.observe(retina + gaze_copy, smooth.step(slip + eye_copy))
            v = np.asarray(memory.velocity(delay_s), dtype=float)
            if np.isfinite(v).all():
                drive = v
        surprise.append(np.nan if memory is None else memory.surprise())
        period.append(np.nan if memory is None else memory.period())
        react = reaction.step(slip)
        eye.step(drive + react)
        drives.append(drive)
        reactions.append(react)
    drives, reactions = np.array(drives), np.array(reactions)
    velocity = drives + reactions
    return PursuitTrace(t=txy[:, 0], target=txy[:, 1:], target_velocity=np.gradient(txy[:, 1:], dt_s, axis=0),
                        gaze=np.array(gaze), eye_velocity=velocity, memory_velocity=drives,
                        reaction_velocity=reactions, surprise=np.array(surprise, dtype=float),
                        period=np.array(period, dtype=float), delay_s=delay_s, blank=blank)
