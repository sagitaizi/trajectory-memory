"""The simulated eye: the visual delay, the reaction to slip, and the gaze it moves.

Works in any one image unit; the loop uses pixels.
"""
from __future__ import annotations

from collections import deque

import numpy as np

VISUAL_DELAY_S = 0.1
# The strongest reaction that does not overshoot at a 100 ms delay (gain x delay = 1/e),
# until the eye's measured first response gives the number.
REACTION_GAIN = 1.0 / (np.e * VISUAL_DELAY_S)


class Delay:
    """What went in `delay_s` ago; NaN until then."""

    def __init__(self, delay_s: float, dt_s: float):
        self.steps = round(delay_s / dt_s)
        if abs(self.steps * dt_s - delay_s) > 1e-9:
            raise ValueError(f"a delay of {delay_s} s is not a whole number of {dt_s} s steps")
        self.held = deque()

    def step(self, x) -> np.ndarray:
        self.held.append(np.array(x, dtype=float))
        if len(self.held) <= self.steps:
            return np.full_like(self.held[-1], np.nan)
        return self.held.popleft()


class Reaction:
    """Eye acceleration in proportion to the (late) retinal slip. An unseen target changes
    nothing: the speed is kept."""

    def __init__(self, dt_s: float, gain: float = REACTION_GAIN):
        self.dt_s, self.gain = dt_s, gain
        self.speed = np.zeros(2)

    def step(self, slip) -> np.ndarray:
        if np.isfinite(slip).all():
            self.speed = self.speed + self.gain * np.asarray(slip) * self.dt_s
        return self.speed


class Eye:
    """A gaze point that moves at the commanded speed at once."""

    def __init__(self, dt_s: float, gaze):
        self.dt_s, self.gaze = dt_s, np.array(gaze, dtype=float)

    def step(self, velocity) -> None:
        self.gaze = self.gaze + np.asarray(velocity) * self.dt_s
