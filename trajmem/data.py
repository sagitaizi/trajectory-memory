"""Load a clip (recorded .aedat4 or simulated) into a uniform Clip object."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Clip:
    events: np.ndarray                                   # structured (x, y, t, polarity)
    duration_us: int
    gt: Callable[[float], tuple[float, float]] | None    # true position; None if unlabelled
    deviation_times: list[float] = field(default_factory=list)
    source: str = "sim"                                  # "sim" or a recording path
    meta: dict = field(default_factory=dict)


def load_recording(path) -> Clip:
    raise NotImplementedError


def load_sim(spec) -> Clip:
    raise NotImplementedError


def attach_labels(clip: Clip, points) -> Clip:
    """Interpolate sparse (t, x, y) hand-labels into clip.gt."""
    raise NotImplementedError
