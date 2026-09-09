"""Analytic path specs + scripted deviations -> exact position vs time. Pure NumPy."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Shape = Literal["circle", "ellipse", "figure8", "lissajous"]
DeviationKind = Literal["shrink", "speed_change", "drift", "switch_shape"]


@dataclass
class Deviation:
    at_t: float
    kind: DeviationKind
    params: dict = field(default_factory=dict)


@dataclass
class TrajectorySpec:
    shape: Shape
    size: tuple[float, float]          # circle: (r, r); ellipse/figure8: (a, b); lissajous: (A, B)
    period_s: float
    center: tuple[float, float] = (0.5, 0.5)   # normalised image coords
    phase0: float = 0.0
    lissajous: tuple[float, float, float] = (3.0, 2.0, np.pi / 2)  # (a, b, delta)
    deviations: list[Deviation] = field(default_factory=list)


def sample(spec: TrajectorySpec, t):
    """Position on the path at time(s) t, in normalised image coords. Vectorised over t."""
    raise NotImplementedError
