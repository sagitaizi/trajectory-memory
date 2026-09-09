"""Analytic path specs + scripted deviations -> exact position vs time. Pure NumPy."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Shape = Literal["circle", "ellipse", "figure8", "lissajous"]
DeviationKind = Literal["shrink", "speed_change", "drift", "switch_shape"]
_DEVIATION_KINDS = {"shrink", "speed_change", "drift", "switch_shape"}


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
    t = np.asarray(t, dtype=float)
    for d in spec.deviations:
        if d.kind not in _DEVIATION_KINDS:
            raise ValueError(f"unknown deviation kind: {d.kind!r}")
    phi = _phase(spec, t)
    sx, sy = _size(spec, t)
    cx, cy = _center(spec, t)
    ox, oy = _offset_with_switches(spec, sx, sy, phi, t)
    return np.stack([cx + ox, cy + oy], axis=-1)


def _phase(spec: TrajectorySpec, t):
    """Accumulated angular phase. Piecewise-constant rate across speed_change events."""
    omega = 2.0 * np.pi / spec.period_s
    changes = sorted((d for d in spec.deviations if d.kind == "speed_change"),
                     key=lambda d: d.at_t)
    if not changes:
        return omega * t + spec.phase0
    bounds = [0.0] + [d.at_t for d in changes]
    rate = omega
    phase = np.full(np.shape(t), spec.phase0, dtype=float)
    for i, lo in enumerate(bounds):
        hi = bounds[i + 1] if i + 1 < len(bounds) else np.inf
        phase = phase + rate * np.clip(np.minimum(t, hi) - lo, 0.0, None)
        if i < len(changes):
            rate = rate * changes[i].params["factor"]
    return phase


def _size(spec: TrajectorySpec, t):
    sx = np.full(np.shape(t), spec.size[0], dtype=float)
    sy = np.full(np.shape(t), spec.size[1], dtype=float)
    for d in spec.deviations:
        if d.kind == "shrink":
            active = t >= d.at_t
            sx = np.where(active, sx * d.params["factor"], sx)
            sy = np.where(active, sy * d.params["factor"], sy)
    return sx, sy


def _center(spec: TrajectorySpec, t):
    cx = np.full(np.shape(t), spec.center[0], dtype=float)
    cy = np.full(np.shape(t), spec.center[1], dtype=float)
    for d in spec.deviations:
        if d.kind == "drift":
            vx, vy = d.params["vel"]
            dt = np.clip(t - d.at_t, 0.0, None)
            cx = cx + vx * dt
            cy = cy + vy * dt
    return cx, cy


def _offset_with_switches(spec: TrajectorySpec, sx, sy, phi, t):
    ox, oy = _shape_offset(spec.shape, sx, sy, phi, spec.lissajous)
    for d in spec.deviations:
        if d.kind == "switch_shape":
            active = t >= d.at_t
            liss = d.params.get("lissajous", spec.lissajous)
            nx, ny = _shape_offset(d.params["to"], sx, sy, phi, liss)
            ox = np.where(active, nx, ox)
            oy = np.where(active, ny, oy)
    return ox, oy


def _shape_offset(shape: Shape, sx, sy, phi, lissajous):
    if shape == "circle":
        return sx * np.cos(phi), sx * np.sin(phi)
    if shape == "ellipse":
        return sx * np.cos(phi), sy * np.sin(phi)
    if shape == "figure8":                                   # lemniscate of Gerono
        return sx * np.sin(phi), sy * np.sin(phi) * np.cos(phi)
    if shape == "lissajous":
        a, b, delta = lissajous
        return sx * np.sin(a * phi + delta), sy * np.sin(b * phi)
    raise ValueError(f"unknown shape: {shape!r}")
