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
class Wobble:
    """Small, smooth imperfections a real repetitive motion has. All zero = perfect."""
    amp_depth: float = 0.0                    # size x (1 + depth sin(2 pi t / amp_period)): beating
    amp_period_s: float = 1.0
    drift: tuple[float, float] = (0.0, 0.0)   # centre wanders by (dx sin, dy cos)(2 pi t / drift_period)
    drift_period_s: float = 1.0
    phase_depth: float = 0.0                  # phase + depth sin(2 pi t / phase_period): period jitter
    phase_period_s: float = 1.0
    growth: float = 0.0                       # size x (1 + growth t): slow gain or decay


@dataclass
class TrajectorySpec:
    shape: Shape
    size: tuple[float, float]          # circle: (r, r); ellipse/figure8: (a, b); lissajous: (A, B)
    period_s: float
    center: tuple[float, float] = (0.5, 0.5)   # normalised image coords
    phase0: float = 0.0
    lissajous: tuple[float, float, float] = (3.0, 2.0, np.pi / 2)  # (a, b, delta)
    rotation: float = 0.0              # radians, turns the path about its centre
    deviations: list[Deviation] = field(default_factory=list)
    wobble: Wobble | None = None


def sample(spec: TrajectorySpec, t):
    """Position on the path at time(s) t, in normalised image coords. Vectorised over t."""
    t = np.asarray(t, dtype=float)
    for d in spec.deviations:
        if d.kind not in _DEVIATION_KINDS:
            raise ValueError(f"unknown deviation kind: {d.kind!r}")
    phi = _phase(spec, t)
    sx, sy = _size(spec, t)
    cx, cy = _center(spec, t)
    if spec.wobble is not None:
        phi, sx, sy, cx, cy = _wobbled(spec.wobble, t, phi, sx, sy, cx, cy)
    ox, oy = _offset_with_switches(spec, sx, sy, phi, t)
    c, s = np.cos(spec.rotation), np.sin(spec.rotation)
    return np.stack([cx + c * ox - s * oy, cy + s * ox + c * oy], axis=-1)


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


def _wobbled(w: Wobble, t, phi, sx, sy, cx, cy):
    two_pi = 2.0 * np.pi
    phi = phi + w.phase_depth * np.sin(two_pi * t / w.phase_period_s)
    gain = (1.0 + w.amp_depth * np.sin(two_pi * t / w.amp_period_s)) * (1.0 + w.growth * t)
    cx = cx + w.drift[0] * np.sin(two_pi * t / w.drift_period_s)
    cy = cy + w.drift[1] * np.cos(two_pi * t / w.drift_period_s)
    return phi, sx * gain, sy * gain, cx, cy


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


def fit_ellipse(points, period_s: float | None = None) -> TrajectorySpec:
    """Least-squares ellipse through sparse (t, x, y) marks, as a TrajectorySpec.

    Any ellipse traced at constant angular rate is x, y = c + M [cos wt, sin wt];
    the SVD of M splits it into semi-axes, a rotation and a phase. A negative
    second semi-axis means the path runs the other way round. Marks with a NaN
    position are skipped. Without `period_s` the period is searched for.
    """
    marks = np.asarray(points, dtype=float)
    marks = marks[np.all(np.isfinite(marks), axis=1)]
    if len(marks) < 5:
        raise ValueError("need at least five marks with a known position")
    t, xy = marks[:, 0], marks[:, 1:3]
    if period_s is None:
        period_s = _search_period(t, xy)

    coef, _ = _harmonic_fit(t, xy, period_s)
    center = coef[0]
    m = coef[1:].T                                         # rows x, y; columns cos, sin
    u, s, vt = np.linalg.svd(m)
    s = s.copy()
    if np.linalg.det(vt) < 0:
        vt[1] *= -1
        s[1] *= -1
    if np.linalg.det(u) < 0:
        u[:, 1] *= -1
        s[1] *= -1
    return TrajectorySpec(
        shape="ellipse", size=(float(s[0]), float(s[1])), period_s=float(period_s),
        center=(float(center[0]), float(center[1])),
        phase0=float(np.arctan2(vt[1, 0], vt[0, 0])),
        rotation=float(np.arctan2(u[1, 0], u[0, 0])),
    )


def _harmonic_fit(t, xy, period_s):
    """Fit xy(t) = c + A cos wt + B sin wt; returns the (3, 2) coefficients and residual."""
    w = 2.0 * np.pi / period_s
    design = np.column_stack([np.ones_like(t), np.cos(w * t), np.sin(w * t)])
    coef, *_ = np.linalg.lstsq(design, xy, rcond=None)
    return coef, float(np.sum((design @ coef - xy) ** 2))


def _search_period(t, xy) -> float:
    """Period with the smallest harmonic-fit residual: a frequency grid, then a fine pass."""
    span = t.max() - t.min()
    f_lo, f_hi = 1.0 / span, 0.5 / np.median(np.diff(np.sort(t)))
    step = 1.0 / (8.0 * span)
    coarse = np.arange(f_lo, f_hi, step)
    best = coarse[np.argmin([_harmonic_fit(t, xy, 1.0 / f)[1] for f in coarse])]
    fine = np.linspace(best - step, best + step, 201)
    fine = fine[fine > 0]
    return float(1.0 / fine[np.argmin([_harmonic_fit(t, xy, 1.0 / f)[1] for f in fine])])
