"""Events -> model input: count frames or time surfaces (Stage 1), a centroid track
(baseline / fallback). Time surfaces reuse the main repo's pipeline.time_surface."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from .data import Clip


def windows(clip: Clip, window_us: int):
    """(t_start_us, events) for consecutive windows tiling the clip; the last may be short."""
    ts = clip.events["timestamp"]
    for t0 in range(0, clip.duration_us, window_us):
        lo, hi = np.searchsorted(ts, [t0, t0 + window_us])
        yield t0, clip.events[lo:hi]


def to_frames(clip: Clip, window_us: int, kind: str = "count", downsample: int = 1,
              tau_us: float | None = None):
    """(t_start_us, frame) per window. `count`: ON/OFF event counts, (2, H, W) float32,
    optionally block-summed by `downsample`. `surface`: the main repo's decaying time
    surface (H, W) in [0, 1], queried at the window's end."""
    w, h = clip.meta["resolution"]
    if kind == "count":
        for t0, ev in windows(clip, window_us):
            yield t0, _count_frame(ev, w, h, downsample)
    elif kind == "surface":
        from pipeline.time_surface import TimeSurface

        cfg = SimpleNamespace(time_surface_tau_us=tau_us or window_us, time_surface_split_polarity=False)
        surface = TimeSurface(cfg, (w, h))
        for t0, ev in windows(clip, window_us):
            surface.update(SimpleNamespace(numpy=lambda ev=ev: ev))   # it expects an EventStore
            yield t0, surface.surface(t0 + window_us)
    else:
        raise ValueError(f"unknown frame kind: {kind!r}")


def _count_frame(ev, w: int, h: int, downsample: int) -> np.ndarray:
    frame = np.zeros((2, h, w), dtype=np.float32)
    np.add.at(frame, (ev["polarity"].astype(np.intp), ev["y"].astype(np.intp),
                      ev["x"].astype(np.intp)), 1.0)
    if downsample > 1:
        frame = frame.reshape(2, h // downsample, downsample, w // downsample, downsample).sum(axis=(2, 4))
    return frame


def to_position(clip: Clip, window_us: int, min_events: int = 5, cell_px: int = 16,
                radius_px: float = 24.0):
    """(t_s, x, y) per window, normalised to the sensor; NaN where there is too little.

    The target is the *densest* patch, not the mean of all events: noise events are
    spread over the whole sensor and would drag a mean toward its centre. So: the
    fullest `cell_px` cell, then the mean of the events within `radius_px` of it.
    `t_s` is the window's centre, the instant the window's events best stand for.
    """
    w, h = clip.meta["resolution"]
    for t0, ev in windows(clip, window_us):
        t_s = (t0 + window_us / 2) / 1e6
        if len(ev) < min_events:
            yield t_s, np.nan, np.nan
            continue
        xs, ys = ev["x"].astype(float), ev["y"].astype(float)
        cx, cy = _densest_cell(xs, ys, w, h, cell_px)
        for _ in range(2):                                   # re-centre once: the cell is coarse
            near = np.hypot(xs - cx, ys - cy) <= radius_px
            if near.sum() < min_events:
                break
            cx, cy = xs[near].mean(), ys[near].mean()
        yield (t_s, np.nan, np.nan) if near.sum() < min_events else (t_s, float(cx / w), float(cy / h))


def _densest_cell(xs, ys, w, h, cell_px):
    nx = -(-w // cell_px)
    counts = np.bincount((ys // cell_px).astype(np.intp) * nx + (xs // cell_px).astype(np.intp),
                         minlength=nx * -(-h // cell_px))
    k = int(np.argmax(counts))
    return (k % nx + 0.5) * cell_px, (k // nx + 0.5) * cell_px


def to_raw(clip: Clip, bin_us: int):
    """Fine spatial+temporal spike tensor. Stage 2 input."""
    raise NotImplementedError
