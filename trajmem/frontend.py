"""Events -> model input: count frames or time surfaces (Stage 1), a centroid track
(baseline / fallback). Time surfaces reuse the main repo's pipeline.time_surface."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from .data import Clip


HOT_PIXEL_RATE = 100.0            # events/s over the clip; a target passing a pixel leaves a few per lap


def without_hot_pixels(clip: Clip, max_rate: float = HOT_PIXEL_RATE) -> Clip:
    """The clip minus the events of pixels that fire faster than `max_rate` over its whole
    length -- stuck pixels, which on the noisier recordings carry a fifth to a third of all
    events and form dense clusters a centroid mistakes for the target."""
    if max_rate <= 0 or len(clip.events) == 0 or clip.duration_us < 5e6:   # a rate needs a few seconds
        return clip
    w, h = clip.meta["resolution"]
    idx = clip.events["y"].astype(np.int64) * w + clip.events["x"].astype(np.int64)
    rate = np.bincount(idx, minlength=w * h) / max(clip.duration_us / 1e6, 1e-6)
    hot = rate > max_rate
    if not hot.any():
        return clip
    return replace(clip, events=clip.events[~hot[idx]], meta={**clip.meta, "hot_pixels": int(hot.sum())})


def windows(clip: Clip, window_us: int):
    """(t_start_us, events) for consecutive windows tiling the clip; the last may be short."""
    starts = np.arange(0, clip.duration_us, window_us)
    # one search over a contiguous copy: the field view is strided, and searching it
    # per window would copy every timestamp every time
    edges = np.searchsorted(np.ascontiguousarray(clip.events["timestamp"]),
                            np.append(starts, starts[-1] + window_us))
    for t0, lo, hi in zip(starts, edges[:-1], edges[1:]):
        yield int(t0), clip.events[lo:hi]


def to_frames(clip: Clip, window_us: int, kind: str = "count", downsample: int = 1,
              tau_us: float | None = None):
    """(t_start_us, frame) per window. `count`: ON/OFF event counts, (2, H, W) float32,
    optionally block-summed by `downsample`. `surface`: the main repo's decaying time
    surface (H, W) in [0, 1], queried at the window's end."""
    w, h = clip.meta["resolution"]
    if kind == "count" and (w % downsample or h % downsample):
        raise ValueError(f"downsample {downsample} must divide the sensor {w}x{h}")
    if kind == "count":
        for t0, ev in windows(without_hot_pixels(clip), window_us):
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
                frac: float = 0.3, hot_pixel_rate: float = HOT_PIXEL_RATE):
    """(t_s, x, y) per window, normalised to the sensor; NaN where there is too little.

    The target is where events are *dense*, not the mean of all of them: noise is
    spread over the whole sensor and would drag a mean toward its middle, and a
    string above a hanging target is long but thin. So: count events per `cell_px`
    cell, keep the cells holding at least `frac` of the fullest one, and average the
    events in those. `t_s` is the window's centre, the instant its events stand for.
    """
    w, h = clip.meta["resolution"]
    nx, ny = -(-w // cell_px), -(-h // cell_px)
    clip = without_hot_pixels(clip, hot_pixel_rate)
    for t0, ev in windows(clip, window_us):
        t_s = (t0 + window_us / 2) / 1e6
        if len(ev) < min_events:
            yield t_s, np.nan, np.nan
            continue
        cx, cy = (ev["x"] // cell_px).astype(np.intp), (ev["y"] // cell_px).astype(np.intp)
        counts = np.bincount(cy * nx + cx, minlength=nx * ny)
        keep = (counts >= frac * counts.max())[cy * nx + cx]
        yield t_s, float(ev["x"][keep].mean() / w), float(ev["y"][keep].mean() / h)


def to_raw(clip: Clip, bin_us: int):
    """Fine spatial+temporal spike tensor. Stage 2 input."""
    raise NotImplementedError
