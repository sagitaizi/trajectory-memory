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


class LiveFrames:
    """`to_frames(kind="count")` for events arriving in chunks from a camera: time counts
    from the first event, as a loaded recording's does, and a window is handed out once an
    event past its end has come. The stuck-pixel cut can only see the past here: the rate
    over every event so far, refreshed each second, from 5 s on."""

    def __init__(self, resolution, window_us: int, downsample: int = 1, max_rate: float = HOT_PIXEL_RATE,
                 refresh_us: int = 1_000_000):
        self.w, self.h = resolution
        if self.w % downsample or self.h % downsample:
            raise ValueError(f"downsample {downsample} must divide the sensor {self.w}x{self.h}")
        self.window_us, self.downsample, self.max_rate, self.refresh_us = window_us, downsample, max_rate, refresh_us
        self.origin_us = None
        self.pending = None
        self.next_k = 0
        self.counts, self.seen = np.zeros(self.w * self.h, np.int64), []
        self.hot = np.zeros(self.w * self.h, bool)
        self.hot_due_us = 5_000_000

    def push(self, events) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """(t_start_us, raw window events, count frame) for every window this chunk closes."""
        if len(events) == 0:
            return []
        if self.origin_us is None:
            self.origin_us = int(events["timestamp"][0])
        ev = events.copy()
        ev["timestamp"] -= self.origin_us
        self.pending = ev if self.pending is None else np.concatenate([self.pending, ev])
        last_k = int(self.pending["timestamp"][-1]) // self.window_us
        starts = np.arange(self.next_k, last_k + 1) * self.window_us
        edges = np.searchsorted(np.ascontiguousarray(self.pending["timestamp"]), starts)
        closed = self.pending[:edges[-1]]
        idx = closed["y"].astype(np.intp) * self.w + closed["x"].astype(np.intp)
        keep = ~self.hot[idx]                                 # the cut as it stood before this chunk
        out = []
        for t0, lo, hi in zip(starts[:-1], edges[:-1], edges[1:]):
            win = self.pending[lo:hi]
            out.append((int(t0), win, _count_frame(win[keep[lo:hi]], self.w, self.h, self.downsample)))
        self._update_hot(idx, last_k * self.window_us)
        self.pending = self.pending[edges[-1]:]
        self.next_k = last_k
        return out

    def _update_hot(self, idx, t_us: int) -> None:
        """Pixel counts are summed only at a refresh, the one time they are read."""
        if self.max_rate <= 0 or len(idx) == 0:
            return
        self.seen.append(idx)
        if t_us >= self.hot_due_us:
            self.counts += np.bincount(np.concatenate(self.seen), minlength=self.w * self.h)
            self.seen = []
            self.hot = self.counts / (t_us / 1e6) > self.max_rate
            self.hot_due_us = t_us + self.refresh_us


def _count_frame(ev, w: int, h: int, downsample: int) -> np.ndarray:
    """Counts straight into the downsampled grid: one bincount over flat cell indices
    (np.add.at on the full-resolution frame was the pipeline's slowest step)."""
    hh, ww = h // downsample, w // downsample
    idx = (ev["polarity"].astype(np.intp) * (hh * ww)
           + (ev["y"].astype(np.intp) // downsample) * ww
           + (ev["x"].astype(np.intp) // downsample))
    return np.bincount(idx, minlength=2 * hh * ww).astype(np.float32).reshape(2, hh, ww)


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
