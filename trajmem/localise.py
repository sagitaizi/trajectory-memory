"""Stage-1 localiser harness: frames in, position out, scored against the truth.

`FrameSet` holds one clip's frames (from scripts/make_frames.py or built on the fly);
`evaluate_localiser` streams them through any `model.Localiser` and returns the per-step
track; `score_track` gives the same error numbers as `check_localiser.py` gives the
classical centroid. `FrameCentroid` is the reference localiser: the frame's centre of
mass, no learning. The spiking localiser goes behind the same Protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import Clip
from .frontend import to_frames


@dataclass
class FrameSet:
    """One clip as the localiser sees it: (N, 2, H, W) uint8 ON/OFF counts per window,
    window centres `t` (s), true position `gt` (N, 2, normalised; NaN where unknown)."""
    name: str
    frames: np.ndarray
    t: np.ndarray
    gt: np.ndarray
    window_us: int
    downsample: int


def frame_set(clip: Clip, window_us: int, downsample: int, name: str = "") -> FrameSet:
    ts, frames = [], []
    for t0, frame in to_frames(clip, window_us, kind="count", downsample=downsample):
        ts.append((t0 + window_us / 2) / 1e6)
        frames.append(np.minimum(frame, 255).astype(np.uint8))
    t = np.array(ts)
    gt = np.atleast_2d(clip.gt(t)) if clip.gt is not None else np.full((len(t), 2), np.nan)
    return FrameSet(name, np.stack(frames), t, np.asarray(gt, dtype=float), window_us, downsample)


def load_frame_set(path, window_us: int, downsample: int) -> FrameSet:
    """A make_frames.py .npz."""
    with np.load(path) as z:
        return FrameSet(Path(path).stem, z["frames"], z["t"], z["gt"], window_us, downsample)


def load_frame_sets(corpus, window_us: int = 5000, downsample: int = 8) -> list[FrameSet]:
    d = Path(corpus) / f"frames_{downsample}x_{window_us}us"
    return [load_frame_set(p, window_us, downsample) for p in sorted(d.glob("sim_*.npz"))]


class FrameCentroid:
    """Reference localiser: centre of mass of the frame's counts (both polarities),
    unseen if fewer than `min_events`. No cleverness -- the string pulls it."""

    def __init__(self, min_events: int = 5):
        self.min_events = min_events

    def locate(self, obs) -> tuple[float, float]:
        f = np.asarray(obs, dtype=float).sum(axis=0)
        total = f.sum()
        if total < self.min_events:
            return (np.nan, np.nan)
        h, w = f.shape
        ys, xs = np.mgrid[0:h, 0:w]
        return (float((f * (xs + 0.5)).sum() / total / w), float((f * (ys + 0.5)).sum() / total / h))

    def reset(self) -> None:
        pass


def evaluate_localiser(localiser, frames: FrameSet) -> np.ndarray:
    """(N, 2) normalised positions, one per frame, NaN where the localiser saw nothing."""
    localiser.reset()
    return np.array([localiser.locate(f) for f in frames.frames], dtype=float).reshape(len(frames.t), 2)


def score_track(track: np.ndarray, frames: FrameSet, resolution=(640, 480)) -> dict:
    """Median / p90 distance to the truth (px), the median signed offset, and the unseen
    fraction -- the numbers check_localiser.py reports for the classical centroid."""
    seen = np.isfinite(track).all(axis=1)
    ok = seen & np.isfinite(frames.gt).all(axis=1)
    d = (track[ok] - frames.gt[ok]) * np.asarray(resolution, dtype=float)
    err = np.hypot(*d.T) if ok.any() else np.array([np.nan])
    return {"median": float(np.nanmedian(err)), "p90": float(np.nanpercentile(err, 90)),
            "dx": float(np.median(d[:, 0])) if ok.any() else np.nan,
            "dy": float(np.median(d[:, 1])) if ok.any() else np.nan,
            "unseen": float(1 - seen.mean()), "n": int(ok.sum())}
