"""Events -> model input. Reuses the main repo's pipeline.accumulator / pipeline.time_surface."""
from __future__ import annotations

from .data import Clip


def to_frames(clip: Clip, window_us: int):
    """Accumulated event frames / time surfaces. Stage 1 input."""
    raise NotImplementedError


def to_raw(clip: Clip, bin_us: int):
    """Fine spatial+temporal spike tensor. Stage 2 input."""
    raise NotImplementedError


def to_position(clip: Clip, window_us: int):
    """Classical centroid per window: iterator of (t, x, y). Baseline / fallback."""
    raise NotImplementedError
