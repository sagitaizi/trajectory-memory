"""Track-level augmentation for pretraining: the same paths mirrored, moved, resized and
replayed at another tempo. Positions are normalised, so every transform is a few lines
and the measured and true tracks stay consistent. Each function returns new Tracks only.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .data import Track


def flips(tracks: list[Track]) -> list[Track]:
    """Mirrored left-right, top-bottom, and both."""
    out = []
    for tr in tracks:
        for fx, fy in ((-1, 1), (1, -1), (-1, -1)):
            f = np.array([fx, fy], dtype=float)
            o = np.where(f < 0, 1.0, 0.0)
            out.append(replace(tr, name=f"{tr.name}_flip{fx:+d}{fy:+d}", obs=o + f * tr.obs, gt=o + f * tr.gt))
    return out


def _span(gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(gt).all(axis=1)
    return gt[ok].min(axis=0), gt[ok].max(axis=0)


def shifts(tracks: list[Track], n: int, rng: np.random.Generator, margin: float = 0.05) -> list[Track]:
    """`n` copies per track, each moved by a random offset that keeps the true path inside
    the image with `margin` to spare."""
    out = []
    for tr in tracks:
        lo, hi = _span(tr.gt)
        for i in range(n):
            d = rng.uniform(margin - lo, 1.0 - margin - hi)
            out.append(replace(tr, name=f"{tr.name}_shift{i}", obs=tr.obs + d, gt=tr.gt + d))
    return out


def scales(tracks: list[Track], n: int, rng: np.random.Generator, lo: float = 0.7, hi: float = 1.3,
           margin: float = 0.05) -> list[Track]:
    """`n` copies per track, resized about the path's centre; the factor is capped so the
    path stays inside the image."""
    out = []
    for tr in tracks:
        a, b = _span(tr.gt)
        c = (a + b) / 2
        room = np.min(np.minimum(c - margin, 1.0 - margin - c) / np.maximum((b - a) / 2, 1e-6))
        for i in range(n):
            s = min(rng.uniform(lo, hi), room)
            out.append(replace(tr, name=f"{tr.name}_scale{i}", obs=c + s * (tr.obs - c), gt=c + s * (tr.gt - c)))
    return out


def stretches(tracks: list[Track], n: int, rng: np.random.Generator, lo: float = 0.8, hi: float = 1.25) -> list[Track]:
    """`n` copies per track replayed at a random speed factor: the same time grid samples
    the original at `speed * t`, so the period and the break time scale by 1/speed. Steps
    past the original's end (speed > 1) are unseen (NaN)."""
    out = []
    for tr in tracks:
        t0 = tr.t - tr.t[0]
        for i in range(n):
            s = rng.uniform(lo, hi)
            src = np.clip(np.searchsorted(t0, s * t0), 0, len(t0) - 1)     # nearest earlier sample
            inside = s * t0 <= t0[-1]
            obs, gt = tr.obs[src].copy(), tr.gt[src].copy()
            obs[~inside], gt[~inside] = np.nan, np.nan
            out.append(replace(tr, name=f"{tr.name}_speed{i}", obs=obs, gt=gt, period_s=tr.period_s / s,
                               deviation_t=tr.deviation_t / s))
    return out


def augment(tracks: list[Track], spec: str, seed: int = 0) -> list[Track]:
    """`spec` like "flips,shift:2,scale:2,stretch:2": the originals plus each named
    transform applied to them (not compounded)."""
    rng = np.random.default_rng(seed)
    out = list(tracks)
    for item in filter(None, (s.strip() for s in spec.split(","))):
        kind, _, count = item.partition(":")
        n = int(count) if count else 1
        if kind == "flips":
            out += flips(tracks)
        elif kind == "shift":
            out += shifts(tracks, n, rng)
        elif kind == "scale":
            out += scales(tracks, n, rng)
        elif kind == "stretch":
            out += stretches(tracks, n, rng)
        else:
            raise ValueError(f"unknown augmentation {kind!r}; one of flips, shift, scale, stretch")
    return out
