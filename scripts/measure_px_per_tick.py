"""Measure how many pixels the image moves per encoder tick, on Setup 4a clips.

    python scripts/measure_px_per_tick.py --group wall
    python scripts/measure_px_per_tick.py corpus/real/wall/scan_pan_slow_01

The scene is static and the camera only rotates, so every edge shifts together. This
tracks that shift frame-to-frame and regresses it on the encoder track, giving px/tick
directly -- no anchor needed, so it runs on unmarked clips.

Compared against the px/tick the calibration implies (focal length / ticks_per_radian),
the ratio says whether `ticks_per_radian` is right. A ratio of 1 means it is.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import _thesis_path  # noqa: F401,E402  (adds the thesis repo to sys.path)
import numpy as np  # noqa: E402

from trajmem.data import load_recording  # noqa: E402

STEP = 0.1          # between tracked frames
WIN = 0.05          # accumulation window per frame
MAX_LAG = 16        # px of travel allowed between neighbouring frames
MIN_SPAN = 40       # an axis moving less than this many ticks tells us nothing


def hot_pixels(events, resolution, duration_s: float, fraction: float = 0.6):
    """Pixels alight in most 1 s slices: they fire constantly and never move.

    Masking these matters more than ordinary denoising -- a moving edge visits each
    pixel briefly, so any per-pixel count threshold keeps the static noise and
    discards the signal.
    """
    w, h = resolution
    alight = np.zeros((h, w), dtype=int)
    slices = np.arange(0, max(duration_s - 1.0, 1.0), 1.0)
    for s in slices:
        lo = np.searchsorted(events["timestamp"], int(s * 1e6))
        hi = np.searchsorted(events["timestamp"], int((s + 0.05) * 1e6))
        e = events[lo:hi]
        f = np.zeros((h, w), dtype=bool)
        f[e["y"].astype(int), e["x"].astype(int)] = True
        alight += f
    return alight > fraction * len(slices)


def marginal(events, resolution, hot, t: float, axis: int) -> np.ndarray:
    """Event counts collapsed onto one image axis, hot pixels dropped and smoothed."""
    w, h = resolution
    lo = np.searchsorted(events["timestamp"], int(t * 1e6))
    hi = np.searchsorted(events["timestamp"], int((t + WIN) * 1e6))
    e = events[lo:hi]
    xs, ys = e["x"].astype(int), e["y"].astype(int)
    keep = ~hot[ys, xs]
    coord, n = (xs, w) if axis == 0 else (ys, h)
    m = np.bincount(coord[keep], minlength=n).astype(float)
    return np.convolve(m, np.ones(5) / 5, mode="same")


def step_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Sub-pixel lag aligning `a` onto `b`, and the correlation it achieved.

    Scored by normalised cross-correlation over the overlap only: plain correlation
    is biased toward zero lag, and phase correlation locks onto the wrong peak on a
    scene of repeated vertical edges.
    """
    n = len(a)
    scores = {}
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        u = a[:n - lag] if lag >= 0 else a[-lag:]
        v = b[lag:] if lag >= 0 else b[:n + lag]
        u, v = u - u.mean(), v - v.mean()
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        scores[lag] = float(u @ v / (nu * nv)) if nu and nv else -2.0
    best = max(scores, key=scores.get)
    peak = scores[best]
    if best - 1 in scores and best + 1 in scores:
        y0, y1, y2 = scores[best - 1], scores[best], scores[best + 1]
        d = y0 - 2 * y1 + y2
        if d:
            best = best + 0.5 * (y0 - y2) / d
    return best, peak


def track_shift(events, resolution, hot, times, axis: int) -> tuple[np.ndarray, float]:
    """Cumulative image shift along `axis` across `times`, and the mean step quality."""
    mars = [marginal(events, resolution, hot, t, axis) for t in times]
    steps, peaks = [], []
    for i in range(len(mars) - 1):
        s, p = step_shift(mars[i], mars[i + 1])
        steps.append(s)
        peaks.append(p)
    return np.concatenate([[0.0], np.cumsum(steps)]), float(np.mean(peaks))


def fit(shift: np.ndarray, ticks: np.ndarray) -> tuple[float, float]:
    """px per tick and the correlation of the fit."""
    if np.ptp(ticks) < MIN_SPAN:
        return float("nan"), float("nan")
    slope = float(np.polyfit(ticks, shift, 1)[0])
    return slope, float(np.corrcoef(ticks, shift)[0, 1])


def predicted_px_per_tick(intrinsics, ticks_per_radian) -> tuple[float, float]:
    """What the calibration implies: focal length in px divided by ticks per radian."""
    fx, fy = intrinsics.camera_matrix[0, 0], intrinsics.camera_matrix[1, 1]
    tpr_pan, tpr_tilt = ticks_per_radian
    return fx / tpr_pan, fy / tpr_tilt


def measure(path) -> dict:
    """px/tick for one clip, on whichever axes actually moved."""
    from recording.player import MotorTrack
    from recording.recorder import motor_sidecar_path

    from trajmem.data import _ticks_per_radian
    from trajmem.simulate import load_intrinsics

    clip = load_recording(path)
    if not clip.meta["has_motors"]:
        raise ValueError(f"{clip.meta['slug']}: no encoder side-car")

    res = clip.meta["resolution"]
    dur = clip.duration_us / 1e6
    d, slug = pathlib.Path(clip.source), clip.meta["slug"]
    track = MotorTrack.from_csv(motor_sidecar_path(d / f"{slug}.aedat4"))
    t0 = clip.meta["t0_device_us"]

    times = np.arange(0.0, dur - WIN, STEP)
    ticks = np.array([track.position_at(int(t0 + t * 1e6)) or (np.nan, np.nan)
                      for t in times], dtype=float)
    ok = ~np.isnan(ticks).any(axis=1)
    times, ticks = times[ok], ticks[ok]

    hot = hot_pixels(clip.events, res, dur)
    out = {"slug": slug, "group": clip.meta["group"], "is_break": clip.meta["is_break"],
           "hot": int(hot.sum()), "n": len(times)}

    pred_pan, pred_tilt = predicted_px_per_tick(load_intrinsics(),
                                                _ticks_per_radian(d, slug))
    out["pred_pan"], out["pred_tilt"] = pred_pan, pred_tilt

    for axis, name, pred in ((0, "pan", pred_pan), (1, "tilt", pred_tilt)):
        span = float(np.ptp(ticks[:, axis]))
        out[f"{name}_span"] = span
        if span < MIN_SPAN:
            out[f"{name}_px_per_tick"] = out[f"{name}_r"] = float("nan")
            continue
        shift, quality = track_shift(clip.events, res, hot, times, axis)
        slope, r = fit(shift, ticks[:, axis])
        out[f"{name}_px_per_tick"] = slope
        out[f"{name}_r"] = r
        out[f"{name}_quality"] = quality
        out[f"{name}_ratio"] = abs(slope) / abs(pred) if pred else float("nan")
    return out


def _rows(results) -> str:
    head = (f"{'clip':34s} {'pan span':>9s} {'px/tick':>8s} {'r':>7s} {'ratio':>6s}  "
            f"{'tilt span':>9s} {'px/tick':>8s} {'r':>7s} {'ratio':>6s}")
    lines = [head, "-" * len(head)]
    for x in results:
        def cell(name):
            span, slope = x.get(f"{name}_span", 0.0), x.get(f"{name}_px_per_tick")
            if slope is None or np.isnan(slope):
                return f"{span:9.0f} {'-':>8s} {'-':>7s} {'-':>6s}"
            return (f"{span:9.0f} {slope:8.4f} {x[f'{name}_r']:+7.4f} "
                    f"{x[f'{name}_ratio']:6.3f}")
        lines.append(f"{x['slug']:34s} {cell('pan')}  {cell('tilt')}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clip", nargs="?", help="one clip directory or .aedat4")
    p.add_argument("--group", help="corpus group to walk, e.g. wall")
    p.add_argument("--corpus", default="corpus/real")
    p.add_argument("--skip-break", action="store_true",
                   help="deviation clips are hand-disturbed, so their scene shift is "
                        "not pure ego-motion")
    args = p.parse_args()

    if args.clip:
        clips = [pathlib.Path(args.clip)]
    elif args.group:
        clips = sorted((pathlib.Path(args.corpus) / args.group).rglob("*.aedat4"))
    else:
        raise SystemExit("give a clip path or --group")

    results = []
    for i, c in enumerate(clips, 1):
        try:
            r = measure(c)
        except Exception as exc:                  # a bad clip should not stop the sweep
            print(f"[{i}/{len(clips)}] {c.stem}: {exc}", file=sys.stderr)
            continue
        if args.skip_break and r["is_break"]:
            continue
        results.append(r)
        print(f"[{i}/{len(clips)}] {r['slug']}", file=sys.stderr)

    print(_rows(results))

    for name in ("pan", "tilt"):
        vals = [x[f"{name}_ratio"] for x in results
                if not x["is_break"] and x.get(f"{name}_r", 0) and
                abs(x.get(f"{name}_r", 0)) > 0.98]
        if vals:
            print(f"\n{name}: ratio over {len(vals)} clean clips with |r|>0.98  "
                  f"mean {np.mean(vals):.3f}  sd {np.std(vals):.3f}  "
                  f"min {min(vals):.3f}  max {max(vals):.3f}")


if __name__ == "__main__":
    main()
