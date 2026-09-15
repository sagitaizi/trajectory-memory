"""Load a clip (recorded .aedat4 or simulated) into a uniform Clip object."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np

# Motors slew to the sweep's start position before the scan begins; measured across
# all 24 side-cars the transient is confined to the first ~0.2 s.
MOTOR_TRIM_S = 0.5

# Encoder-to-camera axis signs, both measured 2026-09-10 by cross-correlating the target's
# marginal against a reference frame and regressing on the encoder track: r >= +0.95 over
# ten clip/axis pairs, including both diagonals, which move on both axes at once. PAN_SIGN
# was +1.0 and predicted the exact mirror image (r=-0.97); TILT_SIGN was already right.
PAN_SIGN, TILT_SIGN = -1.0, 1.0

PARAMS_PATH = Path(__file__).resolve().parent.parent / "params.yaml"


class Anchor(NamedTuple):
    """A marked target pixel and the instant, in seconds from the trimmed start, it holds at."""
    x: float
    y: float
    t_s: float = 0.0


@dataclass
class Clip:
    events: np.ndarray                                   # structured: x, y, timestamp (us), polarity
    duration_us: int
    gt: Callable[[float], tuple[float, float]] | None    # true position; None if unlabelled
    deviation_times: list[float] = field(default_factory=list)
    source: str = "sim"                                  # "sim" or a recording path
    meta: dict = field(default_factory=dict)


# --- clip paths and side-cars ------------------------------------------------

def describe_clip(path) -> dict:
    """Group, slug and deviation flag for a clip, from its path alone.

    Two break-clip conventions coexist in the corpus: a `break/` subdirectory
    (fan, extra) and `_break` in the slug (wall, pendulum, wall_target).
    """
    p = Path(path)
    clip_dir = p.parent if p.suffix == ".aedat4" else p
    slug = p.stem if p.suffix == ".aedat4" else clip_dir.name

    parts = clip_dir.parts
    if "real" in parts:                          # the corpus root, not a "real" above it
        i = len(parts) - 1 - parts[::-1].index("real")
        group, below = parts[i + 1], parts[i + 1:]
    else:
        group, below = clip_dir.parent.name, parts[-2:]
    return {
        "slug": slug,
        "group": group,
        "is_break": "break" in below or "break" in slug,
    }


def _clip_paths(path) -> tuple[Path, str]:
    p = Path(path)
    return (p.parent, p.stem) if p.suffix == ".aedat4" else (p, p.name)


def read_anchor(path):
    """The hand-marked anchor a 4a clip's ground truth is built on, or None.

    By convention the pixel is the **centre of the target**, and `t_s` is the instant
    it holds at, measured from the trimmed start. A side-car without `t_s` predates
    the field and means the trimmed start itself.
    """
    clip_dir, slug = _clip_paths(path)
    sidecar = clip_dir / f"{slug}.anchor.json"
    if not sidecar.exists():
        return None
    d = json.loads(sidecar.read_text())
    return Anchor(float(d["x"]), float(d["y"]), float(d.get("t_s", 0.0)))


def write_anchor(path, x: float, y: float, t_s: float = 0.0) -> Path:
    """Record a marked anchor. See `read_anchor` for which pixel and which instant."""
    clip_dir, slug = _clip_paths(path)
    sidecar = clip_dir / f"{slug}.anchor.json"
    sidecar.write_text(json.dumps({"x": float(x), "y": float(y), "t_s": float(t_s)}))
    return sidecar


def read_labels(path):
    """Sparse hand-labels as (t_s, x, y) rows, or None if the clip has none.

    x and y are **normalised** to the sensor, matching what every `gt` returns.
    """
    clip_dir, slug = _clip_paths(path)
    sidecar = clip_dir / f"{slug}.labels.csv"
    if not sidecar.exists():
        return None
    with open(sidecar, newline="", encoding="utf-8") as fh:
        return [(float(r["t_s"]), float(r["x"]), float(r["y"])) for r in csv.DictReader(fh)]


def write_labels(path, points) -> Path:
    """Record hand-labels, sorted by time. See `read_labels` for units."""
    points = sorted((float(t), float(x), float(y)) for t, x, y in points)
    if not points:
        raise ValueError("no label points")
    clip_dir, slug = _clip_paths(path)
    sidecar = clip_dir / f"{slug}.labels.csv"
    with open(sidecar, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "x", "y"])
        w.writerows(points)
    return sidecar


def read_deviation_times(path) -> list[float]:
    """Break times in seconds from the trimmed start; empty until the label pass."""
    clip_dir, slug = _clip_paths(path)
    sidecar = clip_dir / f"{slug}.deviation.json"
    if not sidecar.exists():
        return []
    return [float(t) for t in json.loads(sidecar.read_text())["times_s"]]


def write_deviation_times(path, times) -> Path:
    """Record break times, sorted. See `read_deviation_times` for the clock they use."""
    times = sorted(float(t) for t in times)
    clip_dir, slug = _clip_paths(path)
    sidecar = clip_dir / f"{slug}.deviation.json"
    sidecar.write_text(json.dumps({"times_s": times}))
    return sidecar


def _ticks_per_radian(clip_dir: Path, slug: str) -> tuple[float, float]:
    """Each clip carries the calibrated encoder scale in its own params side-car."""
    import yaml

    from recording.recorder import params_sidecar_path

    params = yaml.safe_load(params_sidecar_path(clip_dir / f"{slug}.aedat4").read_text())
    return float(params["ticks_per_radian_pan"]), float(params["ticks_per_radian_tilt"])


# --- events ------------------------------------------------------------------

def trim_and_rebase(events: np.ndarray, trim_us: int) -> tuple[np.ndarray, int]:
    """Drop the first `trim_us` of events; restamp the rest from zero.

    Returns the events and the device time that became t=0, which is what the
    motor side-car still has to be queried in.
    """
    if len(events) == 0:
        raise ValueError("no events")
    keep = events[events["timestamp"] >= events["timestamp"][0] + trim_us]
    if len(keep) == 0:
        raise ValueError(f"trim of {trim_us} us leaves no events")

    t0 = int(keep["timestamp"][0])         # boolean indexing above already copied
    keep["timestamp"] -= t0
    return keep, t0


def _read_events(player) -> np.ndarray:
    """Drain a Player into one EVENT_DTYPE array.

    dv hands back (timestamp, x, y, polarity) padded to 16 bytes; EVENT_DTYPE
    orders the fields differently, so this restacks rather than reinterprets —
    without it a recorded clip and a simulated one reach the frontend as
    different dtypes.
    """
    from .simulate import EVENT_DTYPE

    chunks = []
    while player.isRunning():
        batch = player.getNextEventBatch()
        if batch is not None and len(batch):
            chunks.append(batch.numpy())
    if not chunks:
        raise ValueError("recording has no events")

    raw = np.concatenate(chunks)
    out = np.empty(len(raw), dtype=EVENT_DTYPE)
    for name in ("x", "y", "timestamp", "polarity"):
        out[name] = raw[name]
    return out


def slice_clip(clip: Clip, t0_s: float, t1_s: float) -> Clip:
    """The clip between t0 and t1 as a clip of its own: events rebased to zero, ground
    truth and break times shifted. For a segment of a clip (`loop_break_01` is two
    repetitive motions around a break) or to drop a settling transient.
    """
    if t1_s <= t0_s:
        raise ValueError("window must run forwards")
    lo, hi = int(t0_s * 1e6), int(t1_s * 1e6)
    keep = clip.events[(clip.events["timestamp"] >= lo) & (clip.events["timestamp"] < hi)]
    if len(keep) == 0:
        raise ValueError(f"no events in [{t0_s}, {t1_s}) s")
    keep = keep.copy()
    keep["timestamp"] -= lo

    whole_gt = clip.gt
    gt = None if whole_gt is None else (lambda t: whole_gt(np.asarray(t, dtype=float) + t0_s))
    return replace(
        clip, events=keep, duration_us=hi - lo, gt=gt,
        deviation_times=[t - t0_s for t in clip.deviation_times if t0_s <= t < t1_s],
        meta={**clip.meta, "window_s": (float(t0_s), float(t1_s))},
    )


# --- ground truth ------------------------------------------------------------

def label_gt(points, max_gap_s: float | None = None) -> Callable:
    """Sparse (t_s, x, y) marks -> gt(t): linear between marks, NaN where unseen.

    A gap wider than `max_gap_s` means the target was not visible there -- the
    instants skipped in the marking tool -- so gt returns NaN inside it rather than
    a straight line through where the target was not. The same holds beyond the
    first and last marks. Default: three labelling intervals, the interval being the
    median spacing between marks. Within that tolerance the end marks are held.
    """
    marks = np.asarray(sorted(points, key=lambda p: p[0]), dtype=float)
    if len(marks) == 0:
        raise ValueError("no label points")
    t_marks, x_marks, y_marks = marks[:, 0], marks[:, 1], marks[:, 2]
    if max_gap_s is None:
        spacing = np.diff(t_marks)
        max_gap_s = 3 * float(np.median(spacing)) if len(spacing) else np.inf

    def gt(t):
        t = np.asarray(t, dtype=float)
        tt = np.atleast_1d(t)
        out = np.stack([np.interp(tt, t_marks, x_marks),
                        np.interp(tt, t_marks, y_marks)], axis=-1)

        i = np.searchsorted(t_marks, tt, side="right")   # t_marks[i-1] <= t < t_marks[i]
        prev = np.where(i > 0, t_marks[np.clip(i - 1, 0, None)], -np.inf)
        nxt = np.where(i < len(t_marks), t_marks[np.clip(i, None, len(t_marks) - 1)], np.inf)
        on_a_mark = (tt == prev) | (tt == nxt)
        gap = np.where(np.isfinite(prev) & np.isfinite(nxt), nxt - prev,
                       np.minimum(tt - prev, nxt - tt))      # distance past an end
        out[(gap > max_gap_s) & ~on_a_mark] = np.nan
        return out[0] if t.ndim == 0 else out

    return gt


def attach_labels(clip: Clip, points) -> Clip:
    """Interpolate sparse (t_s, x, y) hand-labels into clip.gt. Returns a new Clip."""
    gt = label_gt(points)
    return replace(clip, gt=gt, meta={**clip.meta, "labels": [tuple(p) for p in points]})


def ego_gt(anchor_px, motors, intrinsics, t0_us: int, ticks_per_radian,
           anchor_t_s: float = 0.0) -> Callable:
    """Where a *static* target appears as the camera pans and tilts (Setup 4a).

    The anchor pixel is the target at `anchor_t_s`, not necessarily at t=0: the marker
    shows a window of accumulated events and a click means that window's centre.

    The target never moves; the rig does. So the anchor pixel becomes a ray, the
    ray is carried into each later camera frame by the encoders' rotation, and is
    reprojected through the lens — which is why this is not a linear pixel offset.

    Axis mapping and sign follow `pipeline.egomotion`, where both are recorded as
    assumed and unverified; `PAN_SIGN` / `TILT_SIGN` exist so one measurement
    against a real clip can settle them without touching the geometry.
    """
    from camera.calibration import distort_normalized, undistort_normalized

    tpr_pan, tpr_tilt = ticks_per_radian
    anchor_us = int(t0_us + anchor_t_s * 1e6)
    start = motors.position_at(anchor_us)
    if start is None:
        raise ValueError(f"no encoder sample at or before the anchor ({anchor_us})")
    pan0, tilt0 = start

    a = undistort_normalized(intrinsics, np.asarray([anchor_px], dtype=float))[0]
    ray0 = np.array([a[0], a[1], 1.0])

    def gt(t):
        t = np.asarray(t, dtype=float)
        times = np.atleast_1d(t)
        ticks = np.array([motors.position_at(int(t0_us + ti * 1e6)) or (pan0, tilt0)
                          for ti in times], dtype=float)
        yaw = PAN_SIGN * (ticks[:, 0] - pan0) / tpr_pan
        pitch = TILT_SIGN * (ticks[:, 1] - tilt0) / tpr_tilt

        rays = np.einsum("nji,j->ni", _camera_rotation(yaw, pitch), ray0)
        norm = rays[:, :2] / rays[:, 2:3]
        px = distort_normalized(intrinsics, norm)
        out = px / np.array([intrinsics.width, intrinsics.height], dtype=float)
        return out[0] if t.ndim == 0 else out

    return gt


def _camera_rotation(yaw, pitch) -> np.ndarray:
    """(N, 3, 3) camera rotations: pitch about x, then yaw about y."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    zero, one = np.zeros_like(cy), np.ones_like(cy)
    r_y = np.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy], axis=-1)
    r_x = np.stack([one, zero, zero, zero, cp, -sp, zero, sp, cp], axis=-1)
    return r_y.reshape(-1, 3, 3) @ r_x.reshape(-1, 3, 3)


def _ego_gt_for_clip(clip_dir: Path, slug: str, anchor, t0_us: int) -> Callable:
    """Encoder ground truth for a recorded 4a clip, rebuilt from its side-cars alone."""
    from recording.player import MotorTrack
    from recording.recorder import motor_sidecar_path

    from .simulate import load_intrinsics

    anchor = Anchor(*anchor)
    track = MotorTrack.from_csv(motor_sidecar_path(clip_dir / f"{slug}.aedat4"))
    return ego_gt((anchor.x, anchor.y), track, load_intrinsics(), t0_us,
                  _ticks_per_radian(clip_dir, slug), anchor_t_s=anchor.t_s)


# --- loading -----------------------------------------------------------------

def load_recording(path, anchor=None) -> Clip:
    """Read a recorded clip into a Clip, trimmed, restamped from zero.

    Ground truth comes from the encoders for motor-swept clips once an anchor is
    marked (Setup 4a), from hand-labels where they exist, and is otherwise None.
    """
    from recording.player import Player

    clip_dir, slug = _clip_paths(path)
    aedat4 = clip_dir / f"{slug}.aedat4"
    player = Player(aedat4)

    events = _read_events(player)
    has_motors = not player.motors.is_empty
    events, t0 = trim_and_rebase(events, int(MOTOR_TRIM_S * 1e6) if has_motors else 0)

    meta = describe_clip(aedat4)
    meta["t0_device_us"] = t0
    meta["has_motors"] = has_motors
    meta["resolution"] = tuple(player.getEventResolution())

    gt = None
    forced_anchor = anchor is not None
    anchor = Anchor(*anchor) if forced_anchor else read_anchor(aedat4)
    labels = read_labels(aedat4)
    # Hand-labels win over a side-car anchor: they are where the target was seen to be,
    # while the anchor is that plus a model of the rig. Pass anchor= to force the model.
    if has_motors and anchor is not None and (forced_anchor or not labels):
        gt = _ego_gt_for_clip(clip_dir, slug, anchor, t0)
        meta["anchor"] = tuple(anchor)
    elif labels:
        gt = label_gt(labels)
        meta["labels"] = labels

    return Clip(
        events=events,
        duration_us=int(events["timestamp"][-1]),
        gt=gt,
        deviation_times=read_deviation_times(aedat4),
        source=str(clip_dir),
        meta=meta,
    )


def sim_params(path=None) -> dict:
    """The project's `sim` block from params.yaml."""
    import yaml

    return yaml.safe_load(Path(path or PARAMS_PATH).read_text())["sim"]


def load_sim(spec, sim_cfg=None, seed: int = 0, distort: bool = True) -> Clip:
    """Generate a simulated clip from a trajectory spec. See `load_clip` for cached ones.

    `distort=False` renders an ideal pinhole image, which is the only way to use a
    resolution the calibration was not fitted at.
    """
    from .simulate import simulate

    sim_cfg = sim_cfg or sim_params()
    camera_cfg = {"calibration": sim_cfg.get("calibration")} if distort else None
    return simulate(spec, camera_cfg=camera_cfg, sim_cfg=sim_cfg, seed=seed)


# --- caching -----------------------------------------------------------------

def save_clip(clip: Clip, path) -> Path:
    """Cache a clip to .npz. `gt` is stored as whatever generates it, not sampled."""
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_name(path.name + ".npz")     # np.savez appends it; agree with it
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, events=clip.events, duration_us=clip.duration_us,
             deviation_times=np.asarray(clip.deviation_times, dtype=float),
             source=clip.source, meta=np.array(clip.meta, dtype=object))
    return path


def load_clip(path) -> Clip:
    """Read back a `save_clip` .npz, rebuilding `gt` from the spec or labels in meta."""
    with np.load(path, allow_pickle=True) as z:
        meta = z["meta"].item()
        clip = Clip(
            events=z["events"],
            duration_us=int(z["duration_us"]),
            gt=None,
            deviation_times=[float(t) for t in z["deviation_times"]],
            source=str(z["source"]),
            meta=meta,
        )
    if "spec" in meta:
        from .simulate import apparent_gt, load_intrinsics
        from .trajectories import sample

        spec = meta["spec"]
        if meta.get("distorted"):
            clip.gt = apparent_gt(spec, meta["resolution"], load_intrinsics())
        else:
            clip.gt = lambda t: sample(spec, t)
    elif "labels" in meta:
        clip.gt = label_gt(meta["labels"])
    elif "anchor" in meta:
        clip.gt = _ego_gt_for_clip(Path(clip.source), meta["slug"], meta["anchor"],
                                   meta["t0_device_us"])
    return clip
