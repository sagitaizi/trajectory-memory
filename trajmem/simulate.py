"""v2e wrapper: TrajectorySpec + camera config -> synthetic event stream + exact ground truth.

v2e is a source clone at D:\\Projects\\v2e used via sys.path (see materials/03-tooling).
The EventEmulator Python API is the integration point; construct it with device="cpu".
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .data import Clip
from .trajectories import TrajectorySpec, sample

# The one event dtype the rest of the package sees. dv.EventStore.numpy() orders its
# fields differently, so data.load_recording restacks into this rather than reusing it.
EVENT_DTYPE = np.dtype([("x", "<i2"), ("y", "<i2"), ("timestamp", "<i8"), ("polarity", "i1")])

_V2E_REPO = Path(os.environ.get("V2E_REPO", Path(__file__).resolve().parents[2] / "v2e"))


def render_frames(spec: TrajectorySpec, cfg, intrinsics=None) -> np.ndarray:
    """Sampled path -> frame stack (T, H, W) float32: a blob on the path at cfg['fps'].

    The analytic path lives in an ideal pinhole image of size cfg['resolution'];
    with `intrinsics` the blob is drawn at its lens-distorted pixel instead, so the
    stack resembles the real camera while ground truth stays the undistorted spec.
    """
    import cv2

    w, h = cfg["resolution"]
    if intrinsics is not None and (w, h) != (intrinsics.width, intrinsics.height):
        raise ValueError(
            f"resolution {(w, h)} is not the calibration's {(intrinsics.width, intrinsics.height)}; "
            "distorting into it would draw the blob off the ground-truth path")
    fps = cfg["fps"]
    n = int(round(cfg["duration_s"] * fps))
    times = np.arange(n) / fps
    blob = cfg["blob"]

    px = _path_pixels(spec, times, (w, h), intrinsics)
    frames = np.full((n, h, w), float(blob["bg_intensity"]), dtype=np.float32)
    radius = int(blob["radius_px"])
    fg = float(blob["fg_intensity"])
    for i in range(n):
        cv2.circle(frames[i], (round(px[i, 0]), round(px[i, 1])), radius, fg, thickness=-1)
    return frames


def load_intrinsics(camera_cfg=None):
    """DVXplorer camera intrinsics + distortion, from the main repo's calibration.

    Defaults to `camera.calibration`'s own pinned file (the session-3 refit); pass
    `camera_cfg={"calibration": <abs path>}` only to override it.
    """
    from camera.calibration import Intrinsics, load_calibration

    override = (camera_cfg or {}).get("calibration")
    path = Path(override) if override and Path(override).is_absolute() else None
    return Intrinsics(load_calibration(path))


def _path_pixels(spec: TrajectorySpec, times, resolution, intrinsics=None) -> np.ndarray:
    """Sampled normalised path -> (N, 2) pixel coordinates, lens-distorted if intrinsics given."""
    w, h = resolution
    xy01 = np.atleast_2d(sample(spec, np.asarray(times, dtype=float)))
    px = xy01 * np.array([w, h], dtype=float)
    if intrinsics is None:
        return px
    from camera.calibration import distort_normalized

    norm = np.column_stack([(px[:, 0] - intrinsics.cx) / intrinsics.fx,
                            (px[:, 1] - intrinsics.cy) / intrinsics.fy])
    return distort_normalized(intrinsics, norm)


_V2E_KEYS = ("pos_thres", "neg_thres", "sigma_thres", "cutoff_hz", "leak_rate_hz",
             "refractory_period_s", "shot_noise_rate_hz")


def _run_v2e(frames: np.ndarray, times_s, v2e_cfg, seed: int = 0) -> np.ndarray:
    """Luminance frame stack -> events (EVENT_DTYPE) via v2ecore.emulator.EventEmulator."""
    if not _V2E_REPO.is_dir():
        raise RuntimeError(f"v2e source not found at {_V2E_REPO} (set V2E_REPO)")
    import sys

    if str(_V2E_REPO) not in sys.path:
        sys.path.insert(0, str(_V2E_REPO))
    from v2ecore.emulator import EventEmulator

    params = {k: v2e_cfg[k] for k in _V2E_KEYS if k in v2e_cfg}
    emu = EventEmulator(seed=seed, output_folder=None, dvs_h5=None, dvs_aedat2=None,
                        dvs_text=None, device="cpu", **params)

    chunks = []
    for frame, t in zip(frames, times_s):
        ev = emu.generate_events(np.ascontiguousarray(frame, dtype=np.float32), float(t))
        if ev is not None and len(ev):
            chunks.append(np.asarray(ev))
    if not chunks:
        return np.empty(0, dtype=EVENT_DTYPE)

    raw = np.concatenate(chunks, axis=0)                 # (M, 4): [t_s, x, y, polarity]
    out = np.empty(len(raw), dtype=EVENT_DTYPE)
    out["x"] = np.rint(raw[:, 1]).astype(np.int16)
    out["y"] = np.rint(raw[:, 2]).astype(np.int16)
    out["timestamp"] = np.rint(raw[:, 0] * 1e6).astype(np.int64)
    out["polarity"] = (raw[:, 3] > 0).astype(np.int8)
    return out


def simulate(spec: TrajectorySpec, camera_cfg, sim_cfg, seed: int = 0) -> Clip:
    """Render `spec` through the DVXplorer model and return a Clip with exact ground truth."""
    intrinsics = load_intrinsics(camera_cfg) if camera_cfg is not None else None
    frames = render_frames(spec, sim_cfg, intrinsics=intrinsics)
    times = np.arange(len(frames)) / sim_cfg["fps"]
    events = _run_v2e(frames, times, sim_cfg["v2e"], seed=seed)
    return Clip(
        events=events,
        duration_us=int(round(sim_cfg["duration_s"] * 1e6)),
        gt=lambda t: sample(spec, t),
        deviation_times=[d.at_t for d in spec.deviations],
        source="sim",
        meta={"spec": spec, "seed": seed, "resolution": tuple(sim_cfg["resolution"])},
    )
