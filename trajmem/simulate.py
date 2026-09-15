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
    """`iter_frames` stacked into (T, H, W). For small clips and tests only: a full
    clip at sensor resolution runs to gigabytes, which is why `simulate` streams."""
    return np.stack(list(iter_frames(spec, cfg, intrinsics)))


def iter_frames(spec: TrajectorySpec, cfg, intrinsics=None):
    """Sampled path -> frames (H, W) float32, one at a time: a blob on the path at cfg['fps'].

    The analytic path lives in an ideal pinhole image of size cfg['resolution'];
    with `intrinsics` the blob is drawn at its lens-distorted pixel instead, so the
    stack resembles the real camera. Ground truth follows it (see `apparent_gt`).

    The blob is an ellipse of semi-axes (radius * aspect, radius) at `angle`, and an
    optional `string` is a line from a fixed pivot to it, drawn underneath; a target
    on a string hangs along it, as the brush targets in the real corpus do. An
    optional `texture` fills the body with a fixed pattern that moves rigidly with
    it, so events fire inside the target too, as they do on a real brush.
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
    bg = float(blob["bg_intensity"])
    radius = int(blob["radius_px"])
    axes = (max(1, round(radius * float(blob.get("aspect", 1.0)))), max(1, radius))
    angle_deg = np.degrees(float(blob.get("angle", 0.0)))
    fg = float(blob["fg_intensity"])
    string = blob.get("string")
    pivot = None if not string else np.array(string["pivot"], dtype=float) * (w, h)
    patch = _texture_patch(blob.get("texture"), axes, fg, bg)
    for i in range(n):
        frame = np.full((h, w), bg, dtype=np.float32)
        centre = (round(px[i, 0]), round(px[i, 1]))
        if pivot is not None:
            cv2.line(frame, (round(pivot[0]), round(pivot[1])), centre,
                     float(string["intensity"]), int(string["thickness_px"]))
            angle_deg = np.degrees(np.arctan2(px[i, 1] - pivot[1], px[i, 0] - pivot[0]))
        if patch is None:
            cv2.ellipse(frame, centre, axes, angle_deg, 0, 360, fg, thickness=-1)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, centre, axes, angle_deg, 0, 360, 1, thickness=-1)
            # getRotationMatrix2D turns the other way from cv2.ellipse's angle
            mid = (patch.shape[0] - 1) / 2                      # odd side: a pixel centre
            m = cv2.getRotationMatrix2D((mid, mid), -angle_deg, 1.0)
            m[:, 2] += (px[i, 0] - mid, px[i, 1] - mid)
            warped = cv2.warpAffine(patch, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=fg)
            frame[mask > 0] = warped[mask > 0]
        yield frame


def _texture_patch(texture, axes, fg: float, bg: float):
    """Blurred noise around `fg`, big enough to cover the ellipse at any angle; or None."""
    if not texture or texture["depth"] <= 0:
        return None
    import cv2

    side = 2 * axes[0] + 5
    rng = np.random.default_rng(texture.get("seed", 0))
    noise = cv2.GaussianBlur(rng.standard_normal((side, side)).astype(np.float32),
                             (0, 0), float(texture["scale_px"]))
    noise /= noise.std()                                       # depth is the relative spread
    patch = fg * (1.0 + float(texture["depth"]) * noise)
    return np.clip(patch, bg + 1.0, 255.0).astype(np.float32)   # never darker than the ground


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


def apparent_gt(spec: TrajectorySpec, resolution, intrinsics):
    """gt(t) as the target *appears* on the sensor: the spec, lens-distorted, normalised.

    Hand-labels on real clips are apparent pixels, so simulated ground truth must be
    too, or a model trained on sim learns a lens-shaped offset (up to ~50 px in the
    DVXplorer's corners) that real labels then count as error.
    """
    scale = np.array(resolution, dtype=float)

    def gt(t):
        t = np.asarray(t, dtype=float)
        out = _path_pixels(spec, np.atleast_1d(t), resolution, intrinsics) / scale
        return out[0] if t.ndim == 0 else out

    return gt


_V2E_KEYS = ("pos_thres", "neg_thres", "sigma_thres", "cutoff_hz", "leak_rate_hz",
             "refractory_period_s", "shot_noise_rate_hz")


def _run_v2e(frames, times_s, v2e_cfg, seed: int = 0) -> np.ndarray:
    """Luminance frames (any iterable) -> events (EVENT_DTYPE) via v2ecore.emulator.EventEmulator."""
    if not _V2E_REPO.is_dir():
        raise RuntimeError(f"v2e source not found at {_V2E_REPO} (set V2E_REPO)")
    import sys

    if str(_V2E_REPO) not in sys.path:
        sys.path.insert(0, str(_V2E_REPO))
    from v2ecore.emulator import EventEmulator

    params = {k: v2e_cfg[k] for k in _V2E_KEYS if k in v2e_cfg}
    # v2e takes seed 0 to mean "unseeded", so shift every seed off it
    emu = EventEmulator(seed=seed + 1, output_folder=None, dvs_h5=None, dvs_aedat2=None,
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
    n = int(round(sim_cfg["duration_s"] * sim_cfg["fps"]))
    times = np.arange(n) / sim_cfg["fps"]
    events = _run_v2e(iter_frames(spec, sim_cfg, intrinsics=intrinsics), times,
                      sim_cfg["v2e"], seed=seed)
    resolution = tuple(sim_cfg["resolution"])
    return Clip(
        events=events,
        duration_us=int(round(sim_cfg["duration_s"] * 1e6)),
        gt=apparent_gt(spec, resolution, intrinsics) if intrinsics else lambda t: sample(spec, t),
        deviation_times=[d.at_t for d in spec.deviations],
        source="sim",
        meta={"spec": spec, "seed": seed, "resolution": resolution,
              "distorted": intrinsics is not None,
              "calibration": (camera_cfg or {}).get("calibration")},
    )
