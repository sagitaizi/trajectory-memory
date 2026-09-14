import numpy as np
import pytest

from trajmem.simulate import (
    EVENT_DTYPE,
    _path_pixels,
    _run_v2e,
    load_intrinsics,
    render_frames,
    simulate,
)
from trajmem.trajectories import Deviation, TrajectorySpec, sample

SIM_CFG = {
    "resolution": [64, 48],
    "fps": 200,
    "duration_s": 0.15,
    "blob": {"radius_px": 4, "bg_intensity": 20, "fg_intensity": 180},
    "v2e": {
        "pos_thres": 0.2, "neg_thres": 0.2, "sigma_thres": 0.0, "cutoff_hz": 0,
        "leak_rate_hz": 0.0, "shot_noise_rate_hz": 0.0, "refractory_period_s": 0.0,
    },
}


def a_circle():
    return TrajectorySpec(shape="circle", size=(0.25, 0.25), period_s=0.1, center=(0.5, 0.5))


def test_render_frames_shape_and_dtype():
    frames = render_frames(a_circle(), SIM_CFG)
    assert frames.shape == (30, 48, 64)          # T = round(0.15 * 200), (H, W)
    assert frames.dtype == np.float32


def test_render_frames_blob_centroid_follows_the_path():
    from trajmem.trajectories import sample

    spec = a_circle()
    frames = render_frames(spec, SIM_CFG)
    w, h = SIM_CFG["resolution"]
    fps = SIM_CFG["fps"]
    bg = SIM_CFG["blob"]["bg_intensity"]
    errs = []
    for i, frame in enumerate(frames):
        ys, xs = np.where(frame > bg)
        cx, cy = xs.mean(), ys.mean()
        want = sample(spec, i / fps) * np.array([w, h])
        errs.append(np.hypot(cx - want[0], cy - want[1]))
    assert max(errs) < 1.5
    # the blob actually travels round the circle, not stuck at the centre
    assert np.std([sample(spec, i / fps)[0] for i in range(len(frames))]) > 0.05


def test_render_frames_uses_bg_and_fg_intensities():
    frames = render_frames(a_circle(), SIM_CFG)
    assert np.isclose(frames.min(), SIM_CFG["blob"]["bg_intensity"])
    assert np.isclose(frames.max(), SIM_CFG["blob"]["fg_intensity"])


def test_load_intrinsics_is_the_dvxplorer_geometry():
    intr = load_intrinsics()
    assert (intr.width, intr.height) == (640, 480)
    assert 400 < intr.fx < 700          # DVXplorer refit ~518
    assert 250 < intr.cx < 400


def test_distortion_pulls_off_axis_points_inward():
    intr = load_intrinsics()
    spec = TrajectorySpec(shape="circle", size=(0.45, 0.45), period_s=1.0, center=(0.5, 0.5))
    t = np.linspace(0, 1, 60, endpoint=False)
    pinhole = _path_pixels(spec, t, (640, 480), None)
    distorted = _path_pixels(spec, t, (640, 480), intr)
    c = np.array([intr.cx, intr.cy])
    r_pinhole = np.linalg.norm(pinhole - c, axis=1).mean()
    r_distorted = np.linalg.norm(distorted - c, axis=1).mean()
    assert r_distorted < r_pinhole                       # barrel distortion, k1 < 0


def test_run_v2e_emits_events_within_frame_bounds():
    frames = render_frames(a_circle(), SIM_CFG)
    times = np.arange(len(frames)) / SIM_CFG["fps"]
    events = _run_v2e(frames, times, SIM_CFG["v2e"])
    assert events.dtype == EVENT_DTYPE
    assert len(events) > 0
    assert events["x"].min() >= 0 and events["x"].max() < SIM_CFG["resolution"][0]
    assert events["y"].min() >= 0 and events["y"].max() < SIM_CFG["resolution"][1]
    assert np.all(np.diff(events["timestamp"]) >= 0)
    assert set(np.unique(events["polarity"])).issubset({0, 1})


def test_run_v2e_event_count_grows_with_blob_speed():
    fast = TrajectorySpec(shape="circle", size=(0.25, 0.25), period_s=0.05, center=(0.5, 0.5))
    slow = TrajectorySpec(shape="circle", size=(0.25, 0.25), period_s=0.30, center=(0.5, 0.5))
    times = np.arange(30) / SIM_CFG["fps"]
    n_fast = len(_run_v2e(render_frames(fast, SIM_CFG), times, SIM_CFG["v2e"]))
    n_slow = len(_run_v2e(render_frames(slow, SIM_CFG), times, SIM_CFG["v2e"]))
    assert n_fast > n_slow


def test_simulate_returns_clip_with_exact_ground_truth():
    spec = TrajectorySpec(
        shape="circle", size=(0.25, 0.25), period_s=0.1, center=(0.5, 0.5),
        deviations=[Deviation(at_t=0.08, kind="shrink", params={"factor": 0.5})],
    )
    clip = simulate(spec, camera_cfg=None, sim_cfg=SIM_CFG, seed=1)
    assert clip.source == "sim"
    assert clip.events.dtype == EVENT_DTYPE
    assert len(clip.events) > 0
    assert clip.duration_us == 150_000
    assert clip.deviation_times == [0.08]
    np.testing.assert_allclose(clip.gt(0.037), sample(spec, 0.037))


def test_simulate_with_camera_cfg_runs_the_distortion_branch():
    """Distortion is only meaningful at the resolution the calibration was fitted at."""
    cfg = {**SIM_CFG, "resolution": [640, 480], "duration_s": 0.05}
    clip = simulate(a_circle(), camera_cfg={}, sim_cfg=cfg, seed=1)
    assert clip.source == "sim"
    assert len(clip.events) > 0
    assert clip.events["x"].max() < 640 and clip.events["y"].max() < 480


def test_render_frames_rejects_a_resolution_the_calibration_does_not_cover():
    with pytest.raises(ValueError, match="resolution"):
        render_frames(a_circle(), SIM_CFG, intrinsics=load_intrinsics())


def test_simulate_records_the_resolution_like_a_recording_does():
    clip = simulate(a_circle(), camera_cfg=None, sim_cfg=SIM_CFG)
    assert clip.meta["resolution"] == (64, 48)


def test_simulated_ground_truth_is_the_apparent_position_under_the_lens():
    """Hand-labels on real clips are where the target shows up on the sensor, so a
    simulated clip's gt must be the lens-distorted pixel, not the ideal path."""
    cfg = {**SIM_CFG, "resolution": [640, 480], "duration_s": 0.05}
    spec = TrajectorySpec(shape="circle", size=(0.4, 0.4), period_s=1.0, center=(0.5, 0.5))
    clip = simulate(spec, camera_cfg={}, sim_cfg=cfg)
    intr = load_intrinsics()
    t = np.array([0.0, 0.02])
    want = _path_pixels(spec, t, (640, 480), intr) / np.array([640, 480])
    assert np.allclose(clip.gt(t), want)
    assert not np.allclose(clip.gt(t), sample(spec, t))
    assert np.allclose(clip.gt(0.02), want[1])                  # scalar t too


def test_saved_simulated_clip_keeps_the_apparent_ground_truth(tmp_path):
    from trajmem.data import load_clip, save_clip

    cfg = {**SIM_CFG, "resolution": [640, 480], "duration_s": 0.05}
    spec = TrajectorySpec(shape="circle", size=(0.4, 0.4), period_s=1.0, center=(0.5, 0.5))
    clip = simulate(spec, camera_cfg={}, sim_cfg=cfg)
    back = load_clip(save_clip(clip, tmp_path / "c"))
    t = np.linspace(0, 0.05, 5)
    assert np.allclose(back.gt(t), clip.gt(t))
