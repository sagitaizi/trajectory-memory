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


def _blob_axes(frame, bg):
    """Semi-axis lengths of the lit region, from its second moments."""
    ys, xs = np.where(frame > bg)
    cov = np.cov(np.stack([xs, ys]))
    return 2 * np.sqrt(np.sort(np.linalg.eigvalsh(cov))[::-1])   # uniform ellipse: a = 2 sigma


def test_render_frames_draws_an_elongated_target_with_the_given_aspect():
    cfg = {**SIM_CFG, "resolution": [160, 120],
           "blob": {**SIM_CFG["blob"], "radius_px": 6, "aspect": 3.0, "angle": 0.0}}
    spec = TrajectorySpec(shape="circle", size=(0.0, 0.0), period_s=1.0)   # sits still
    frame = render_frames(spec, cfg)[0]
    long, short = _blob_axes(frame, cfg["blob"]["bg_intensity"])
    assert 2.4 < long / short < 3.6
    ys, xs = np.where(frame > cfg["blob"]["bg_intensity"])
    assert np.ptp(xs) > np.ptp(ys)                                # angle 0: long axis along x


def test_render_frames_draws_a_string_from_the_pivot_to_the_target():
    cfg = {**SIM_CFG, "resolution": [160, 120],
           "blob": {**SIM_CFG["blob"], "radius_px": 4,
                    "string": {"pivot": [0.5, -0.5], "thickness_px": 1, "intensity": 100}}}
    spec = TrajectorySpec(shape="circle", size=(0.0, 0.0), period_s=1.0)
    frame = render_frames(spec, cfg)[0]
    column = frame[:, 80]
    assert (column[:50] == 100).all()                             # string above the blob
    assert column[60] == cfg["blob"]["fg_intensity"]              # blob drawn on top
    assert (frame[:50, 70] == cfg["blob"]["bg_intensity"]).all()   # nothing beside it

    plain = render_frames(spec, {**cfg, "blob": {**cfg["blob"], "string": None}})[0]
    assert (plain[:50, 80] == cfg["blob"]["bg_intensity"]).all()


def test_an_elongated_target_on_a_string_hangs_along_it():
    cfg = {**SIM_CFG, "resolution": [160, 120],
           "blob": {**SIM_CFG["blob"], "radius_px": 4, "aspect": 3.0, "angle": 0.0,
                    "string": {"pivot": [0.5, -0.5], "thickness_px": 1, "intensity": 100}}}
    spec = TrajectorySpec(shape="circle", size=(0.0, 0.0), period_s=1.0)
    frame = render_frames(spec, cfg)[0]
    ys, xs = np.where(frame == cfg["blob"]["fg_intensity"])
    assert np.ptp(ys) > np.ptp(xs)                                # long axis along the string


def test_texture_puts_structure_inside_the_target_and_none_outside():
    base = {**SIM_CFG, "resolution": [160, 120],
            "blob": {**SIM_CFG["blob"], "radius_px": 10, "aspect": 2.0, "angle": 0.0}}
    spec = TrajectorySpec(shape="circle", size=(0.0, 0.0), period_s=1.0)
    plain = render_frames(spec, base)[0]
    cfg = {**base, "blob": {**base["blob"], "texture": {"depth": 0.5, "scale_px": 2, "seed": 3}}}
    textured = render_frames(spec, cfg)[0]
    bg, fg = base["blob"]["bg_intensity"], base["blob"]["fg_intensity"]
    inside = plain == fg
    assert (textured[~inside] == plain[~inside]).all()           # footprint unchanged
    assert textured[inside].std() > 0.15 * fg                    # varied inside
    assert abs(textured[inside].mean() - fg) < 0.15 * fg         # around the same brightness
    assert textured[inside].min() > bg                           # still brighter than the background


def test_texture_turns_rigidly_with_the_target():
    cfg = {**SIM_CFG, "resolution": [160, 120],
           "blob": {**SIM_CFG["blob"], "radius_px": 10, "aspect": 2.0, "angle": 0.0,
                    "texture": {"depth": 0.5, "scale_px": 2, "seed": 3}}}
    spec = TrajectorySpec(shape="circle", size=(0.0, 0.0), period_s=1.0)
    at0 = render_frames(spec, cfg)[0]
    at90 = render_frames(spec, {**cfg, "blob": {**cfg["blob"], "angle": np.pi / 2}})[0]
    r = 24
    crop0, crop90 = at0[60 - r:61 + r, 80 - r:81 + r], at90[60 - r:61 + r, 80 - r:81 + r]
    turned = np.rot90(crop0, k=-1)                               # +90 deg in image coords
    inside = (turned > cfg["blob"]["bg_intensity"]) & (crop90 > cfg["blob"]["bg_intensity"])
    assert inside.sum() > 400
    assert np.abs(turned[inside] - crop90[inside]).mean() < 0.05 * cfg["blob"]["fg_intensity"]


def test_iter_frames_yields_the_same_frames_one_at_a_time():
    """simulate() streams frames into v2e: a 15 s clip at 650 fps would be ~12 GB stacked."""
    from trajmem.simulate import iter_frames

    stacked = render_frames(a_circle(), SIM_CFG)
    streamed = list(iter_frames(a_circle(), SIM_CFG))
    assert len(streamed) == len(stacked)
    assert all(np.array_equal(a, b) for a, b in zip(streamed, stacked))
    assert all(f.shape == stacked.shape[1:] for f in streamed)


def test_seed_zero_is_reproducible_too():
    """v2e treats seed 0 as 'unseeded'; the wrapper must not pass it through."""
    cfg = {**SIM_CFG, "v2e": {**SIM_CFG["v2e"], "shot_noise_rate_hz": 5.0, "sigma_thres": 0.03}}
    a = simulate(a_circle(), camera_cfg=None, sim_cfg=cfg, seed=0).events
    b = simulate(a_circle(), camera_cfg=None, sim_cfg=cfg, seed=0).events
    assert len(a) > 0 and np.array_equal(a, b)


def test_a_reloaded_sim_clip_rebuilds_its_truth_with_the_calibration_it_was_rendered_with(tmp_path, monkeypatch):
    import trajmem.data as data
    from trajmem.data import load_clip, save_clip

    cfg = {**SIM_CFG, "resolution": [640, 480], "duration_s": 0.02}
    clip = simulate(a_circle(), camera_cfg={"calibration": None}, sim_cfg=cfg)
    assert clip.meta["distorted"] and "calibration" in clip.meta
    seen = []
    real = data.load_intrinsics if hasattr(data, "load_intrinsics") else None
    from trajmem import simulate as sim_module
    monkeypatch.setattr(sim_module, "load_intrinsics",
                        lambda camera_cfg=None: seen.append(camera_cfg) or load_intrinsics())
    load_clip(save_clip(clip, tmp_path / "c")).gt(0.0)
    assert seen == [{"calibration": None}]
