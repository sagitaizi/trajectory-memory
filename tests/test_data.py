from pathlib import Path

import numpy as np
import pytest

from trajmem.data import (
    MOTOR_TRIM_S,
    Clip,
    attach_labels,
    describe_clip,
    ego_gt,
    load_clip,
    load_recording,
    load_sim,
    read_anchor,
    save_clip,
    sim_params,
    trim_and_rebase,
)
from trajmem.simulate import EVENT_DTYPE
from trajmem.trajectories import TrajectorySpec, sample


def some_events(times_us):
    ev = np.zeros(len(times_us), dtype=EVENT_DTYPE)
    ev["timestamp"] = times_us
    ev["x"] = np.arange(len(times_us))
    return ev


def an_unlabelled_clip():
    return Clip(events=some_events([0, 1000]), duration_us=1_000_000, gt=None,
                source="corpus/real/fan/fan_brush_slow_01", meta={"group": "fan"})


def test_describe_clip_reads_group_and_slug_from_the_path():
    d = describe_clip("corpus/real/wall/scan_pan_slow_01/scan_pan_slow_01.aedat4")
    assert d["group"] == "wall"
    assert d["slug"] == "scan_pan_slow_01"
    assert d["is_break"] is False


def test_describe_clip_accepts_the_clip_directory():
    d = describe_clip("corpus/real/pendulum/wide_01")
    assert d["group"] == "pendulum"
    assert d["slug"] == "wide_01"


def test_describe_clip_flags_break_from_a_break_subdirectory():
    d = describe_clip("corpus/real/fan/break/fan_brush_break_01/fan_brush_break_01.aedat4")
    assert d["group"] == "fan"           # not "break"
    assert d["is_break"] is True


def test_describe_clip_flags_break_from_the_slug():
    assert describe_clip("corpus/real/wall/scan_diag_break_01")["is_break"] is True
    assert describe_clip("corpus/real/pendulum/wide_break")["is_break"] is True
    assert describe_clip("corpus/real/wall_target/loop_break_02")["is_break"] is True


def test_describe_clip_ignores_a_break_directory_above_the_corpus():
    d = describe_clip("D:/break/trajectory-memory/corpus/real/fan/fan_brush_slow_01")
    assert d["group"] == "fan"
    assert d["is_break"] is False


def test_describe_clip_uses_the_last_real_directory_in_the_path():
    d = describe_clip("D:/real/work/corpus/real/wall/scan_pan_slow_01")
    assert d["group"] == "wall"
    assert d["slug"] == "scan_pan_slow_01"


def test_trim_and_rebase_drops_the_head_and_zeroes_the_clock():
    ev = some_events([1000, 1500, 2000, 2500])
    out, t0 = trim_and_rebase(ev, trim_us=1000)
    assert t0 == 2000                       # device time that became t=0
    assert list(out["timestamp"]) == [0, 500]
    assert list(out["x"]) == [2, 3]         # the surviving events, not just their times


def test_trim_and_rebase_with_no_trim_still_zeroes_the_clock():
    ev = some_events([1789022718953626, 1789022718963626])
    out, t0 = trim_and_rebase(ev, trim_us=0)
    assert t0 == 1789022718953626
    assert list(out["timestamp"]) == [0, 10000]


def test_trim_and_rebase_rejects_a_trim_that_empties_the_clip():
    with pytest.raises(ValueError):
        trim_and_rebase(some_events([0, 100, 200]), trim_us=5000)


def test_motor_trim_is_half_a_second():
    assert MOTOR_TRIM_S == 0.5


def test_attach_labels_interpolates_between_marks():
    clip = attach_labels(an_unlabelled_clip(), [(0.0, 0.1, 0.2), (1.0, 0.5, 0.6)])
    np.testing.assert_allclose(clip.gt(0.5), [0.3, 0.4])


def test_attach_labels_holds_the_end_marks_just_outside_the_labelled_span():
    """A clip's first mark sits a few ms in; before it the target is where it was."""
    clip = attach_labels(an_unlabelled_clip(), [(1.0, 0.1, 0.2), (2.0, 0.5, 0.6)])
    np.testing.assert_allclose(clip.gt(0.0), [0.1, 0.2])
    np.testing.assert_allclose(clip.gt(4.0), [0.5, 0.6])


def test_attach_labels_is_unknown_far_beyond_the_last_mark():
    """A target that left the frame at 2 s did not sit still for the next seven."""
    clip = attach_labels(an_unlabelled_clip(), [(1.0, 0.1, 0.2), (2.0, 0.5, 0.6)])
    assert np.isnan(clip.gt(9.0)).all()


def test_attach_labels_is_unknown_inside_a_gap_where_the_target_was_not_seen():
    """Skipped instants in the marking tool must not become a straight line."""
    marks = [(t, 0.1 * t, 0.5) for t in (0.0, 0.1, 0.2, 0.3)] + \
            [(t, 0.1 * t, 0.5) for t in (5.0, 5.1, 5.2)]
    clip = attach_labels(an_unlabelled_clip(), marks)
    np.testing.assert_allclose(clip.gt(0.15), [0.015, 0.5])
    assert np.isnan(clip.gt(2.5)).all()
    np.testing.assert_allclose(clip.gt(5.05), [0.505, 0.5])


def test_attach_labels_knows_the_marks_either_side_of_a_gap():
    marks = [(0.0, 0.0, 0.5), (0.1, 0.01, 0.5), (5.0, 0.5, 0.5), (5.1, 0.51, 0.5)]
    clip = attach_labels(an_unlabelled_clip(), marks)
    np.testing.assert_allclose(clip.gt(0.1), [0.01, 0.5])
    np.testing.assert_allclose(clip.gt(5.0), [0.5, 0.5])


def test_attach_labels_gap_tolerance_can_be_set_explicitly():
    marks = [(0.0, 0.0, 0.5), (1.0, 0.1, 0.5), (3.0, 0.3, 0.5)]
    wide = attach_labels(an_unlabelled_clip(), marks)             # default: 3 x 1 s
    np.testing.assert_allclose(wide.gt(2.0), [0.2, 0.5])
    from trajmem.data import label_gt
    strict = label_gt(marks, max_gap_s=1.5)
    assert np.isnan(strict(2.0)).all()


def test_attach_labels_vectorised_query_marks_only_the_unknown_instants():
    marks = [(0.0, 0.0, 0.5), (0.1, 0.01, 0.5), (5.0, 0.5, 0.5), (5.1, 0.51, 0.5)]
    clip = attach_labels(an_unlabelled_clip(), marks)
    out = clip.gt(np.array([0.05, 2.5, 5.05]))
    assert out.shape == (3, 2)
    assert not np.isnan(out[0]).any()
    assert np.isnan(out[1]).all()
    assert not np.isnan(out[2]).any()


def test_attach_labels_gt_is_vectorised_like_the_sim_ground_truth():
    clip = attach_labels(an_unlabelled_clip(), [(0.0, 0.0, 0.0), (2.0, 1.0, 1.0)])
    out = clip.gt(np.array([0.0, 1.0, 2.0]))
    assert out.shape == (3, 2)
    np.testing.assert_allclose(out[:, 0], [0.0, 0.5, 1.0])


def test_attach_labels_sorts_marks_given_out_of_order():
    clip = attach_labels(an_unlabelled_clip(), [(1.0, 0.5, 0.6), (0.0, 0.1, 0.2)])
    np.testing.assert_allclose(clip.gt(0.5), [0.3, 0.4])


def test_attach_labels_leaves_the_original_clip_untouched():
    original = an_unlabelled_clip()
    labelled = attach_labels(original, [(0.0, 0.1, 0.2), (1.0, 0.5, 0.6)])
    assert original.gt is None
    assert labelled is not original
    assert labelled.source == original.source


def test_attach_labels_keeps_the_marks_so_the_clip_can_be_saved():
    clip = attach_labels(an_unlabelled_clip(), [(0.0, 0.1, 0.2), (1.0, 0.5, 0.6)])
    np.testing.assert_allclose(clip.meta["labels"], [(0.0, 0.1, 0.2), (1.0, 0.5, 0.6)])
    assert clip.meta["group"] == "fan"          # existing meta survives


def test_attach_labels_rejects_an_empty_mark_list():
    with pytest.raises(ValueError):
        attach_labels(an_unlabelled_clip(), [])


def a_sim_clip():
    spec = TrajectorySpec(shape="circle", size=(0.2, 0.2), period_s=1.0)
    return Clip(events=some_events([0, 500, 1000]), duration_us=2_000_000,
                gt=lambda t: sample(spec, t), deviation_times=[1.25],
                source="sim", meta={"spec": spec, "seed": 7})


def test_save_and_load_round_trips_the_events_and_scalars(tmp_path):
    clip = a_sim_clip()
    save_clip(clip, tmp_path / "c.npz")
    back = load_clip(tmp_path / "c.npz")

    np.testing.assert_array_equal(back.events, clip.events)
    assert back.events.dtype == EVENT_DTYPE
    assert back.duration_us == 2_000_000
    assert back.deviation_times == [1.25]
    assert back.source == "sim"
    assert back.meta["seed"] == 7


def test_load_rebuilds_sim_ground_truth_from_the_saved_spec(tmp_path):
    clip = a_sim_clip()
    save_clip(clip, tmp_path / "c.npz")
    back = load_clip(tmp_path / "c.npz")
    np.testing.assert_allclose(back.gt(0.37), clip.gt(0.37))


def test_load_rebuilds_hand_label_ground_truth(tmp_path):
    clip = attach_labels(an_unlabelled_clip(), [(0.0, 0.1, 0.2), (1.0, 0.5, 0.6)])
    save_clip(clip, tmp_path / "c.npz")
    back = load_clip(tmp_path / "c.npz")
    np.testing.assert_allclose(back.gt(0.5), [0.3, 0.4])


def test_load_keeps_ground_truth_absent_when_the_clip_had_none(tmp_path):
    save_clip(an_unlabelled_clip(), tmp_path / "c.npz")
    assert load_clip(tmp_path / "c.npz").gt is None


def test_save_clip_returns_the_path_it_actually_wrote(tmp_path):
    """np.savez appends .npz; the returned path has to be the file that exists."""
    written = save_clip(an_unlabelled_clip(), tmp_path / "c")
    assert written.exists()
    assert load_clip(written).duration_us == 1_000_000


# --- 4a: apparent motion of a static target under a known camera rotation ---

TPR = (740.0, 740.8)          # ticks per radian, pan/tilt (main repo params.yaml)
PX_PER_TICK = 0.663           # measured 2026-08-02, +-1.4%


def motors(pan_ticks, tilt_ticks, times_us=None):
    from recording.player import MotorTrack

    n = len(pan_ticks)
    times_us = list(range(0, n * 30_000, 30_000)) if times_us is None else times_us
    return MotorTrack(times_us, pan_ticks, tilt_ticks)


def intrinsics():
    from trajmem.simulate import load_intrinsics

    return load_intrinsics()


def test_ego_gt_with_a_still_camera_stays_on_the_anchor():
    intr = intrinsics()
    gt = ego_gt((320.0, 240.0), motors([2048] * 4, [1000] * 4), intr, t0_us=0,
                ticks_per_radian=TPR)
    np.testing.assert_allclose(gt(0.0), [320.0 / intr.width, 240.0 / intr.height], atol=1e-6)
    np.testing.assert_allclose(gt(0.09), gt(0.0), atol=1e-6)


def test_ego_gt_pans_the_target_horizontally():
    intr = intrinsics()
    gt = ego_gt((320.0, 240.0), motors([2048, 2148], [1000, 1000]), intr, t0_us=0,
                ticks_per_radian=TPR)
    start, end = gt(0.0), gt(0.03)
    assert abs(end[0] - start[0]) > 0.05                  # moved across the frame
    assert abs(end[1] - start[1]) < 0.01                  # but barely up or down


def test_ego_gt_tilts_the_target_vertically():
    intr = intrinsics()
    gt = ego_gt((320.0, 240.0), motors([2048, 2048], [1000, 1100]), intr, t0_us=0,
                ticks_per_radian=TPR)
    start, end = gt(0.0), gt(0.03)
    assert abs(end[1] - start[1]) > 0.05
    assert abs(end[0] - start[0]) < 0.01


def test_ego_gt_displacement_matches_the_measured_pixels_per_tick():
    intr = intrinsics()
    d_ticks = 200
    gt = ego_gt((320.0, 240.0), motors([2048, 2048 + d_ticks], [1000, 1000]), intr,
                t0_us=0, ticks_per_radian=TPR)
    moved_px = abs(gt(0.03)[0] - gt(0.0)[0]) * intr.width
    expected = d_ticks * PX_PER_TICK
    assert 0.85 * expected < moved_px < 1.15 * expected


def test_ego_gt_pan_direction_follows_the_sign_convention():
    """Measured on scan_pan_slow_01: the target tracks +pan ticks, r=+0.97 over two sweeps."""
    gt = ego_gt((320.0, 240.0), motors([2048, 2248], [1000, 1000]), intrinsics(),
                t0_us=0, ticks_per_radian=TPR)
    assert gt(0.03)[0] > gt(0.0)[0]                       # +pan ticks -> target moves right


def test_ego_gt_tilt_direction_follows_the_sign_convention():
    """Measured on four tilt clips and both diagonals: the target tracks +tilt ticks,
    r=+0.95..+1.00."""
    gt = ego_gt((320.0, 240.0), motors([2048, 2048], [1000, 1200]), intrinsics(),
                t0_us=0, ticks_per_radian=TPR)
    assert gt(0.03)[1] > gt(0.0)[1]                       # +tilt ticks -> target moves down


def test_ego_gt_is_vectorised_over_time():
    gt = ego_gt((320.0, 240.0), motors([2048, 2098, 2148], [1000, 1000, 1000]),
                intrinsics(), t0_us=0, ticks_per_radian=TPR)
    out = gt(np.array([0.0, 0.03, 0.06]))
    assert out.shape == (3, 2)
    assert out[0, 0] < out[1, 0] < out[2, 0] or out[0, 0] > out[1, 0] > out[2, 0]


def test_ego_gt_queries_the_side_car_in_device_time():
    """Encoder times are device timestamps; gt takes seconds since the trimmed start."""
    t0 = 1789022718953626
    track = motors([2048, 2148], [1000, 1000], times_us=[t0, t0 + 30_000])
    gt = ego_gt((320.0, 240.0), track, intrinsics(), t0_us=t0, ticks_per_radian=TPR)
    assert abs(gt(0.03)[0] - gt(0.0)[0]) > 0.05


def test_ego_gt_anchors_on_the_instant_the_pixel_was_marked():
    """The marker clicks the centre of a smeared window, not the clip's first instant."""
    intr = intrinsics()
    track = motors([2048, 2148, 2248], [1000, 1000, 1000])      # 0, 30ms, 60ms
    gt = ego_gt((320.0, 240.0), track, intr, t0_us=0, ticks_per_radian=TPR,
                anchor_t_s=0.03)
    np.testing.assert_allclose(gt(0.03), [320.0 / intr.width, 240.0 / intr.height],
                               atol=1e-6)
    assert not np.allclose(gt(0.0), gt(0.03))                   # t=0 is elsewhere


def test_read_anchor_returns_none_when_unmarked(tmp_path):
    assert read_anchor(tmp_path / "wide_01") is None


def test_read_anchor_reads_a_marked_pixel(tmp_path):
    clip_dir = tmp_path / "scan_pan_slow_01"
    clip_dir.mkdir()
    (clip_dir / "scan_pan_slow_01.anchor.json").write_text('{"x": 301.5, "y": 244.0}')
    assert read_anchor(clip_dir) == (301.5, 244.0, 0.0)


def test_read_anchor_keys_off_the_file_stem_not_the_folder(tmp_path):
    """A clip file need not share its folder's name; the stem is what names side-cars."""
    clip_dir = tmp_path / "renamed_folder"
    clip_dir.mkdir()
    (clip_dir / "scan_pan_slow_01.anchor.json").write_text('{"x": 12.0, "y": 34.0}')
    assert read_anchor(clip_dir / "scan_pan_slow_01.aedat4") == (12.0, 34.0, 0.0)


# --- integration against the real corpus (gitignored; skipped when absent) ---

CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "real"
A_MOTOR_CLIP = CORPUS / "wall" / "scan_pan_slow_01"
A_STATIC_CLIP = CORPUS / "pendulum" / "small_02"

needs_motor_clip = pytest.mark.skipif(not A_MOTOR_CLIP.is_dir(), reason="corpus absent")
needs_static_clip = pytest.mark.skipif(not A_STATIC_CLIP.is_dir(), reason="corpus absent")


@needs_motor_clip
def test_load_recording_trims_a_motor_clip_and_zeroes_its_clock():
    clip = load_recording(A_MOTOR_CLIP)
    assert clip.events["timestamp"][0] == 0
    assert np.all(np.diff(clip.events["timestamp"]) >= 0)
    assert 39.7 < clip.duration_us / 1e6 < 39.9      # 40.28 s recorded, 0.5 s trimmed
    assert clip.meta["group"] == "wall"
    assert clip.meta["is_break"] is False
    assert clip.meta["t0_device_us"] > 0


@needs_motor_clip
def test_load_recording_records_the_sensor_resolution():
    """The marking tool needs the sensor size; nothing else in the Clip carries it."""
    assert load_recording(A_MOTOR_CLIP).meta["resolution"] == (640, 480)


@needs_motor_clip
def test_load_recording_yields_the_same_dtype_as_a_simulated_clip():
    """The whole point of Clip: the frontend cannot tell real from simulated."""
    assert load_recording(A_MOTOR_CLIP).events.dtype == EVENT_DTYPE


AN_UNMARKED_CLIP = next((p for p in sorted(CORPUS.glob("wall/*/*.aedat4"))
                         if read_anchor(p) is None), None)


@pytest.mark.skipif(AN_UNMARKED_CLIP is None, reason="every wall clip is marked")
def test_load_recording_leaves_ground_truth_absent_until_the_anchor_is_marked():
    assert load_recording(AN_UNMARKED_CLIP).gt is None


@needs_motor_clip
def test_load_recording_builds_ground_truth_once_the_anchor_is_marked():
    """scan_pan_slow_01 was marked on 2026-09-10, so its gt comes from the encoders."""
    clip = load_recording(A_MOTOR_CLIP)
    assert clip.gt is not None
    assert clip.meta["anchor"] == (220.0, 40.0, 0.01)


@needs_motor_clip
def test_load_recording_builds_analytic_ground_truth_from_a_given_anchor():
    clip = load_recording(A_MOTOR_CLIP, anchor=(320.0, 240.0))
    assert clip.gt is not None
    track = clip.gt(np.linspace(0.0, 30.0, 300))
    assert track.shape == (300, 2)
    assert np.ptp(track[:, 0]) > 0.05                # the sweep really moves it
    assert np.ptp(track[:, 1]) < 0.02                # a pan clip: little vertical drift


@needs_motor_clip
def test_save_and_load_keeps_encoder_ground_truth_through_the_cache(tmp_path):
    """A 4a clip's gt lives in the encoders, so the cache has to rebuild it too."""
    clip = load_recording(A_MOTOR_CLIP, anchor=(320.0, 240.0))
    back = load_clip(save_clip(clip, tmp_path / "w.npz"))
    assert back.gt is not None
    track = back.gt(np.linspace(0.0, 30.0, 300))
    assert np.ptp(track[:, 0]) > 0.05                # a real sweep, not a constant
    np.testing.assert_allclose(track, clip.gt(np.linspace(0.0, 30.0, 300)))


@needs_motor_clip
def test_load_recording_honours_the_instant_an_anchor_was_marked_at():
    """An anchor carries the instant it was marked at, and gt returns it there."""
    clip = load_recording(A_MOTOR_CLIP, anchor=(320.0, 240.0, 1.5))
    np.testing.assert_allclose(clip.gt(1.5), [320.0 / 640, 240.0 / 480], atol=1e-6)
    assert not np.allclose(clip.gt(0.0), clip.gt(1.5))


@needs_motor_clip
def test_load_recording_still_accepts_an_anchor_without_an_instant():
    clip = load_recording(A_MOTOR_CLIP, anchor=(320.0, 240.0))
    np.testing.assert_allclose(clip.gt(0.0), [320.0 / 640, 240.0 / 480], atol=1e-6)


@needs_static_clip
def test_load_recording_does_not_trim_a_clip_recorded_without_motors():
    clip = load_recording(A_STATIC_CLIP)
    assert 24.5 < clip.duration_us / 1e6 < 25.5
    assert clip.meta["group"] == "pendulum"
    assert clip.gt is None


# --- simulated clips ---

TINY_SIM = {
    "resolution": [64, 48], "fps": 200, "duration_s": 0.15,
    "blob": {"radius_px": 4, "bg_intensity": 20, "fg_intensity": 180},
    "calibration": None,
    "v2e": {"pos_thres": 0.2, "neg_thres": 0.2, "sigma_thres": 0.0, "cutoff_hz": 0,
            "leak_rate_hz": 0.0, "shot_noise_rate_hz": 0.0, "refractory_period_s": 0.0},
}


def test_sim_params_reads_the_project_defaults():
    cfg = sim_params()
    assert cfg["resolution"] == [640, 480]
    assert "v2e" in cfg


A_TINY_CIRCLE = TrajectorySpec(shape="circle", size=(0.25, 0.25), period_s=0.1)


def test_load_sim_returns_a_clip_matching_a_recorded_one():
    clip = load_sim(A_TINY_CIRCLE, sim_cfg=TINY_SIM, seed=1, distort=False)
    assert clip.source == "sim"
    assert clip.events.dtype == EVENT_DTYPE
    assert len(clip.events) > 0
    np.testing.assert_allclose(clip.gt(0.037), sample(A_TINY_CIRCLE, 0.037))


def test_load_sim_puts_the_events_where_the_ground_truth_says():
    """The events and clip.gt must describe the same path, not just coexist."""
    clip = load_sim(A_TINY_CIRCLE, sim_cfg=TINY_SIM, seed=1, distort=False)
    w, h = TINY_SIM["resolution"]
    t = 0.075
    near = np.abs(clip.events["timestamp"] - t * 1e6) < 5000
    assert near.sum() > 20
    centroid = [clip.events["x"][near].mean() / w, clip.events["y"][near].mean() / h]
    np.testing.assert_allclose(centroid, sample(A_TINY_CIRCLE, t), atol=0.1)


def test_load_sim_rejects_a_resolution_the_calibration_does_not_cover():
    """Distorting into a 64x48 frame with 640x480 intrinsics silently moves the path."""
    with pytest.raises(ValueError, match="resolution"):
        load_sim(A_TINY_CIRCLE, sim_cfg=TINY_SIM, seed=1)


def test_load_sim_survives_a_save_and_load_round_trip(tmp_path):
    clip = load_sim(A_TINY_CIRCLE, sim_cfg=TINY_SIM, seed=1, distort=False)
    back = load_clip(save_clip(clip, tmp_path / "s.npz"))
    np.testing.assert_array_equal(back.events, clip.events)
    np.testing.assert_allclose(back.gt(0.037), clip.gt(0.037))


# --- hand-labels win over a side-car anchor ----------------------------------

@needs_motor_clip
def test_hand_labels_take_precedence_over_a_marked_anchor():
    """A 4a clip labelled by hand uses the labels: they are observation, not a model."""
    from trajmem.data import write_labels

    sidecar = A_MOTOR_CLIP / f"{A_MOTOR_CLIP.name}.labels.csv"
    if sidecar.exists():
        pytest.skip("clip already labelled; not clobbering real labels")
    try:
        write_labels(A_MOTOR_CLIP, [(0.0, 0.25, 0.5), (1.0, 0.75, 0.5)])
        clip = load_recording(A_MOTOR_CLIP)
        assert "labels" in clip.meta
        assert "anchor" not in clip.meta
        np.testing.assert_allclose(clip.gt(0.5), (0.5, 0.5))
    finally:
        sidecar.unlink(missing_ok=True)


@needs_motor_clip
def test_an_explicitly_given_anchor_still_forces_the_encoder_ground_truth():
    """Passing anchor= asks for the rig model by name, so labels must not override it."""
    from trajmem.data import write_labels

    sidecar = A_MOTOR_CLIP / f"{A_MOTOR_CLIP.name}.labels.csv"
    if sidecar.exists():
        pytest.skip("clip already labelled; not clobbering real labels")
    try:
        write_labels(A_MOTOR_CLIP, [(0.0, 0.25, 0.5), (1.0, 0.75, 0.5)])
        clip = load_recording(A_MOTOR_CLIP, anchor=(320.0, 240.0))
        assert clip.meta["anchor"] == (320.0, 240.0, 0.0)
        assert "labels" not in clip.meta
    finally:
        sidecar.unlink(missing_ok=True)


# --- slicing ------------------------------------------------------------------

def a_labelled_clip():
    ev = some_events(np.arange(0, 10_000_000, 250_000))          # one event every 0.25 s
    marks = [(t, 0.1 * t, 0.5) for t in range(0, 11)]             # x runs with time
    clip = Clip(events=ev, duration_us=10_000_000, gt=None, deviation_times=[2.0, 7.5],
                source="sim", meta={"resolution": (640, 480)})
    return attach_labels(clip, marks)


def test_slice_clip_keeps_the_window_and_rebases_the_clock():
    from trajmem.data import slice_clip

    part = slice_clip(a_labelled_clip(), 3.0, 6.0)
    assert part.duration_us == 3_000_000
    assert part.events["timestamp"][0] == 0
    assert part.events["timestamp"][-1] < 3_000_000
    assert len(part.events) == 12                                 # 3 s at 4 per second


def test_slice_clip_shifts_ground_truth_and_break_times():
    from trajmem.data import slice_clip

    whole = a_labelled_clip()
    part = slice_clip(whole, 3.0, 6.0)
    assert np.allclose(part.gt(0.0), whole.gt(3.0))
    assert np.allclose(part.gt([1.0, 2.5]), whole.gt([4.0, 5.5]))
    assert part.deviation_times == []                             # 2.0 and 7.5 fall outside
    assert slice_clip(whole, 1.0, 8.0).deviation_times == [1.0, 6.5]


def test_slice_clip_leaves_the_original_and_records_the_window():
    from trajmem.data import slice_clip

    whole = a_labelled_clip()
    part = slice_clip(whole, 3.0, 6.0)
    assert whole.duration_us == 10_000_000 and whole.events["timestamp"][0] == 0
    assert part.meta["window_s"] == (3.0, 6.0)
    assert part.meta["resolution"] == (640, 480)


def test_slice_clip_rejects_an_empty_or_backwards_window():
    from trajmem.data import slice_clip

    with pytest.raises(ValueError):
        slice_clip(a_labelled_clip(), 6.0, 3.0)
    with pytest.raises(ValueError):
        slice_clip(a_labelled_clip(), 20.0, 25.0)
