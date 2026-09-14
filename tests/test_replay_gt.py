import numpy as np
import pytest

from scripts.replay_gt import (frame_times, scaled_ticks_per_radian, to_pixels,
                               trail_points)


def a_gt(fn):
    """Wrap a plain function into the vectorised shape Clip.gt promises."""
    def gt(t):
        t = np.asarray(t, dtype=float)
        out = np.stack(fn(np.atleast_1d(t)), axis=-1)
        return out[0] if t.ndim == 0 else out
    return gt


# --- frame timing ------------------------------------------------------------

def test_frame_times_starts_at_zero_and_stays_inside_the_clip():
    times = frame_times(2.0, fps=10)
    assert times[0] == 0.0
    assert times[-1] < 2.0


def test_frame_times_spaces_frames_by_one_over_fps():
    assert frame_times(1.0, fps=4) == pytest.approx([0.0, 0.25, 0.5, 0.75])


def test_frame_times_rejects_a_non_positive_fps():
    with pytest.raises(ValueError):
        frame_times(1.0, fps=0)


# --- the trail ---------------------------------------------------------------

def test_trail_covers_the_requested_seconds_before_now():
    gt = a_gt(lambda t: (t, np.zeros_like(t)))
    trail = trail_points(gt, t=5.0, trail_s=2.0)
    assert trail[0][0] == pytest.approx(3.0)
    assert trail[-1][0] == pytest.approx(5.0)


def test_trail_is_clipped_at_the_start_of_the_clip():
    gt = a_gt(lambda t: (t, np.zeros_like(t)))
    trail = trail_points(gt, t=0.5, trail_s=2.0)
    assert trail[0][0] == pytest.approx(0.0)


def test_trail_at_the_first_frame_is_a_single_point():
    gt = a_gt(lambda t: (t, np.zeros_like(t)))
    assert trail_points(gt, t=0.0, trail_s=2.0).shape == (1, 2)


def test_turning_the_trail_off_leaves_only_the_current_point():
    gt = a_gt(lambda t: (t, np.zeros_like(t)))
    trail = trail_points(gt, t=5.0, trail_s=0.0)
    assert trail.shape == (1, 2)
    assert trail[0][0] == pytest.approx(5.0)


# --- normalised -> pixels ----------------------------------------------------

def test_to_pixels_scales_by_the_sensor_resolution():
    assert to_pixels((0.5, 0.25), (640, 480))[0] == pytest.approx([320.0, 120.0])


def test_to_pixels_accepts_a_whole_track():
    px = to_pixels([(0.0, 0.0), (1.0, 1.0)], (640, 480))
    assert px.shape == (2, 2)
    assert px[1] == pytest.approx([640.0, 480.0])


# --- the trial encoder scale -------------------------------------------------

def test_a_scale_above_one_lowers_ticks_per_radian_on_both_axes():
    assert scaled_ticks_per_radian((740.0, 370.0), 1.307) == pytest.approx(
        (740.0 / 1.307, 370.0 / 1.307))


def test_a_scale_of_one_leaves_the_calibration_alone():
    assert scaled_ticks_per_radian((740.0, 370.0), 1.0) == pytest.approx((740.0, 370.0))


def test_a_non_positive_scale_is_rejected():
    with pytest.raises(ValueError):
        scaled_ticks_per_radian((740.0, 370.0), 0.0)


# --- opening clips -------------------------------------------------------------

def test_open_clip_reads_a_saved_simulated_clip(tmp_path):
    from scripts.replay_gt import open_clip, render
    from trajmem.data import save_clip
    from trajmem.simulate import simulate
    from trajmem.trajectories import TrajectorySpec

    cfg = {"resolution": [64, 48], "fps": 200, "duration_s": 0.1,
           "blob": {"radius_px": 4, "bg_intensity": 20, "fg_intensity": 180},
           "v2e": {"pos_thres": 0.2, "neg_thres": 0.2, "sigma_thres": 0.0, "cutoff_hz": 0,
                   "leak_rate_hz": 0.0, "shot_noise_rate_hz": 0.0, "refractory_period_s": 0.0}}
    spec = TrajectorySpec(shape="circle", size=(0.25, 0.25), period_s=0.1)
    path = save_clip(simulate(spec, camera_cfg=None, sim_cfg=cfg), tmp_path / "sim_01")

    clip, label = open_clip(path)
    assert label == "sim_01"
    assert clip.gt is not None
    frame = render(clip, 0.02, 0.01, 40, 0.05, 1.0, label)   # a sim clip has no motor HUD
    assert frame.shape[0] > 0
