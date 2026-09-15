import _thesis_path  # noqa: F401  (side effect: adds the thesis repo to sys.path)
import numpy as np
import pytest

from trajmem.baseline import HarmonicFit
from trajmem.data import Clip
from trajmem.experiment import Trace, evaluate_clip, make_memory, score_trace
from trajmem.simulate import EVENT_DTYPE
from trajmem.trajectories import Deviation, TrajectorySpec, sample

RES = (640, 480)


def a_clip_of_events(spec, duration_s=8.0, per_window=30, window_us=5000, seed=0):
    """Events clustered on the analytic path, one clump per window, no simulator."""
    rng = np.random.default_rng(seed)
    n_win = int(duration_s * 1e6 / window_us)
    t_win = (np.arange(n_win) + 0.5) * window_us / 1e6
    centres = sample(spec, t_win) * RES
    xs = np.repeat(centres[:, 0], per_window) + rng.normal(0, 2, n_win * per_window)
    ys = np.repeat(centres[:, 1], per_window) + rng.normal(0, 2, n_win * per_window)
    ts = np.repeat(np.arange(n_win) * window_us, per_window) + rng.integers(0, window_us, n_win * per_window)
    ev = np.zeros(len(xs), dtype=EVENT_DTYPE)
    ev["x"], ev["y"], ev["timestamp"], ev["polarity"] = np.clip(xs, 0, 639), np.clip(ys, 0, 479), ts, 1
    ev = ev[np.argsort(ev["timestamp"], kind="stable")]
    return Clip(events=ev, duration_us=int(duration_s * 1e6), gt=lambda t: sample(spec, t),
                deviation_times=[d.at_t for d in spec.deviations], meta={"resolution": RES})


def a_spec(deviations=None):
    return TrajectorySpec(shape="circle", size=(0.2, 0.2), period_s=1.0, deviations=deviations or [])


def test_evaluate_clip_walks_the_clip_and_lines_up_truth_at_the_horizon():
    clip = a_clip_of_events(a_spec())
    trace = evaluate_clip(HarmonicFit(dt_s=0.005), clip, window_us=5000, horizon_s=0.1)
    assert isinstance(trace, Trace)
    assert len(trace.t) == 1600 and trace.pred.shape == (1600, 2) == trace.gt_ahead.shape
    assert trace.t[0] == pytest.approx(0.0025) and trace.horizon_s == 0.1
    assert np.allclose(trace.gt_ahead[:10], sample(a_spec(), trace.t[:10] + 0.1))
    assert np.isnan(trace.gt_ahead[-1]).all()          # t + h runs past the clip's end
    late = trace.t > 5.0
    err = np.hypot(*(trace.pred[late] - trace.gt_ahead[late]).T)
    assert np.nanmedian(err) < 0.01


def test_score_trace_reports_the_three_metrics_in_pixels():
    spec = a_spec(deviations=[Deviation(at_t=6.0, kind="shrink", params={"factor": 0.5})])
    clip = a_clip_of_events(spec)
    trace = evaluate_clip(HarmonicFit(dt_s=0.005), clip, window_us=5000, horizon_s=0.1)
    r = score_trace(trace, clip, tol_px=10.0, settle_s=4.0)
    assert r["error_px"]["median"] < 6.0                 # pre-break steps only
    assert 0 < r["lock_on_s"] < 4.0
    assert r["deviation"]["auc"] > 0.95 and r["deviation"]["latency_s"] < 0.5


def test_score_trace_without_a_break_has_no_detection_numbers_but_a_false_alarm_rate():
    clip = a_clip_of_events(a_spec())
    trace = evaluate_clip(HarmonicFit(dt_s=0.005), clip, window_us=5000, horizon_s=0.1)
    r = score_trace(trace, clip, tol_px=10.0, settle_s=4.0, threshold=1e9)
    assert np.isnan(r["deviation"]["auc"]) and r["deviation"]["fp_per_min"] == 0.0


def test_make_memory_builds_the_named_baseline():
    assert isinstance(make_memory("harmonic", dt_s=0.005), HarmonicFit)
    with pytest.raises(ValueError):
        make_memory("nonsense", dt_s=0.005)
