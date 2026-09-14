import _thesis_path  # noqa: F401  (side effect: adds the thesis repo to sys.path)
import numpy as np

from scripts.match_sim_real import event_stats
from trajmem.data import Clip
from trajmem.simulate import EVENT_DTYPE


def a_clip(xs, ys, ts_us, pols, gt, duration_us=1_000_000):
    ev = np.empty(len(xs), dtype=EVENT_DTYPE)
    ev["x"], ev["y"], ev["timestamp"], ev["polarity"] = xs, ys, ts_us, pols
    return Clip(events=ev, duration_us=duration_us, gt=gt)


def still_gt(t):
    t = np.asarray(t, dtype=float)
    out = np.broadcast_to([0.5, 0.5], t.shape + (2,)).copy()
    return out[0] if t.ndim == 0 else out


def test_event_stats_splits_target_from_noise_by_distance():
    # 6 events on the target (3 px off centre), 4 far away, over one second
    xs = [323] * 6 + [10, 20, 30, 40]
    ys = [240] * 6 + [10, 20, 30, 40]
    ts = np.linspace(0, 999_999, 10).astype(np.int64)
    pols = [1, 1, 1, 1, 0, 0, 1, 1, 1, 1]
    clip = a_clip(xs, ys, ts, pols, still_gt)

    s = event_stats(clip, (640, 480), radius_px=50)
    assert s["target_rate"] == 6.0
    assert s["on_fraction"] == 4 / 6
    assert s["footprint_px"] == 3.0
    assert np.isclose(s["noise_rate_per_px"], 4 / (640 * 480 - np.pi * 50 ** 2))


def test_event_stats_skips_events_where_the_target_is_unknown():
    def gt(t):
        t = np.asarray(t, dtype=float)
        out = np.where(np.atleast_1d(t)[:, None] < 0.5, [0.5, 0.5], [np.nan, np.nan])
        return out[0] if t.ndim == 0 else out

    ts = np.array([100_000, 200_000, 700_000, 800_000], dtype=np.int64)
    clip = a_clip([320] * 4, [240] * 4, ts, [1] * 4, gt)
    s = event_stats(clip, (640, 480), radius_px=50)
    assert s["target_rate"] == 2.0
