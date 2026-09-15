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
    # 6 events on the target (3 px off centre) and a uniform floor of one event per
    # pixel per second everywhere else, over one second
    rng = np.random.default_rng(0)
    n_noise = 640 * 480
    nx, ny = rng.integers(0, 640, n_noise), rng.integers(0, 480, n_noise)
    xs = np.concatenate([[323] * 6, nx])
    ys = np.concatenate([[240] * 6, ny])
    ts = np.sort(rng.integers(0, 1_000_000, len(xs))).astype(np.int64)
    pols = np.concatenate([[1, 1, 1, 1, 0, 0], rng.integers(0, 2, n_noise)])
    clip = a_clip(xs, ys, ts, pols, still_gt)

    s = event_stats(clip, (640, 480), radius_px=50)
    assert abs(s["target_rate"] - (6.0 + np.pi * 50 ** 2)) < 300   # target + the floor inside the disc
    assert np.isclose(s["noise_rate_per_px"], 1.0, atol=0.15)


def test_noise_floor_is_the_typical_tile_not_the_hot_spots():
    rng = np.random.default_rng(1)
    n_noise = 640 * 480 // 10                                   # 0.1 ev/px/s everywhere
    xs = list(rng.integers(0, 640, n_noise)) + [10] * 5000      # plus one hot pixel
    ys = list(rng.integers(0, 480, n_noise)) + [10] * 5000
    ts = np.sort(rng.integers(0, 1_000_000, len(xs))).astype(np.int64)
    clip = a_clip(xs, ys, ts, [1] * len(xs), still_gt)
    assert np.isclose(event_stats(clip, (640, 480), radius_px=50)["noise_rate_per_px"], 0.1, atol=0.03)


def test_event_stats_skips_events_where_the_target_is_unknown():
    def gt(t):
        t = np.asarray(t, dtype=float)
        out = np.where(np.atleast_1d(t)[:, None] < 0.5, [0.5, 0.5], [np.nan, np.nan])
        return out[0] if t.ndim == 0 else out

    ts = np.array([100_000, 200_000, 700_000, 800_000], dtype=np.int64)
    clip = a_clip([320] * 4, [240] * 4, ts, [1] * 4, gt)
    s = event_stats(clip, (640, 480), radius_px=50)
    assert s["target_rate"] == 2.0


def test_event_stats_measures_how_far_events_trail_behind_the_target():
    def moving_gt(t):                                   # 100 px/s along +x
        t = np.asarray(t, dtype=float)
        tt = np.atleast_1d(t)
        out = np.stack([0.25 + 100 / 640 * tt, np.full_like(tt, 0.5)], axis=-1)
        return out[0] if t.ndim == 0 else out

    ts = np.linspace(100_000, 900_000, 9).astype(np.int64)
    xs = np.rint(160 + 100 * ts / 1e6 - 10).astype(int)   # every event 10 px behind
    clip = a_clip(xs, [240] * 9, ts, [1] * 9, moving_gt)
    s = event_stats(clip, (640, 480), radius_px=50)
    assert np.isclose(s["trail_px"], 10.0, atol=0.6)

    ahead = a_clip(xs + 20, [240] * 9, ts, [1] * 9, moving_gt)   # 10 px ahead: no trail
    assert np.isclose(event_stats(ahead, (640, 480), radius_px=50)["trail_px"], 0.0, atol=0.6)
