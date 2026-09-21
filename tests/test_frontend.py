import _thesis_path  # noqa: F401  (side effect: adds the thesis repo to sys.path)
import numpy as np
import pytest

from trajmem.data import Clip
from trajmem.frontend import to_frames, to_position, windows
from trajmem.simulate import EVENT_DTYPE

RES = (64, 48)


def events_at(xs, ys, ts_us, pols=None):
    ev = np.zeros(len(xs), dtype=EVENT_DTYPE)
    ev["x"], ev["y"], ev["timestamp"] = xs, ys, ts_us
    ev["polarity"] = 1 if pols is None else pols
    return ev[np.argsort(ev["timestamp"], kind="stable")]


def a_clip(ev, duration_us):
    return Clip(events=ev, duration_us=duration_us, gt=None, meta={"resolution": RES})


# --- windows -------------------------------------------------------------------

def test_windows_tile_the_clip_and_hand_out_each_window_once():
    ev = events_at([1] * 5, [1] * 5, [0, 900, 1000, 2500, 3999])
    got = list(windows(a_clip(ev, 4000), 1000))
    assert [t for t, _ in got] == [0, 1000, 2000, 3000]
    assert [len(w) for _, w in got] == [2, 1, 1, 1]


def test_windows_include_a_last_partial_window():
    ev = events_at([1] * 2, [1] * 2, [0, 2500])
    assert [t for t, _ in windows(a_clip(ev, 2600), 1000)] == [0, 1000, 2000]


# --- frames --------------------------------------------------------------------

def test_count_frames_split_polarity_and_count_every_event():
    ev = events_at([3, 3, 10], [4, 4, 20], [10, 20, 30], pols=[1, 1, 0])
    (t, frame), = list(to_frames(a_clip(ev, 1000), 1000, kind="count"))
    assert t == 0
    assert frame.shape == (2, 48, 64) and frame.dtype == np.float32
    assert frame[1, 4, 3] == 2 and frame[0, 20, 10] == 1
    assert frame.sum() == 3


def test_count_frames_can_be_block_downsampled_keeping_the_counts():
    ev = events_at([3, 2, 63], [4, 5, 47], [10, 20, 30])
    (_, frame), = list(to_frames(a_clip(ev, 1000), 1000, kind="count", downsample=4))
    assert frame.shape == (2, 12, 16)
    assert frame[1, 1, 0] == 2 and frame[1, 11, 15] == 1 and frame.sum() == 3


def test_surface_frames_decay_from_the_window_end():
    ev = events_at([3, 10], [4, 20], [500, 1500])
    frames = list(to_frames(a_clip(ev, 2000), 1000, kind="surface", tau_us=1000))
    (_, f0), (_, f1) = frames
    assert f0.shape == (48, 64) and 0 <= f0.min() and f0.max() <= 1
    assert np.isclose(f0[4, 3], np.exp(-0.5))           # 500 us old at the first window's end
    assert np.isclose(f1[20, 10], np.exp(-0.5)) and np.isclose(f1[4, 3], np.exp(-1.5))
    assert f0[20, 10] == 0                              # not happened yet


# --- position ------------------------------------------------------------------

def test_position_finds_the_dense_cluster_not_the_mean():
    rng = np.random.default_rng(0)
    n = 400
    tx, ty = 12 + rng.normal(0, 1.5, n), 30 + rng.normal(0, 1.5, n)         # target near (12, 30)
    nx, ny = rng.uniform(0, 64, n), rng.uniform(0, 48, n)                      # as much noise
    ev = events_at(np.clip(np.r_[tx, nx], 0, 63), np.clip(np.r_[ty, ny], 0, 47),
                   rng.integers(0, 5000, 2 * n))
    (t, x, y), = list(to_position(a_clip(ev, 5000), 5000))
    assert t == pytest.approx(0.0025)                                          # window centre, s
    assert abs(x * 64 - 12) < 1.5 and abs(y * 48 - 30) < 1.5
    mean_x = np.r_[tx, nx].mean()
    assert abs(mean_x - 12) > 5                                                # the mean is not


def test_position_is_unknown_for_an_empty_or_sparse_window():
    ev = events_at([1, 2], [1, 2], [0, 100])
    (_, x0, y0), (_, x1, y1) = list(to_position(a_clip(ev, 2000), 1000, min_events=5))
    assert np.isnan(x0) and np.isnan(y0) and np.isnan(x1)


def test_position_tracks_a_moving_cluster_window_by_window():
    rng = np.random.default_rng(1)
    xs, ys, ts = [], [], []
    for k in range(4):                                  # cluster moves 8 px per 1 ms window
        xs += list(10 + 8 * k + rng.normal(0, 1, 100)); ys += list(24 + rng.normal(0, 1, 100))
        ts += list(rng.integers(1000 * k, 1000 * (k + 1), 100))
    ev = events_at(np.clip(xs, 0, 63), np.clip(ys, 0, 47), ts)
    got = list(to_position(a_clip(ev, 4000), 1000))
    assert np.allclose([x * 64 for _, x, _ in got], [10, 18, 26, 34], atol=1.0)


def test_count_frames_reject_a_downsample_that_does_not_divide_the_sensor():
    ev = events_at([1], [1], [10])
    with pytest.raises(ValueError, match="divide"):
        list(to_frames(a_clip(ev, 1000), 1000, kind="count", downsample=5))   # 48 / 5


def test_without_hot_pixels_drops_stuck_pixels_on_a_long_clip_only():
    from trajmem.frontend import without_hot_pixels
    from trajmem.simulate import EVENT_DTYPE

    rng = np.random.default_rng(0)
    n_target, n_stuck = 2000, 3000
    ev = np.zeros(n_target + n_stuck, dtype=EVENT_DTYPE)
    ev["x"][:n_target] = rng.integers(100, 140, n_target)                # a target region, 40 x 40 px
    ev["y"][:n_target] = rng.integers(100, 140, n_target)
    ev["x"][n_target:], ev["y"][n_target:] = 5, 5                         # one stuck pixel, 300 ev/s
    ev["timestamp"] = rng.integers(0, 10_000_000, len(ev))
    ev.sort(order="timestamp")
    clip = Clip(events=ev, duration_us=10_000_000, gt=None, meta={"resolution": (640, 480)})
    out = without_hot_pixels(clip)
    assert len(out.events) == n_target and out.meta["hot_pixels"] == 1
    short = Clip(events=ev, duration_us=2_000_000, gt=None, meta={"resolution": (640, 480)})
    assert len(without_hot_pixels(short).events) == len(ev)
