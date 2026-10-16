import _thesis_path  # noqa: F401  (side effect: adds the thesis repo to sys.path)
import numpy as np

from trajmem.data import Clip
from trajmem.frontend import LiveFrames, to_frames
from trajmem.simulate import EVENT_DTYPE

RES = (64, 48)


def random_events(n, duration_us, seed=0):
    rng = np.random.default_rng(seed)
    ev = np.zeros(n, dtype=EVENT_DTYPE)
    ev["x"], ev["y"] = rng.integers(0, RES[0], n), rng.integers(0, RES[1], n)
    ev["polarity"] = rng.integers(0, 2, n)
    ev["timestamp"] = np.sort(rng.integers(0, duration_us, n))
    return ev


def in_chunks(ev, seed=1):
    cuts = np.sort(np.random.default_rng(seed).integers(0, len(ev), 40))
    return np.split(ev, cuts)


def live_frames(ev, origin_us=0, **kw):
    shifted = ev.copy()
    shifted["timestamp"] += origin_us
    lf = LiveFrames(RES, 5000, **kw)
    return [(t0, frame) for chunk in in_chunks(shifted) for t0, _, frame in lf.push(chunk)]


def test_live_frames_match_the_offline_frames_whatever_the_chunking():
    ev = random_events(20_000, 3_000_000)
    ev["timestamp"] -= ev["timestamp"][0]
    offline = list(to_frames(Clip(ev, int(ev["timestamp"][-1]), None, meta={"resolution": RES}), 5000,
                             kind="count", downsample=8))
    live = live_frames(ev, downsample=8)
    assert len(live) == len(offline) - 1                  # the last window is still open
    for (t_live, f_live), (t_off, f_off) in zip(live, offline):
        assert t_live == t_off and np.array_equal(f_live, f_off)


def test_live_frames_count_time_from_the_first_event_like_a_loaded_recording():
    ev = random_events(5_000, 1_000_000)
    ev["timestamp"] -= ev["timestamp"][0]
    assert [t for t, _ in live_frames(ev, origin_us=123_456_789)] == [t for t, _ in live_frames(ev)]


def test_live_frames_hand_back_the_raw_window_events_rebased():
    ev = random_events(5_000, 1_000_000)
    ev["timestamp"] -= ev["timestamp"][0]
    lf = LiveFrames(RES, 5000)
    windows = [w for chunk in in_chunks(ev) for _, w, _ in lf.push(chunk)]
    joined = np.concatenate(windows)
    assert np.array_equal(joined, ev[:len(joined)])


def test_live_frames_drop_a_stuck_pixel_only_once_five_seconds_have_passed():
    ev = random_events(20_000, 8_000_000)
    stuck = np.zeros(2400, dtype=EVENT_DTYPE)                      # 300 ev/s at (5, 5)
    stuck["x"], stuck["y"], stuck["polarity"] = 5, 5, 1
    stuck["timestamp"] = np.linspace(0, 7_999_000, len(stuck)).astype(np.int64)
    ev = np.sort(np.concatenate([ev, stuck]), order="timestamp", kind="stable")
    ev["timestamp"] -= ev["timestamp"][0]
    stuck_px = [frame[1, 5, 5] for _, frame in live_frames(ev)]
    early, late = stuck_px[:900], stuck_px[1100:]                  # before 4.5 s, after 5.5 s
    assert sum(early) > 0 and sum(late) == 0


def test_to_event_array_restacks_a_dv_batch_into_the_clip_dtype():
    from trajmem.data import to_event_array

    dv_like = np.zeros(3, dtype=[("timestamp", "<i8"), ("x", "<i2"), ("y", "<i2"), ("polarity", "u1"),
                                 ("pad", "V3")])
    dv_like["timestamp"], dv_like["x"], dv_like["y"], dv_like["polarity"] = [1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 1]
    out = to_event_array(dv_like)
    assert out.dtype == EVENT_DTYPE
    assert list(out["timestamp"]) == [1, 2, 3] and list(out["x"]) == [4, 5, 6]
    assert list(out["y"]) == [7, 8, 9] and list(out["polarity"]) == [0, 1, 1]


# --- the live script -----------------------------------------------------------

def test_live_positions_match_the_offline_localiser_on_the_same_events():
    from scripts.run_live import live_positions
    from tests.test_experiment import a_clip_of_events, a_spec
    from trajmem.experiment import positions
    from trajmem.snn_localise import SpikingLocaliser

    clip = a_clip_of_events(a_spec(), duration_s=0.5)
    loc = SpikingLocaliser(n_cells=4, ch=(2, 3), hidden=8)
    offline = list(positions(clip, 5000, loc))
    live = [p[:3] for p in live_positions(in_chunks(clip.events), loc, 5000, clip.meta["resolution"])]
    assert len(live) == len(offline) - 1
    np.testing.assert_allclose(np.array(live, dtype=float), np.array(offline[:-1], dtype=float), equal_nan=True)


def test_the_live_view_scores_and_flags_exactly_as_the_recorded_clip_bench_does():
    from scripts import bench
    from scripts.run_live import LiveView
    from tests.test_experiment import a_clip_of_events, a_spec
    from trajmem.experiment import positions
    from trajmem.metrics import RatchetAlarm
    from trajmem.trajectories import Deviation

    clip = a_clip_of_events(a_spec([Deviation(at_t=6.0, kind="shrink", params={"factor": 0.4})]), duration_s=8.0)
    overlay = bench.make_overlays(clip, None, 5000, 0.1, 5.0, pipelines=bench.PIPELINES[1:2])[0]
    k = bench.Panels(clip, [overlay], "x", 0.02, 40, 1.0, 1.0).alarms[0].k
    offline = []
    for t in np.arange(0.1, 8.0, 0.1):                          # as the bench plays it, frame by frame
        overlay.at(t)
        offline += overlay.stepped
    alarm = RatchetAlarm(k=k)
    offline_flags = [alarm.update(score, tw) for score, tw in offline]

    view = LiveView(clip.meta["resolution"], horizon_s=0.1, warmup_s=5.0, k=k)
    live = [(view.step(t, x, y)["score"], view.flagged) for t, x, y in positions(clip, 5000)][:len(offline)]
    assert len(offline) > 1500
    np.testing.assert_allclose([s for s, _ in live], [s for s, _ in offline])
    assert [f for _, f in live] == offline_flags


def test_the_live_view_draws_the_deviation_mark_while_flagged():
    from scripts.run_live import LiveView

    view = LiveView((640, 480), horizon_s=0.1, warmup_s=5.0)
    view.step(0.0025, 0.5, 0.5)
    view.step(6.0025, 0.5, 0.5)                                   # past the warm-up, when the mark is drawn
    view.flagged = True
    frame = view.frame(np.zeros((480, 640), np.uint8), lag_ms=0.0)
    assert frame.shape[:2] == (480, 640) and list(frame[3, 50]) == [0, 0, 255]   # the red border


def test_the_live_view_forgets_the_path_on_reset():
    from scripts.run_live import LiveView

    view = LiveView((100, 100), horizon_s=0.1, warmup_s=3.0, k=3.0)
    for i in range(800):
        view.step(i * 0.005, 0.5, 0.5)
    assert np.isfinite(view.alarm.bar)
    view.reset()
    assert not view.flagged and not np.isfinite(view.alarm.bar)
    for i in range(800, 1300):                                    # the alarm's warm-up restarts too
        view.step(i * 0.005, 0.5, 0.5)
    assert not np.isfinite(view.alarm.bar)


def test_live_positions_skip_the_localiser_while_behind_and_mark_it():
    from scripts.run_live import live_positions
    from tests.test_experiment import a_clip_of_events, a_spec
    from trajmem.snn_localise import SpikingLocaliser

    clip = a_clip_of_events(a_spec(), duration_s=0.5)
    loc = SpikingLocaliser(n_cells=4, ch=(2, 3), hidden=8)
    stalled = iter(np.arange(1000) * 0.05)                        # wall clock: 50 ms per 5 ms window
    out = list(live_positions(in_chunks(clip.events), loc, 5000, clip.meta["resolution"],
                              max_lag_s=0.1, clock=lambda: next(stalled)))
    skipped = [p[4] for p in out]
    assert not any(skipped[:2]) and all(skipped[5:])
    assert all(np.isnan(p[1]) and np.isnan(p[2]) for p in out if p[4])
    assert not any(p[4] for p in live_positions(in_chunks(clip.events), loc, 5000, clip.meta["resolution"]))


def test_the_live_view_draws_nothing_of_the_memory_while_it_learns_the_path():
    from scripts.run_live import LiveView

    view = LiveView((640, 480), horizon_s=0.1, warmup_s=5.0)
    blank = np.zeros((480, 640), np.uint8)
    for i in range(400):                                          # 2 s in: still learning
        view.step(i * 0.005 + 0.0025, 0.5, 0.5)
    learning = view.frame(blank, lag_ms=0.0)
    assert view.learning
    centre = (slice(220, 260), slice(300, 340), 0)                # where the magenta prediction would be
    assert not (learning[centre] == 255).any()
    for i in range(400, 1200):                                    # 6 s in: drawn
        view.step(i * 0.005 + 0.0025, 0.5, 0.5)
    assert not view.learning
    assert (view.frame(blank, lag_ms=0.0)[centre] == 255).any()
    view.reset()                                                  # r: learning again
    view.step(6.0025, 0.5, 0.5)
    assert view.learning


def test_lag_starts_at_zero_even_when_the_first_window_came_late():
    from scripts.run_live import Lag

    walls = iter([1.0, 1.0, 1.0, 1.02, 1.5])                      # the first window was 0.5 s late
    lag = Lag(clock=lambda: next(walls))
    out = [lag(t) for t in (0.5, 0.505, 0.51, 0.52, 0.53)]
    assert out[0] == 0 and min(out) >= 0
    assert abs(out[-1] - 0.48) < 1e-9                             # 0.97 s gap less the best, 0.49 s


def test_the_live_view_keeps_no_growing_history():
    from scripts.run_live import LiveView

    view = LiveView((640, 480), horizon_s=0.1, warmup_s=5.0)
    for i in range(300):
        view.step(i * 0.005 + 0.0025, 0.5, 0.5)
    assert len(view.overlay.stepped) == 0


def test_live_frames_reject_a_downsample_that_does_not_divide_the_sensor():
    import pytest

    with pytest.raises(ValueError, match="divide"):
        LiveFrames((346, 260), 5000, downsample=8)


def test_recording_a_replay_is_refused():
    import pytest

    from scripts.run_live import main

    with pytest.raises(SystemExit):
        main(["--replay", "a.aedat4", "--record", "a.aedat4"])
