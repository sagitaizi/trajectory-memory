import _thesis_path  # noqa: F401  (side effect: adds the thesis repo to sys.path)
import numpy as np

from scripts.make_frames import frames_of
from scripts.make_tracks import track_of
from tests.test_experiment import a_clip_of_events, a_spec
from trajmem.trajectories import Deviation


def test_track_of_pairs_measured_and_true_positions_per_window():
    spec = a_spec(deviations=[Deviation(at_t=6.0, kind="shrink", params={"factor": 0.5})])
    clip = a_clip_of_events(spec, duration_s=8.0)
    tr = track_of(clip, window_us=5000)
    assert tr["t"].shape == (1600,) and tr["obs"].shape == tr["gt"].shape == (1600, 2)
    assert tr["t"][0] == 0.0025 and tr["deviation_t"] == 6.0
    err = np.hypot(*((tr["obs"] - tr["gt"]) * (640, 480)).T)
    assert np.nanmedian(err) < 3.0


def test_frames_of_gives_downsampled_uint8_counts_with_truth_per_window():
    clip = a_clip_of_events(a_spec(), duration_s=0.1)
    fr = frames_of(clip, window_us=5000, downsample=8)
    assert fr["frames"].shape == (20, 2, 60, 80) and fr["frames"].dtype == np.uint8
    assert fr["gt"].shape == (20, 2) and fr["t"].shape == (20,)
    assert fr["frames"].sum() == len(clip.events)               # no count lost or clipped
    ys, xs = np.nonzero(fr["frames"][0].sum(axis=0))
    assert abs(xs.mean() / 80 - fr["gt"][0, 0]) < 0.03           # counts sit on the truth
