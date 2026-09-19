import _thesis_path  # noqa: F401  (side effect: adds the thesis repo to sys.path)
import numpy as np

from tests.test_experiment import a_clip_of_events, a_spec
from trajmem.localise import (FrameCentroid, FrameSet, evaluate_localiser, frame_set,
                              load_frame_set, score_track)
from trajmem.model import Localiser


def test_frame_set_matches_make_frames_layout_and_truth():
    clip = a_clip_of_events(a_spec(), duration_s=0.2)
    fs = frame_set(clip, window_us=5000, downsample=8, name="x")
    assert fs.frames.shape == (40, 2, 60, 80) and fs.frames.dtype == np.uint8
    assert fs.t[0] == 0.0025 and fs.gt.shape == (40, 2)


def test_frame_centroid_finds_a_clean_blob_and_reports_empty_frames(tmp_path):
    clip = a_clip_of_events(a_spec(), duration_s=1.0)
    fs = frame_set(clip, 5000, 8)
    loc = FrameCentroid(min_events=5)
    assert isinstance(loc, Localiser)
    track = evaluate_localiser(loc, fs)
    r = score_track(track, fs)
    assert r["median"] < 6.0 and r["unseen"] == 0.0 and r["n"] == 200
    empty = FrameSet("e", np.zeros((3, 2, 60, 80), np.uint8), np.arange(3) * 0.005, np.full((3, 2), 0.5), 5000, 8)
    assert np.isnan(evaluate_localiser(loc, empty)).all() and score_track(evaluate_localiser(loc, empty), empty)["unseen"] == 1.0
    np.savez(tmp_path / "sim_000.npz", frames=fs.frames, t=fs.t, gt=fs.gt)
    back = load_frame_set(tmp_path / "sim_000.npz", 5000, 8)
    assert back.name == "sim_000" and np.array_equal(back.frames, fs.frames)
