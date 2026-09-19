import _thesis_path  # noqa: F401  (side effect: adds the thesis repo to sys.path)
import numpy as np

from tests.test_snn import a_track
from trajmem.augment import augment, flips, scales, shifts, stretches
from trajmem.snn import path_position, path_targets
from trajmem.trajectories import Deviation


def test_flips_mirror_both_tracks_consistently():
    tr = a_track(center=(0.3, 0.6))
    out = flips([tr])
    assert [o.name.split("flip")[1] for o in out] == ["-1+1", "+1-1", "-1-1"]
    assert np.allclose(out[0].gt[:, 0], 1 - tr.gt[:, 0]) and np.allclose(out[0].gt[:, 1], tr.gt[:, 1])
    assert np.allclose(out[2].obs, 1 - tr.obs)


def test_shifts_and_scales_keep_the_path_inside_the_image():
    tr = a_track(center=(0.5, 0.5))
    rng = np.random.default_rng(0)
    for o in shifts([tr], 20, rng) + scales([tr], 20, rng, lo=0.5, hi=3.0):
        assert o.gt.min() >= 0.05 - 1e-9 and o.gt.max() <= 0.95 + 1e-9
        assert np.allclose(o.obs - o.gt, tr.obs - tr.gt)       # measurement error carried over
    sc = scales([tr], 1, np.random.default_rng(1), lo=0.5, hi=0.5)[0]
    assert np.allclose(sc.gt.max(0) - sc.gt.min(0), 0.5 * (tr.gt.max(0) - tr.gt.min(0)))


def test_stretch_rescales_period_and_break_and_marks_the_tail_unseen():
    tr = a_track(period_s=1.0, deviations=[Deviation(at_t=4.0, kind="shrink", params={"factor": 0.5})])
    st = stretches([tr], 1, np.random.default_rng(3), lo=1.25, hi=1.25)[0]
    assert st.period_s == 0.8 and st.deviation_t == 3.2
    assert np.isnan(st.gt[-1]).all() and np.isfinite(st.gt[0]).all()
    p = path_targets(st.t, st.gt, st.period_s, until_s=st.deviation_t)     # NaN rows ignored
    ok = np.isfinite(st.gt).all(1) & (st.t < 3.2)
    assert np.abs(path_position(p, 0.0)[ok] - st.gt[ok]).max() < 0.02


def test_augment_spec_counts():
    tr = [a_track(name="a"), a_track(name="b")]
    out = augment(tr, "flips,shift:2,scale:1,stretch:3")
    assert len(out) == 2 * (1 + 3 + 2 + 1 + 3)
    assert augment(tr, "") == tr
