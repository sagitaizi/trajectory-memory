import numpy as np
import pytest

from pursuit.velocity import Rate, Smooth

DT = 0.005


def test_rate_is_the_change_per_second():
    r = Rate(DT)
    r.step(np.array([1.0, 2.0]))
    assert np.allclose(r.step(np.array([2.0, 1.0])), [200.0, -200.0])


def test_rate_is_unknown_at_the_first_sample_and_after_a_gap():
    r, gap = Rate(DT), np.array([np.nan, np.nan])
    out = [r.step(x) for x in (np.zeros(2), gap, np.ones(2), np.ones(2))]
    assert [bool(np.isfinite(v).all()) for v in out] == [False, False, False, True]


def test_smooth_settles_on_a_constant_and_lags_a_ramp_by_its_time_constant():
    s = Smooth(DT, tau_s=0.02)
    for _ in range(200):
        out = s.step(np.array([3.0, -1.0]))
    assert np.allclose(out, [3.0, -1.0])
    ramp = [s.step(np.array([3.0 + 10.0 * i * DT, -1.0]))[0] for i in range(400)]
    assert 3.0 + 10.0 * 399 * DT - ramp[-1] == pytest.approx(10.0 * (0.02 - DT), rel=0.01)


def test_smooth_reports_a_gap_as_unknown_and_carries_on_after_it():
    s, gap = Smooth(DT, tau_s=0.02), np.array([np.nan, np.nan])
    s.step(np.ones(2))
    assert np.isnan(s.step(gap)).all()
    assert np.allclose(s.step(np.ones(2)), 1.0)
