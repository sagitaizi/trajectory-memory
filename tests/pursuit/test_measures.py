import numpy as np
import pytest

from pursuit.measures import alarms, gain_lag

T = np.arange(0, 12, 0.005)
PERIOD = 1.5
AXIS = np.array([0.8, 0.6])


def swing(lag_s=0.0, gain=1.0):
    return np.outer(gain * np.cos(2 * np.pi * (T - lag_s) / PERIOD), AXIS)


@pytest.mark.parametrize("gain, lag_s", [(1.0, 0.0), (0.8, 0.1), (1.1, -0.05)])
def test_gain_and_lag_of_a_shifted_scaled_copy(gain, lag_s):
    g, lag = gain_lag(T, swing(lag_s, gain), swing(), PERIOD)
    assert g == pytest.approx(gain, rel=1e-3) and lag == pytest.approx(lag_s, abs=1e-3)


def test_lag_does_not_depend_on_where_in_the_cycle_the_run_starts():
    k = 77
    g, lag = gain_lag(T[k:], swing(0.1)[k:], swing()[k:], PERIOD)
    assert lag == pytest.approx(0.1, abs=1e-3)


def test_unseen_steps_and_noise_are_left_out_of_it():
    rng = np.random.default_rng(0)
    target = swing() + rng.normal(0, 1.0, (len(T), 2))
    target[100:140] = np.nan
    g, lag = gain_lag(T, swing(0.1), target, PERIOD)
    assert g == pytest.approx(1.0, abs=0.06) and lag == pytest.approx(0.1, abs=0.02)


def test_eye_movement_across_the_motion_does_not_count():
    across = np.outer(np.sin(2 * np.pi * T / PERIOD), [-0.6, 0.8])
    g, lag = gain_lag(T, swing() + across, swing(), PERIOD)
    assert g == pytest.approx(1.0, abs=1e-3) and lag == pytest.approx(0.0, abs=1e-3)


def a_score(jump_at=None, duration_s=20.0, seed=0):
    t = np.arange(0, duration_s, 0.005)
    s = 10.0 + np.random.default_rng(seed).normal(0, 1.0, len(t))
    s[t < 6.0] = np.nan
    if jump_at is not None:
        s[t >= jump_at] += 100.0
    return t, s


def test_a_steady_score_raises_no_alarm():
    assert alarms(*a_score(), 0.005, k=10.0, floor=1.0) == []


def test_a_jump_in_the_score_raises_one_alarm_where_it_starts():
    out = alarms(*a_score(jump_at=15.0), 0.005, k=10.0, floor=1.0)
    assert len(out) == 1 and out[0] == pytest.approx(15.01, abs=0.006)


def test_there_is_no_alarm_before_the_score_has_filled_the_window():
    t, s = a_score()
    s[t < 14.0] = np.nan
    s[(t > 15.0) & (t < 15.5)] = 500.0
    assert alarms(t, s, 0.005, k=10.0, floor=1.0) == []


def test_the_chosen_alarm_constant_is_the_default():
    assert alarms(*a_score(jump_at=15.0), 0.005) == alarms(*a_score(jump_at=15.0), 0.005, k=10.0, floor=0.0)


def test_below_the_floor_nothing_is_an_alarm():
    assert alarms(*a_score(jump_at=15.0), 0.005, k=10.0, floor=500.0) == []
