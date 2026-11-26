import numpy as np
import pytest

from pursuit.clock import ClockMemory
from pursuit.loop import PursuitMemory, follow
from pursuit.measures import gain_lag

DT = 0.005
CENTRE, SWING = np.array([320.0, 240.0]), np.array([200.0, 30.0])
CLOCKS = (1.3, 2.0)


def swing_txy(period_s=1.2, duration_s=14.0, change=None):
    """`change` = (t, factor): the rate is multiplied by factor at t."""
    t = np.arange(0, duration_s, DT) + DT / 2
    rate = np.full_like(t, 2 * np.pi / period_s)
    if change is not None:
        rate[t >= change[0]] *= change[1]
    return np.column_stack([t, CENTRE + np.outer(np.sin(np.cumsum(rate) * DT), SWING)])


def a_clock(**kw):
    return ClockMemory(DT, periods_s=CLOCKS, **kw)


@pytest.fixture(scope="module")
def steady():
    m = a_clock()
    return m, follow(m, swing_txy(), DT)


def test_it_is_a_pursuit_memory():
    assert isinstance(a_clock(), PursuitMemory)


def test_it_says_nothing_before_it_has_seen_the_target():
    m = a_clock()
    assert np.isnan(m.velocity(0.1)).all() and np.isnan(m.period())
    m.observe(np.full(2, np.nan), np.full(2, np.nan))
    assert np.isnan(m.velocity(0.1)).all()


def test_it_chooses_a_clock_from_the_start_but_judges_nothing_until_it_has_settled():
    m = a_clock()
    follow(m, swing_txy(duration_s=0.5), DT)
    assert np.isfinite(m.period()) and np.isnan(m.surprise())


def test_it_saves_the_slow_copy_three_cycles_after_it_first_moves_the_eye():
    tr = follow(ClockMemory(DT), swing_txy(duration_s=10.0), DT)
    first = np.flatnonzero(np.any(tr.memory_velocity != 0, axis=1))[0]
    saved = np.flatnonzero(np.isfinite(tr.surprise))[0]
    assert tr.t[saved] - tr.t[first] >= 3 * tr.period[saved] - DT


def test_once_it_moves_the_eye_on_a_steady_swing_it_keeps_doing_so():
    tr = follow(ClockMemory(DT), swing_txy(duration_s=10.0), DT)
    on = np.any(tr.memory_velocity != 0, axis=1)
    assert on[np.flatnonzero(on)[0]:].mean() > 0.95


def test_on_a_slow_swing_it_starts_moving_the_eye_within_four_cycles():
    tr = follow(ClockMemory(DT), swing_txy(period_s=4.0, duration_s=16.0), DT)
    assert np.any(tr.memory_velocity[tr.t < 16.0] != 0)


def test_after_a_gap_its_check_waits_until_both_sides_were_made_on_a_seen_target():
    m = a_clock()
    follow(m, swing_txy(duration_s=5.0), DT)
    for _ in range(50):
        m.observe(np.full(2, np.nan), np.full(2, np.nan))
    before = m.predicted.copy()
    for _ in range(m.check_steps - 1):
        m.observe(np.array([320.0, 240.0]), np.array([100.0, 0.0]))
    assert np.array_equal(m.predicted, before)


def test_it_never_moves_the_eye_when_keeping_the_current_speed_predicts_as_well():
    t = np.arange(0, 4.0, DT) + DT / 2
    tr = follow(a_clock(), np.column_stack([t, 100 + 50 * t, np.full_like(t, 240.0)]), DT)
    assert (tr.memory_velocity == 0).all()


def test_it_does_not_overshoot_while_its_clock_falls_into_step():
    tr = follow(ClockMemory(DT), swing_txy(duration_s=6.0), DT)
    for k in range(4):
        cycle = (tr.t >= k * 1.2) & (tr.t < (k + 1) * 1.2)
        assert gain_lag(tr.t[cycle], tr.eye_velocity[cycle], tr.target_velocity[cycle], 1.2)[0] < 1.15


def test_it_locks_on_and_drives_the_eye_at_the_target_velocity(steady):
    m, tr = steady
    late = tr.t > 8.0
    for velocity in (tr.memory_velocity, tr.eye_velocity):
        gain, lag = gain_lag(tr.t[late], velocity[late], tr.target_velocity[late], 1.2)
        assert gain == pytest.approx(1.0, abs=0.1) and abs(lag) < 0.03
    assert m.period() == pytest.approx(1.2, rel=0.06)


def test_the_reaction_has_little_left_to_do_once_it_has_locked_on(steady):
    _, tr = steady
    late = tr.t > 8.0
    assert tr.reaction_velocity[late].std(axis=0)[0] < 0.3 * tr.target_velocity[late].std(axis=0)[0]


def test_while_the_target_is_hidden_it_goes_on_through_a_turnaround_and_fades():
    m = a_clock()
    tr = follow(m, swing_txy(), DT, blank=(9.0, 12.0))
    before = tr.memory_velocity[(tr.t > 7) & (tr.t < 9), 0].std()
    first, second = ((tr.t > 9.2 + k * 1.2) & (tr.t < 10.4 + k * 1.2) for k in (0, 1))
    assert np.ptp(np.sign(tr.memory_velocity[first, 0])) == 2
    assert 0.3 * before < tr.memory_velocity[first, 0].std() < 0.95 * before
    assert tr.memory_velocity[second, 0].std() < 0.6 * tr.memory_velocity[first, 0].std()


def test_without_damping_it_does_not_fade():
    m = a_clock(damping_cycles=None)
    tr = follow(m, swing_txy(), DT, blank=(9.0, 12.0))
    before = tr.memory_velocity[(tr.t > 7) & (tr.t < 9), 0].std()
    assert tr.memory_velocity[(tr.t > 10.4) & (tr.t < 12), 0].std() > 0.85 * before


def test_it_learns_nothing_while_the_target_is_hidden():
    m = a_clock()
    follow(m, swing_txy(duration_s=5.0), DT)
    before = m.map.clone()
    for _ in range(50):
        m.observe(np.full(2, np.nan), np.full(2, np.nan))
    assert (m.map == before).all()


def test_surprise_is_low_on_a_steady_motion_and_rises_when_the_rhythm_changes():
    tr = follow(a_clock(), swing_txy(duration_s=20.0, change=(16.0, 1.5)), DT)
    steady, after = tr.surprise[(tr.t > 13) & (tr.t < 16)], tr.surprise[(tr.t > 16.3) & (tr.t < 17.5)]
    assert np.isnan(tr.surprise[tr.t < 3 * 1.2]).all() and np.median(after) > 3 * np.median(steady)


def test_surprise_stays_finite_over_a_gap():
    m = a_clock()
    follow(m, swing_txy(), DT, blank=(11.0, 12.0))
    assert m.slow is not None and np.isfinite(m.surprise())


def test_the_same_run_twice_is_the_same():
    m, txy = a_clock(), swing_txy(duration_s=5.0)
    assert np.array_equal(follow(m, txy, DT).gaze, follow(m, txy, DT).gaze)
