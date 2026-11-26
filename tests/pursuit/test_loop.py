import numpy as np
import pytest

from pursuit.eye import VISUAL_DELAY_S
from pursuit.loop import PursuitMemory, follow

DT, D = 0.005, round(VISUAL_DELAY_S / 0.005)
CENTRE, SWING, PERIOD = np.array([320.0, 240.0]), np.array([200.0, 30.0]), 1.5
W = 2 * np.pi / PERIOD


def truth(t):
    return CENTRE + SWING * np.sin(W * np.asarray(t)[..., None])


def true_velocity(t):
    return SWING * W * np.cos(W * np.asarray(t)[..., None])


def a_swing(duration_s=9.0):
    t = np.arange(0, duration_s, DT) + DT / 2
    return np.column_stack([t, truth(t)])


class TrueVelocity:
    """A perfect memory: knows the motion, and that what it sees is one delay old."""

    def __init__(self, reads_ahead=True):
        self.reads_ahead = reads_ahead
        self.reset()

    def reset(self):
        self.n, self.seen = 0, []

    def observe(self, position, velocity):
        self.n += 1
        self.seen.append((position, velocity))

    def velocity(self, ahead_s):
        # the eye moves from this step's centre to the next: the speed midway
        t_seen = self.n * DT - VISUAL_DELAY_S
        return true_velocity(t_seen + (ahead_s if self.reads_ahead else 0.0))

    def surprise(self):
        return float(self.n)

    def period(self):
        return PERIOD


class Silent(TrueVelocity):
    def velocity(self, ahead_s):
        return np.array([np.nan, np.nan])


def test_a_perfect_memory_is_a_pursuit_memory():
    assert isinstance(TrueVelocity(), PursuitMemory)


def test_with_a_perfect_memory_the_eye_stays_on_the_target():
    tr = follow(TrueVelocity(), a_swing(), DT)
    assert np.abs(tr.gaze - tr.target).max() < 1.0
    assert np.abs(tr.reaction_velocity).max() < 0.01 * np.abs(tr.eye_velocity).max()


def test_without_reading_ahead_the_eye_is_one_delay_late():
    tr = follow(TrueVelocity(reads_ahead=False), a_swing(), DT, reaction_gain=0.0)
    late = true_velocity(tr.t + DT / 2 - VISUAL_DELAY_S)
    assert np.allclose(tr.eye_velocity, late)


def test_the_reaction_alone_follows_late_and_short():
    tr = follow(None, a_swing(), DT)
    assert np.all(tr.eye_velocity[:D] == 0.0) and np.all(tr.memory_velocity == 0.0)
    settled = tr.t > 4.5
    v, e = tr.target_velocity[settled, 0], tr.eye_velocity[settled, 0]
    lag = np.argmax([np.dot(v[: len(v) - k], e[k:]) / (len(v) - k) for k in range(150)]) * DT
    assert 0.15 < lag < 0.35 and 0.6 < e.std() / v.std() < 1.0


def test_the_memory_sees_the_true_motion_one_delay_late_whatever_the_eye_does():
    still, moving = Silent(), TrueVelocity()
    assert np.ptp(follow(still, a_swing(), DT, reaction_gain=0.0).gaze, axis=0).max() == 0.0
    follow(moving, a_swing(), DT, reaction_gain=2.0)
    p_still, v_still = (np.array(x) for x in zip(*still.seen))
    p_moving, v_moving = (np.array(x) for x in zip(*moving.seen))
    assert np.isnan(p_still[:D]).all() and np.isfinite(p_still[D:]).all()
    assert np.allclose(p_still[D:], p_moving[D:]) and np.allclose(v_still[D + 1:], v_moving[D + 1:])
    assert np.allclose(p_still[D:], truth(a_swing()[:-D, 0]))


def test_a_hidden_target_is_unseen_by_the_memory_one_delay_later():
    m = TrueVelocity()
    tr = follow(m, a_swing(), DT, blank=(3.0, 4.0))
    hidden = np.array([np.isnan(p).all() for p, _ in m.seen])
    t_hidden = tr.t[hidden & (tr.t > 1.0)]
    assert t_hidden[0] == pytest.approx(3.1, abs=DT) and t_hidden[-1] == pytest.approx(4.1, abs=DT)
    assert np.isfinite(tr.target).all() and np.isfinite(tr.gaze).all()


def test_positions_the_localiser_missed_do_not_break_the_eye():
    txy = a_swing()
    txy[::50, 1:] = np.nan
    tr = follow(None, txy, DT)
    assert np.isfinite(tr.gaze).all()
    clean = follow(None, a_swing(), DT)
    assert np.abs(tr.eye_velocity - clean.eye_velocity).max() < 0.15 * np.abs(clean.eye_velocity).max()


def test_a_memory_with_nothing_to_say_leaves_the_eye_to_the_reaction():
    assert np.allclose(follow(Silent(), a_swing(), DT).gaze, follow(None, a_swing(), DT).gaze)


def test_the_trace_keeps_the_memory_surprise_and_period():
    tr = follow(TrueVelocity(), a_swing(), DT)
    assert np.array_equal(tr.surprise, np.arange(1, len(tr.t) + 1)) and (tr.period == PERIOD).all()
    alone = follow(None, a_swing(), DT)
    assert np.isnan(alone.surprise).all() and np.isnan(alone.period).all()


def test_the_memory_is_reset_before_each_run():
    m = TrueVelocity()
    first = follow(m, a_swing(), DT)
    assert np.array_equal(first.gaze, follow(m, a_swing(), DT).gaze)
