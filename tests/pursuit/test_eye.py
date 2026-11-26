import numpy as np
import pytest

from pursuit.eye import REACTION_GAIN, VISUAL_DELAY_S, Delay, Eye, Reaction

DT = 0.005


def step_response(gain, duration_s=3.0):
    """Eye speed when a target starts at speed 1, seen one visual delay late."""
    delay, reaction, speed, out = Delay(VISUAL_DELAY_S, DT), Reaction(DT, gain), np.zeros(2), []
    for _ in range(round(duration_s / DT)):
        speed = reaction.step(delay.step(np.array([1.0, 0.0]) - speed))
        out.append(speed[0])
    return np.array(out)


def test_delay_returns_what_went_in_one_delay_ago():
    d = Delay(0.1, DT)
    out = [d.step(np.array([i, -i], dtype=float)) for i in range(50)]
    assert np.isnan(out[:20]).all()
    assert np.array_equal(out[20], [0.0, 0.0]) and np.array_equal(out[49], [29.0, -29.0])


def test_delay_keeps_its_own_copy_of_the_input():
    d, x = Delay(DT, DT), np.zeros(2)
    d.step(x)
    x += 5.0
    assert np.array_equal(d.step(x), [0.0, 0.0])


def test_a_zero_delay_passes_the_input_through():
    assert np.array_equal(Delay(0.0, DT).step(np.array([1.0, 2.0])), [1.0, 2.0])


def test_delay_refuses_a_delay_that_is_not_a_whole_number_of_steps():
    with pytest.raises(ValueError):
        Delay(0.0125, DT)


def test_reaction_changes_the_speed_in_proportion_to_the_slip():
    r = Reaction(DT, gain=2.0)
    r.step(np.array([10.0, -4.0]))
    assert np.allclose(r.step(np.array([10.0, -4.0])), [2 * 2.0 * 10 * DT, -2 * 2.0 * 4 * DT])


def test_reaction_brings_the_eye_to_the_targets_speed_without_overshoot():
    v = step_response(REACTION_GAIN)
    assert np.all(v[:20] == 0.0) and v[25] > 0.0
    assert v.max() < 1.001 and v[-1] == pytest.approx(1.0, abs=0.01)


def test_a_stronger_reaction_would_overshoot():
    assert step_response(1.5 * REACTION_GAIN).max() > 1.01


def test_reaction_keeps_its_speed_while_the_target_is_unseen():
    r = Reaction(DT, gain=2.0)
    before = r.step(np.array([10.0, 0.0]))
    assert np.array_equal(r.step(np.array([np.nan, np.nan])), before)


def test_eye_moves_at_the_commanded_speed_at_once():
    eye = Eye(DT, gaze=np.array([320.0, 240.0]))
    eye.step(np.array([100.0, -50.0]))
    assert np.allclose(eye.gaze, [320.5, 239.75])
