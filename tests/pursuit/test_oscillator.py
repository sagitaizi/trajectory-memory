import numpy as np
import pytest

from pursuit.oscillator import OscillatorBank

DT = 0.005


def run(bank, period_s, driven_s, free_s=0.0):
    """(t, swing, phase error to the drive, period) of the bank's first clock."""
    bank.reset()
    out = []
    for i in range(round((driven_s + free_s) / DT)):
        t = i * DT
        drive = np.cos(2 * np.pi * t / period_s) if t < driven_s else np.nan
        bank.step(drive)
        err = (bank.phase[0] - 2 * np.pi * t / period_s + np.pi) % (2 * np.pi) - np.pi
        out.append((t, bank.swing[0], err, 2 * np.pi / bank.omega[0]))
    return np.array(out)


@pytest.mark.parametrize("period_s", [1.2, 4.0])
def test_a_driven_clock_holds_its_swing_in_step_with_the_drive(period_s):
    o = run(OscillatorBank(DT, (period_s * 1.1,)), period_s, 8 * period_s)
    late = o[o[:, 0] > 4 * period_s]
    assert 0.85 < late[:, 1].mean() < 1.1
    assert abs(late[:, 2].mean()) < 0.6
    assert late[:, 3].mean() == pytest.approx(period_s, rel=0.06)


@pytest.mark.parametrize("cycles", [0.5, 1.0, 2.0])
def test_an_undriven_clock_fades_to_about_a_third_in_its_damping_time(cycles):
    T = 1.2
    o = run(OscillatorBank(DT, (T,), damping_cycles=cycles), T, 6 * T, 1.2 * cycles * T)
    at = np.abs(o[:, 0] - (6 + cycles) * T) < 0.1 * cycles * T
    before = (o[:, 0] > 4 * T) & (o[:, 0] < 6 * T)
    assert o[at, 1].mean() / o[before, 1].mean() == pytest.approx(0.37, abs=0.1)


def test_without_damping_the_clock_keeps_its_swing():
    o = run(OscillatorBank(DT, (1.2,), damping_cycles=None), 1.2, 6.0, 3.6)
    assert o[-100:, 1].mean() > 0.9


def test_clocks_of_a_bank_run_side_by_side():
    bank = OscillatorBank(DT, (0.9, 2.5))
    run(bank, 1.2, 3.0)
    assert bank.swing.shape == bank.phase.shape == bank.omega.shape == (2,)
    assert bank.omega[0] > bank.omega[1]


def test_reset_starts_the_same_run_again():
    bank = OscillatorBank(DT, (1.2,))
    assert np.array_equal(run(bank, 1.2, 2.0), run(bank, 1.2, 2.0))


def test_reading_ahead_says_where_a_driven_clock_will_be():
    bank, T, ahead = OscillatorBank(DT, (1.2,)), 1.2, 20
    read, state = [], []
    for i in range(round(9.6 / DT)):
        bank.step(np.cos(2 * np.pi * i * DT / T))
        read.append(bank.ahead(ahead * DT)[0])
        state.append(bank.xy[0])
    read, state = np.array(read)[1000:-ahead], np.array(state)[1000 + ahead:]
    assert np.sqrt(np.mean(np.sum((read - state) ** 2, axis=1))) < 0.1
    plain = np.hypot(*read.T).mean()
    assert plain == pytest.approx(np.hypot(*state.T).mean(), rel=0.03)


def test_reading_ahead_fades_when_the_target_is_unseen_and_turns_only_when_undamped():
    for cycles, expected in ((1.0, np.exp(-0.5)), (None, 1.0)):
        bank = OscillatorBank(DT, (1.2,), damping_cycles=cycles)
        run(bank, 1.2, 4.8)
        bank.step(np.nan)
        half_cycle = np.pi / bank.omega[0]
        assert np.allclose(bank.ahead(half_cycle)[0], -expected * bank.xy[0], atol=1e-6)
