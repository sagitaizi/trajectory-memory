import numpy as np
import pytest

from scripts.measure_px_per_tick import fit, marginal, step_shift, track_shift
from trajmem.simulate import EVENT_DTYPE


def a_ramp(n=640, centre=200, width=12):
    """A single bright bar, the thing the tracker actually locks onto."""
    m = np.zeros(n)
    m[centre - width // 2:centre + width // 2] = 1.0
    return np.convolve(m, np.ones(5) / 5, mode="same")


# --- the shift estimator -----------------------------------------------------

def test_step_shift_recovers_a_known_translation():
    a = a_ramp(centre=200)
    b = a_ramp(centre=205)
    lag, peak = step_shift(a, b)
    assert lag == pytest.approx(5.0, abs=0.5)
    assert peak > 0.9


def test_step_shift_recovers_a_negative_translation():
    lag, _ = step_shift(a_ramp(centre=300), a_ramp(centre=293))
    assert lag == pytest.approx(-7.0, abs=0.5)


def test_step_shift_reports_zero_for_an_unmoved_signal():
    lag, peak = step_shift(a_ramp(), a_ramp())
    assert lag == pytest.approx(0.0, abs=0.2)
    assert peak > 0.99


# --- the regression ----------------------------------------------------------

def test_fit_recovers_pixels_per_tick():
    ticks = np.linspace(0, 200, 50)
    slope, r = fit(0.9 * ticks + 3.0, ticks)
    assert slope == pytest.approx(0.9)
    assert r == pytest.approx(1.0)


def test_fit_declines_to_guess_when_the_axis_barely_moved():
    ticks = np.linspace(0, 5, 50)
    slope, r = fit(0.9 * ticks, ticks)
    assert np.isnan(slope) and np.isnan(r)


# --- marginals ---------------------------------------------------------------

def events_at(pixels, times_us):
    ev = np.zeros(len(pixels), dtype=EVENT_DTYPE)
    ev["x"] = [p[0] for p in pixels]
    ev["y"] = [p[1] for p in pixels]
    ev["timestamp"] = times_us
    return ev


def test_marginal_collapses_onto_the_requested_axis():
    ev = events_at([(10, 5), (10, 7), (10, 9)], [0, 0, 0])
    hot = np.zeros((480, 640), dtype=bool)
    assert marginal(ev, (640, 480), hot, 0.0, axis=0).sum() == pytest.approx(3.0)
    assert marginal(ev, (640, 480), hot, 0.0, axis=1).sum() == pytest.approx(3.0)


def test_marginal_drops_hot_pixels():
    ev = events_at([(10, 5), (99, 5)], [0, 0])
    hot = np.zeros((480, 640), dtype=bool)
    hot[5, 99] = True
    assert marginal(ev, (640, 480), hot, 0.0, axis=0).sum() == pytest.approx(1.0)


def test_track_shift_accumulates_step_by_step():
    """A bar marching 4 px per frame should integrate to 4, 8, 12 ..."""
    hot = np.zeros((480, 640), dtype=bool)
    pixels, times = [], []
    for i in range(4):
        for dx in range(12):
            pixels.append((100 + 4 * i + dx, 5))
            times.append(int(i * 0.1 * 1e6))
    ev = events_at(pixels, times)
    shift, quality = track_shift(ev, (640, 480), hot, np.arange(0, 0.4, 0.1), axis=0)
    assert shift == pytest.approx([0.0, 4.0, 8.0, 12.0], abs=0.6)
    assert quality > 0.9
