import _thesis_path  # noqa: F401  (side effect: adds the thesis repo to sys.path)
import numpy as np
import pytest

from trajmem.lmu import ExactLmu, delay_readout, legendre_matrices


def test_legendre_matrices_have_the_published_entries():
    a, b = legendre_matrices(4, theta=2.0)
    assert a.shape == (4, 4) and b.shape == (4,)
    assert a[0, 0] == -0.5 and a[1, 0] == 1.5 and a[0, 1] == -0.5 and a[1, 1] == -1.5
    assert b[0] == 0.5 and b[1] == -1.5 and b[2] == 2.5
    assert np.allclose(delay_readout(4, 2.0, 0.0), [1, -1, 1, -1])       # no lag: u(t) = sum (-1)^i m_i


def test_exact_lmu_reconstructs_a_delayed_sinusoid():
    dt = 0.005
    lmu = ExactLmu(q=24, theta=2.0, dt=dt, n_channels=1)
    state = lmu.init_state(1)
    t = np.arange(0, 3.0, dt)
    for ti in t:
        state = lmu.step([[np.sin(2 * np.pi * ti)]], state)
    for lag in (0.0, 0.5, 1.3):
        assert lmu.reconstruct(state, lag)[0, 0] == pytest.approx(np.sin(2 * np.pi * (t[-1] - lag)), abs=0.02)
    lags = np.linspace(0, 2.0, 5)
    assert lmu.reconstruct(state, lags).shape == (1, 1, 5)
