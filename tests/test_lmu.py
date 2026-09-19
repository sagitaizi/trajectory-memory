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


# --- spiking population ----------------------------------------------------------------

def a_population(q=8, n_per_dim=200, theta=1.0, seed=0):
    from trajmem.lmu import build_population

    return build_population(q, theta, n_per_dim, radii=1.0, n_channels=1, seed=seed)


def run_both(weights, u_of_t, duration_s, dt=0.005, freeze_after=None):
    """Exact and spiking LMU on the same input; decoded states over time."""
    import torch

    from trajmem.lmu import SpikingLmu

    spiking = SpikingLmu(weights, dt=dt)
    exact = ExactLmu(weights.q, weights.theta, dt, n_channels=1)
    st_e, st_s = exact.init_state(1), spiking.init_state(1)
    t = np.arange(0, duration_s, dt)
    dec_e, dec_s, u_last = [], [], 0.0
    for ti in t:
        u_last = u_of_t(ti) if freeze_after is None or ti < freeze_after else u_last
        st_e = exact.step([[u_last]], st_e)
        act, st_s = spiking.step(torch.tensor([[u_last]], dtype=torch.float32), st_s)
        dec_e.append(st_e[0, 0]); dec_s.append(spiking.decode(act)[0, 0].numpy())
    return t, np.array(dec_e), np.array(dec_s), exact


def test_spiking_population_tracks_the_exact_lmu_and_reconstructs_a_delay():
    weights = a_population()
    t, dec_e, dec_s, exact = run_both(weights, lambda ti: 0.5 * np.sin(2 * np.pi * ti), 3.0)
    settled = t > 1.0
    assert np.sqrt(np.mean((dec_s[settled] - dec_e[settled]) ** 2)) < 0.05          # amplitude 0.5
    back = exact.reconstruct(dec_s[-1][None, None], 0.4)[0, 0]
    assert back == pytest.approx(0.5 * np.sin(2 * np.pi * (t[-1] - 0.4)), abs=0.08)


def test_spiking_population_keeps_the_window_sliding_when_the_input_is_frozen():
    weights = a_population()
    t, dec_e, dec_s, _ = run_both(weights, lambda ti: 0.5 * np.sin(2 * np.pi * ti), 3.0, freeze_after=2.0)
    frozen = t > 2.0
    assert np.sqrt(np.mean((dec_s[frozen] - dec_e[frozen]) ** 2)) < 0.06
    assert np.abs(dec_e[-1] - dec_e[np.searchsorted(t, 2.0)]).max() > 0.1           # the state did move


def test_torch_population_matches_nengo_simulating_the_same_network():
    import nengo
    import torch

    from trajmem.lmu import SpikingLmu, legendre_matrices

    q, n, theta, tau = 6, 200, 1.0, 0.02
    weights = a_population(q=q, n_per_dim=n, theta=theta, seed=3)
    a, b = legendre_matrices(q, theta)
    with nengo.Network(seed=3) as net:                       # same objects in the same order as the build
        gather = nengo.Node(size_in=q)
        ensembles = [nengo.Ensemble(n, 1, radius=1.0, neuron_type=nengo.LIF(), seed=3000 + i) for i in range(q)]
        for i, ens in enumerate(ensembles):
            nengo.Connection(ens, gather[i], synapse=None)
        inp = nengo.Node(lambda ti: 0.5 * np.sin(2 * np.pi * ti))
        scatter = nengo.Node(size_in=q)
        for i, ens in enumerate(ensembles):
            nengo.Connection(scatter[i], ens, synapse=None)
        nengo.Connection(gather, scatter, transform=tau * a + np.eye(q), synapse=tau)
        nengo.Connection(inp, scatter, transform=tau * b[:, None], synapse=tau)
        probe = nengo.Probe(gather, synapse=0.02)
    with nengo.Simulator(net, dt=0.001, progress_bar=False) as sim:
        assert np.allclose(sim.data[ensembles[0]].bias, weights.bias[:n].numpy(), atol=1e-5)
        sim.run(2.0)
        ref = sim.data[probe]
    spiking = SpikingLmu(weights, dt=0.005, substeps=5)
    st = spiking.init_state(1)
    out, alpha, y = [], 1 - np.exp(-0.005 / 0.02), np.zeros(q)
    for ti in np.arange(0, 2.0, 0.005):
        act, st = spiking.step(torch.tensor([[0.5 * np.sin(2 * np.pi * ti)]], dtype=torch.float32), st)
        y = y + alpha * (spiking.decode(act)[0, 0].numpy() - y)
        out.append(y.copy())
    out = np.array(out)
    late = slice(200, None)
    assert np.sqrt(np.mean((out[late] - ref[4::5][late]) ** 2)) < 0.05
