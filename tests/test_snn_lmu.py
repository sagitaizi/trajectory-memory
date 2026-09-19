import _thesis_path  # noqa: F401  (side effect: adds the thesis repo to sys.path)
import numpy as np
import pytest
import torch

from trajmem.model import TrajectoryMemory
from trajmem.snn_lmu import LmuMemory, LmuNet
from test_snn import DT, a_track


def tiny_memory(**kw):
    return LmuMemory(dt_s=DT, q=6, n_per_dim=30, theta_s=1.5, n_per_axis=12, n_fast=48, device="cpu", **kw)


def test_lmu_net_steps_and_carries_state():
    net = LmuNet(n_in=10, n_fast=16, n_lmu=20, dt_s=DT, tau_fast=(0.01, 0.025), readout_tau_s=0.015,
                 n_horizons=4, n_cells=6, n_path=17, n_short=2)
    state = net.init_state(3, "cpu")
    heads, path, state, rate = net.step(torch.rand(3, 10), torch.rand(3, 20) * 100, state)
    assert heads.shape == (3, 4, 2, 6) and path.shape == (3, 17) and 0.0 <= rate.item() <= 1.0
    heads2, _, state2, _ = net.step(torch.rand(3, 10), torch.rand(3, 20) * 100, state)
    assert not torch.equal(state2[2], state[2])                                # the readout trace moved


def test_lmu_memory_satisfies_the_protocol_and_is_empty_before_observing():
    m = tiny_memory()
    assert isinstance(m, TrajectoryMemory)
    m.reset()
    assert np.isnan(m.predict(0.1)).all() and m.deviation_score() == 0.0
    assert np.isnan(m.period()) and m.path_points([0.0]) is None
    m.observe(0.5, 0.5)
    assert np.isfinite(m.predict(0.1)).all()


def test_unseen_steps_are_fed_the_networks_own_position_estimate():
    m = tiny_memory()
    m.reset()
    tr = a_track(duration_s=1.0)
    for x, y in tr.obs[:100]:
        m.observe(x, y)
    seen_u = m.last_u.copy()
    preds = []
    for _ in range(60):                                       # 0.3 s blank
        m.observe(np.nan, np.nan)
        preds.append(m.predict(0.1))
        assert np.isfinite(m.last_u).all()                    # the LMU kept receiving a position
    assert np.isfinite(preds).all()
    assert np.hypot(*(m.last_u - seen_u)) < 0.5               # a sane position, not garbage


def test_fit_lowers_the_error_and_logs_the_path_px():
    torch.manual_seed(0)
    tracks = [a_track(name=f"c{i}", period_s=1.0 + 0.1 * i, center=(0.4 + 0.05 * i, 0.5)) for i in range(6)]
    m = tiny_memory()
    m.set_radii_from(tracks)
    log = m.fit(tracks, epochs=12, chunk_s=2.0, batch=6, val_fraction=1 / 6, lr=1e-2, seed=0, blank_prob=0.5)
    assert log[-1]["val_px"] < log[0]["val_px"] * 0.6
    assert np.isfinite(log[-1]["val_path_px"])
    tr = tracks[0]
    m.reset()
    err = []
    for i, (x, y) in enumerate(tr.obs):
        m.observe(x, y)
        if tr.t[i] > 2.0 and i + 20 < len(tr.t):
            err.append(np.hypot(*(np.array(m.predict(0.1)) - tr.gt[i + 20])))
    assert np.median(err) * 640 < 30


def test_checkpoint_round_trip_keeps_the_predictions(tmp_path):
    torch.manual_seed(0)
    m = tiny_memory()
    m.path_mean, m.path_std = np.arange(17, dtype=float), np.ones(17) * 2
    m.save(tmp_path / "lmu.pt")
    m2 = LmuMemory.load(tmp_path / "lmu.pt")
    tr = a_track(duration_s=0.5)
    out = []
    for mem in (m, m2):
        mem.reset()
        for x, y in tr.obs:
            mem.observe(x, y)
        out.append((mem.predict(0.1), mem.last_path.copy()))
    assert np.allclose(out[0][0], out[1][0]) and np.allclose(out[0][1], out[1][1])
    assert m2.q == 6 and m2.n_per_dim == 30 and m2.theta_s == 1.5
