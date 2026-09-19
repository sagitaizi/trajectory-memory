import _thesis_path  # noqa: F401  (side effect: adds the thesis repo to sys.path)
import numpy as np
import pytest
import torch

from trajmem.data import Track
from trajmem.model import TrajectoryMemory
from trajmem.snn import (N_PATH, AdaptiveLeaky, PlaceCells, SpikingMemory, TwoLayerNet,
                         path_position, path_targets)
from trajmem.trajectories import Deviation, TrajectorySpec, sample

DT = 0.005


def a_track(name="a", period_s=1.0, duration_s=6.0, deviations=None, shape="ellipse",
            center=(0.5, 0.5), noise=0.0, seed=0):
    spec = TrajectorySpec(shape=shape, size=(0.2, 0.12), period_s=period_s, center=center,
                          rotation=0.4, deviations=deviations or [])
    t = np.arange(0, duration_s, DT) + DT / 2
    gt = sample(spec, t)
    obs = gt + np.random.default_rng(seed).normal(0, noise, gt.shape)
    dev = float(min(spec.deviations, key=lambda d: d.at_t).at_t) if spec.deviations else np.nan
    return Track(name=name, t=t, obs=obs, gt=gt, deviation_t=dev, period_s=period_s)


# --- place-cell encoder ---------------------------------------------------------

def test_place_cells_light_up_the_neighbours_of_the_position():
    enc = PlaceCells(n_per_axis=32)
    out = enc.encode(np.array([[16 / 31, 0.0]]))          # on cell 16's centre
    assert out.shape == (1, 64)
    x_cells, y_cells = out[0, :32], out[0, 32:]
    assert x_cells.argmax() == 16 and x_cells.max() == pytest.approx(1.0)
    assert x_cells[15] == pytest.approx(x_cells[17]) and 0.5 < x_cells[15] < 0.7
    assert (x_cells > 0.05).sum() <= 5                    # sparse: a handful of x cells
    assert y_cells[0] == pytest.approx(1.0) and y_cells[10] < 1e-6


def test_place_cells_give_nothing_for_an_unseen_position():
    out = PlaceCells(n_per_axis=8).encode(np.array([[np.nan, 0.3], [0.3, 0.3]]))
    assert torch.all(out[0] == 0) and out[1].sum() > 0


def test_place_cells_can_cover_a_signed_range():
    enc = PlaceCells(n_per_axis=5, lo=-0.1, hi=0.1)
    out = enc.bumps(np.array([[0.0, -0.05]]))
    assert out[0, 0].argmax() == 2 and out[0, 1].argmax() == 1
    assert torch.allclose(enc.decode(torch.log(out + 1e-9)), torch.tensor([[0.0, -0.05]]), atol=0.01)


# --- path-parameter targets -------------------------------------------------------

def test_path_targets_reconstruct_the_path_at_any_horizon():
    tr = a_track(shape="figure8")
    p = path_targets(tr.t, tr.gt, tr.period_s)
    assert p.shape == (len(tr.t), N_PATH)
    assert np.allclose(p[:, 0], tr.period_s)
    assert np.allclose(np.hypot(p[:, 1], p[:, 2]), 1.0)  # phase as (cos, sin)
    now = path_position(p, horizon_s=0.0)
    ahead = path_position(p, horizon_s=0.1)
    assert np.abs(now - tr.gt).max() < 2e-3
    inside = tr.t + 0.1 < tr.t[-1]
    assert np.abs(ahead[inside] - sample_at(tr, 0.1)[inside]).max() < 2e-3


def sample_at(tr, h):
    spec = TrajectorySpec(shape="figure8", size=(0.2, 0.12), period_s=tr.period_s,
                          center=(0.5, 0.5), rotation=0.4)
    return sample(spec, tr.t + h)


def test_path_targets_use_the_same_phase_convention_wherever_the_clip_starts():
    """Phase 0 is a point on the path, not the clip's first sample: two clips of the same
    path that start at different moments get the same shape numbers."""
    a = a_track(period_s=1.5)
    b = Track(name="b", t=a.t, obs=a.obs, gt=sample(
        TrajectorySpec(shape="ellipse", size=(0.2, 0.12), period_s=1.5, center=(0.5, 0.5),
                       rotation=0.4, phase0=2.0), a.t), deviation_t=np.nan, period_s=1.5)
    pa, pb = path_targets(a.t, a.gt, 1.5), path_targets(b.t, b.gt, 1.5)
    assert np.allclose(pa[0, 3:], pb[0, 3:], atol=1e-3)  # shape identical
    assert not np.allclose(pa[0, 1:3], pb[0, 1:3])       # phase differs


def test_path_targets_fit_the_part_before_a_break_only():
    tr = a_track(deviations=[Deviation(at_t=3.0, kind="shrink", params={"factor": 0.3})])
    p = path_targets(tr.t, tr.gt, tr.period_s, until_s=tr.deviation_t)
    before = tr.t < 3.0
    assert np.abs(path_position(p, 0.0)[before] - tr.gt[before]).max() < 2e-3


# --- the network -----------------------------------------------------------------

def test_place_cells_decode_is_the_centre_of_mass_of_the_cells():
    enc = PlaceCells(n_per_axis=8)
    logits = torch.log(enc.bumps(np.array([[0.3, 0.6]])) + 1e-9)     # a bump on the position
    assert torch.allclose(enc.decode(logits), torch.tensor([[0.3, 0.6]]), atol=0.01)


def test_adaptive_neuron_raises_its_threshold_after_spiking_and_forgets_slowly():
    n = AdaptiveLeaky(beta=torch.full((1,), 0.8), rho=torch.full((1,), 0.99), scale=1.8)
    state = (torch.zeros(1, 1), torch.zeros(1, 1))
    spk, state = n(torch.full((1, 1), 2.0), state)
    assert spk.item() == 1.0 and state[1].item() == 1.0
    spk2, state = n(torch.full((1, 1), 1.5), state)   # over the old threshold, under the raised one
    assert spk2.item() == 0.0 and 0.98 < state[1].item() < 0.995
    for _ in range(600):                                  # ~3 s at 5 ms: the reluctance fades
        _, state = n(torch.zeros(1, 1), state)
    assert state[1].item() < 0.01


@pytest.mark.parametrize("slow_kind", ["adaptive", "leaky"])
def test_two_layer_net_runs_a_sequence_and_carries_state(slow_kind):
    net = TwoLayerNet(n_in=16, n_fast=24, n_slow=8, dt_s=DT, n_horizons=4, n_cells=8, slow_kind=slow_kind)
    x = torch.rand(30, 2, 16)
    heads, path, state = net(x)
    assert heads.shape == (30, 2, 4, 2, 8) and path.shape == (30, 2, N_PATH)
    assert net.rates[0].requires_grad                     # the rate regulariser can act on it
    heads2, _, _ = net(x[:1], state)
    heads3, _, _ = net(x[:1])
    assert not torch.allclose(heads2, heads3)             # the carried state matters
    assert net.fast.beta.requires_grad and net.slow.beta.requires_grad
    if slow_kind == "adaptive":
        assert net.slow.rho.min() > 0.99                  # adaptation decays over seconds
    else:
        assert net.fast.beta.mean() < net.slow.beta.mean()   # slow layer leaks less


# --- the memory, end to end ---------------------------------------------------------

def tiny_memory(**kw):
    return SpikingMemory(dt_s=DT, n_per_axis=12, n_fast=48, n_slow=24, device="cpu", **kw)


def test_spiking_memory_satisfies_the_protocol_and_predicts_nothing_before_observing():
    m = tiny_memory()
    assert isinstance(m, TrajectoryMemory)
    m.reset()
    assert np.isnan(m.predict(0.1)).all() and m.deviation_score() == 0.0
    m.observe(np.nan, np.nan)                             # unseen: no reference yet
    assert np.isnan(m.predict(0.1)).all()
    with pytest.raises(ValueError):
        m.observe(0.5, 0.5)
        m.predict(0.3)                                    # not one of the trained horizons


def test_fit_lowers_the_error_and_the_memory_then_predicts_ahead():
    torch.manual_seed(0)
    tracks = [a_track(name=f"c{i}", period_s=1.0 + 0.1 * i, center=(0.4 + 0.05 * i, 0.5))
              for i in range(6)]
    m = tiny_memory()
    log = m.fit(tracks, epochs=20, chunk_s=2.0, batch=6, val_fraction=1 / 6, lr=1e-2, seed=0)
    assert log[-1]["val_px"] < log[0]["val_px"] * 0.5
    tr = tracks[0]
    m.reset()
    err = []
    for i, (x, y) in enumerate(tr.obs):
        m.observe(x, y)
        if tr.t[i] > 2.0 and i + 20 < len(tr.t):
            err.append(np.hypot(*(np.array(m.predict(0.1)) - tr.gt[i + 20])))
    assert np.median(err) * 640 < 25                      # px on a 640-wide image


def test_deviation_score_rises_after_a_break(tmp_path):
    torch.manual_seed(0)
    tracks = [a_track(name=f"c{i}", period_s=1.2, center=(0.45 + 0.02 * i, 0.5), duration_s=4.0)
              for i in range(5)]
    m = tiny_memory()
    m.fit(tracks, epochs=24, chunk_s=2.0, batch=5, val_fraction=0.2, lr=1e-2, seed=0)
    m.save(tmp_path / "m.pt")
    m2 = SpikingMemory.load(tmp_path / "m.pt", device="cpu")
    broken = a_track(period_s=1.2, deviations=[Deviation(at_t=4.0, kind="shrink", params={"factor": 0.2})])
    m2.reset()
    score = []
    for x, y in broken.obs:
        m2.observe(x, y)
        score.append(m2.deviation_score())
    score = np.array(score)
    before, after = score[(broken.t > 2.5) & (broken.t < 4.0)], score[broken.t > 4.3]
    assert np.median(after) > 2 * np.median(before)


def test_checkpoint_round_trip_keeps_the_predictions(tmp_path):
    m = tiny_memory()
    m.save(tmp_path / "m.pt")
    m2 = SpikingMemory.load(tmp_path / "m.pt", device="cpu")
    for mem in (m, m2):
        mem.reset()
        for _ in range(10):
            mem.observe(0.5, 0.5)
    assert np.allclose(m.predict(0.1), m2.predict(0.1))
    assert m2.n_per_axis == 12 and m2.horizons_s == m.horizons_s


def test_checkpoints_saved_before_slow_kind_load_as_the_plain_lif_layer(tmp_path):
    m = SpikingMemory(dt_s=DT, n_per_axis=4, n_fast=8, n_slow=4, slow_kind="leaky")
    m.save(tmp_path / "old.pt")
    ck = torch.load(tmp_path / "old.pt", weights_only=False)
    for k in ("slow_kind", "tau_adapt", "adapt_scale"):
        ck["config"].pop(k)
    torch.save(ck, tmp_path / "old.pt")
    assert SpikingMemory.load(tmp_path / "old.pt").slow_kind == "leaky"


def test_anchor_smoothing_damps_centroid_jitter_in_the_prediction():
    from trajmem.snn import _anchor

    obs = np.tile([[0.5, 0.5]], (40, 1)) + np.random.default_rng(0).normal(0, 0.01, (40, 2))
    obs[10] = np.nan
    raw, smooth = _anchor(obs, 1.0), _anchor(obs, 0.25)
    assert np.allclose(raw[10], raw[9]) and np.allclose(smooth[10], smooth[9])   # unseen: hold
    assert np.nanstd(np.diff(smooth[20:], axis=0)) < 0.5 * np.nanstd(np.diff(raw[20:], axis=0))
    m = SpikingMemory(dt_s=DT, n_per_axis=4, n_fast=8, n_slow=4, anchor_tau_s=0.02)
    m.reset()
    for x, y in obs:
        m.observe(x, y)
    assert np.allclose(m.ref, smooth[-1], atol=1e-6)


def test_an_unseen_first_window_does_not_poison_the_deviation_score():
    m = tiny_memory()
    m.reset()
    m.observe(np.nan, np.nan)
    for _ in range(40):
        m.observe(0.5, 0.5)
    assert np.isfinite(m.deviation_score()) and np.isfinite(m.err_imm)


def test_place_cells_accept_tensors_on_any_device():
    enc = PlaceCells(n_per_axis=4)
    xy = torch.tensor([[0.5, 0.5]])
    assert torch.allclose(enc.bumps(xy), enc.bumps(xy.numpy()))


def test_path_points_come_from_the_path_head():
    m = tiny_memory()
    m.reset()
    assert m.path_points([0.0]) is None and np.isnan(m.period())
    m.observe(0.5, 0.5)
    cycle = m.path_points(np.arange(8) / 8)
    assert cycle is None or cycle.shape == (8, 2)          # None while the period readout is not positive
    p = path_targets(np.arange(0, 2, DT), a_track().gt[:400], 1.0)[0]
    m.last_path = p                                       # a known path, as the head would report it
    assert m.period() == 1.0
    cycle = m.path_points(np.arange(64) / 64)
    expected = path_position(np.repeat(p[None], 64, axis=0), np.linspace(0, 1.0, 64, endpoint=False))
    assert np.allclose(cycle, expected)


def test_shape_points_draw_the_path_head_at_even_phases():
    from trajmem.snn import _shape_points

    tr = a_track(shape="figure8")
    p = path_targets(tr.t, tr.gt, tr.period_s)[:1]                # one row of exact parameters
    pts = _shape_points(torch.tensor(p), 8).numpy()[0]           # (8, 2)
    phi0 = np.arctan2(p[0, 2], p[0, 1])
    theta = np.arange(8) * 2 * np.pi / 8
    expected = path_position(np.repeat(p, 8, axis=0), (theta - phi0) * tr.period_s / (2 * np.pi))
    assert pts.shape == (8, 2) and np.allclose(pts, expected, atol=1e-6)


def test_geometric_path_loss_is_zero_on_the_targets_and_logs_path_px():
    torch.manual_seed(0)
    tracks = [a_track(name=f"c{i}", period_s=1.0 + 0.1 * i) for i in range(3)]
    m = tiny_memory()
    log = m.fit(tracks, epochs=1, chunk_s=2.0, batch=3, val_fraction=1 / 3, lr=1e-2, path_loss="geometric")
    assert np.isfinite(log[-1]["val_path_px"]) and log[-1]["val_path_px"] > 0
    arrays = m._arrays(tracks)
    m._standardise(arrays, fit=False)
    yp, mp = arrays[0]["yp"][None], arrays[0]["mp"][None]
    cells = torch.zeros(1, 1, len(m.horizons_s), 2, m.n_per_axis)
    yh, mh = torch.zeros(1, 1, len(m.horizons_s), 2), torch.zeros(1, 1, len(m.horizons_s), dtype=torch.bool)
    _, _, lp, px = m._loss(cells, yp, yh, mh, yp, mp, path_weight=1.0)
    assert lp == pytest.approx(0.0, abs=1e-4) and px == pytest.approx(0.0, abs=1e-2)


def test_split_routing_reads_short_horizons_from_the_fast_layer_and_long_from_the_slow():
    torch.manual_seed(0)
    net = TwoLayerNet(n_in=10, n_fast=16, n_slow=8, heads_from="split", n_horizons=4, n_cells=6, n_short=2)
    assert net.read_h.out_features == 2 * 2 * 6 and net.read_hs.out_features == 2 * 2 * 6
    heads, path, _ = net(torch.rand(5, 3, 10))
    assert heads.shape == (5, 3, 4, 2, 6) and path.shape == (5, 3, N_PATH)
    m = tiny_memory(heads_from="split")
    assert m.net.n_short == 2                              # 25 and 50 ms
    with pytest.raises(ValueError):
        TwoLayerNet(n_in=10, n_fast=16, n_slow=8, heads_from="sideways")
