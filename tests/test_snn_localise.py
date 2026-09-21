import _thesis_path  # noqa: F401  (side effect: adds the thesis repo to sys.path)
import numpy as np
import pytest
import torch

from tests.test_experiment import a_clip_of_events
from trajmem.localise import FrameSet, evaluate_localiser, frame_set, score_track
from trajmem.model import Localiser
from trajmem.snn_localise import SpikingLocaliser, SpikingLocaliserNet, augment_frames
from trajmem.trajectories import TrajectorySpec


def a_frame_set(seed=0, period_s=1.0, center=(0.5, 0.5), duration_s=3.0):
    spec = TrajectorySpec(shape="circle", size=(0.2, 0.2), period_s=period_s, center=center)
    return frame_set(a_clip_of_events(spec, duration_s=duration_s, seed=seed), 5000, 8, name=f"s{seed}")


def tiny():
    return SpikingLocaliser(n_cells=12, ch=(4, 8), hidden=32, device="cpu")


def test_augmentation_adds_stuck_pixels_and_noise_and_flips_the_truth():
    fs = a_frame_set()
    rng = np.random.default_rng(0)
    f, g, p = augment_frames(fs.frames, fs.gt, fs.present, rng, stuck=(100, 100), rate=(1000, 1000),
                             background=(0.0, 0.0), flips=False)
    assert f.shape == fs.frames.shape and f.dtype == np.float32 and np.array_equal(g, fs.gt) and np.array_equal(p, fs.present)
    added = f - fs.frames
    assert added.min() >= 0 and (added.sum(axis=(0, 1)) > 0).sum() <= 100         # at most 100 stuck cells
    assert added.sum() == pytest.approx(100 * 1000 * 0.005 * len(fs.t), rel=0.15)   # Poisson at rate x dt per frame
    f2, g2, _ = augment_frames(fs.frames, fs.gt, fs.present, np.random.default_rng(1), stuck=(0, 0),
                               background=(0.3, 0.3), flips=False)
    assert (f2 - fs.frames).mean() == pytest.approx(0.3, rel=0.1)
    flipped = False
    for seed in range(20):                                                         # some seed flips horizontally
        f3, g3, _ = augment_frames(fs.frames, fs.gt, fs.present, np.random.default_rng(seed), stuck=(0, 0), background=(0, 0))
        if not np.array_equal(g3[:, 0], fs.gt[:, 0]):
            assert np.allclose(g3[:, 0], 1 - fs.gt[:, 0])
            flipped = True
    assert flipped


def test_net_steps_and_carries_state():
    net = SpikingLocaliserNet(n_cells=12, ch=(4, 8), hidden=32)
    state = net.init_state(3, "cpu")
    cells, present, state, rate = net.step(torch.rand(3, 2, 60, 80) * 5, state)
    assert cells.shape == (3, 2, 12) and present.shape == (3,) and 0 <= rate.item() <= 1
    _, _, state2, _ = net.step(torch.rand(3, 2, 60, 80) * 5, state)
    assert not torch.equal(state2[0], state[0])                                   # membranes moved


def test_localiser_satisfies_the_protocol_and_learns_to_find_the_blob(tmp_path):
    torch.manual_seed(0)
    centres = [(0.3, 0.4), (0.7, 0.4), (0.3, 0.6), (0.7, 0.6), (0.5, 0.5), (0.45, 0.55)]   # the last, inside, validates
    sets = [a_frame_set(seed=i, center=c) for i, c in enumerate(centres)]
    loc = tiny()
    assert isinstance(loc, Localiser)
    before = score_track(evaluate_localiser(loc, sets[5]), sets[5])
    log = loc.fit(sets[:5], val_sets=sets[5:], epochs=15, batch=5, augment=False)
    after = score_track(evaluate_localiser(loc, sets[5]), sets[5])
    assert np.isfinite(log[-1]["val_px"]) and after["median"] < 40                # untrained: all unseen (NaN)
    empty = FrameSet("e", np.zeros((40, 2, 60, 80), np.uint8), np.arange(40) * 0.005, np.full((40, 2), 0.5), 5000, 8)
    assert np.isnan(evaluate_localiser(loc, empty)[-1]).all()                      # says "not seen" on nothing
    loc.save(tmp_path / "loc.pt")
    back = SpikingLocaliser.load(tmp_path / "loc.pt")
    assert np.allclose(evaluate_localiser(back, sets[5]), evaluate_localiser(loc, sets[5]), equal_nan=True)
