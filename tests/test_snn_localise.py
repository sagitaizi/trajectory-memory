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
    clip = a_clip_of_events(spec, duration_s=duration_s, per_window=300, seed=seed)     # sim-like brightness
    return frame_set(clip, 5000, 8, name=f"s{seed}")


def tiny():
    return SpikingLocaliser(n_cells=12, ch=(4, 8), hidden=32, device="cpu")


def test_augmentation_adds_stuck_pixels_and_noise_and_flips_the_truth():
    fs = a_frame_set()
    rng = np.random.default_rng(0)
    f, g, p = augment_frames(fs.frames, fs.gt, fs.present, rng, stuck=(100, 100), rate=(1000, 1000),
                             background=(0.0, 0.0), flips=False, blanks=(0, 0), brightness=None, polarity_swap=False, rotate=0)
    assert f.shape == fs.frames.shape and f.dtype == np.float32 and np.array_equal(g, fs.gt) and np.array_equal(p, fs.present)
    added = f - fs.frames
    assert added.min() >= 0 and (added.sum(axis=(0, 1)) > 0).sum() <= 100         # at most 100 stuck cells
    assert added.sum() == pytest.approx(100 * 1000 * 0.005 * len(fs.t), rel=0.15)   # Poisson at rate x dt per frame
    f2, g2, _ = augment_frames(fs.frames, fs.gt, fs.present, np.random.default_rng(1), stuck=(0, 0),
                               background=(0.3, 0.3), flips=False, blanks=(0, 0), brightness=None, polarity_swap=False, rotate=0)
    assert (f2 - fs.frames).mean() == pytest.approx(0.3, rel=0.1)
    flipped = False
    for seed in range(20):                                                         # some seed flips horizontally
        f3, g3, _ = augment_frames(fs.frames, fs.gt, fs.present, np.random.default_rng(seed), stuck=(0, 0), background=(0, 0), blanks=(0, 0), brightness=None, polarity_swap=False, rotate=0)
        if not np.array_equal(g3[:, 0], fs.gt[:, 0]):
            assert np.allclose(g3[:, 0], 1 - fs.gt[:, 0])
            flipped = True
    assert flipped
    f4, _, p4 = augment_frames(fs.frames, fs.gt, fs.present, np.random.default_rng(2), stuck=(0, 0), background=(0, 0),
                               flips=False, blanks=(2, 2), blank_s=(0.2, 0.2), brightness=None, polarity_swap=False, rotate=0)
    assert 0 < (~p4).sum() <= 80 and f4[~p4].sum() < fs.frames[~p4].sum() * 0.1     # the target is gone there
    f5, _, _ = augment_frames(fs.frames, fs.gt, fs.present, np.random.default_rng(3), stuck=(0, 0), background=(0, 0),
                              flips=False, blanks=(0, 0), brightness=(0.5, 0.5), polarity_swap=False, rotate=0)
    assert np.allclose(f5, fs.frames * 0.5)


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
    log = loc.fit(sets[:5], val_sets=sets[5:], epochs=15, batch=5, augment_kw={"stuck": (20, 60), "brightness": None, "polarity_swap": False, "rotate": 0})
    after = score_track(evaluate_localiser(loc, sets[5]), sets[5])
    assert np.isfinite(log[-1]["val_px"]) and after["median"] < 60
    empty = FrameSet("e", np.zeros((40, 2, 60, 80), np.uint8), np.arange(40) * 0.005, np.full((40, 2), 0.5), 5000, 8)
    assert np.isnan(evaluate_localiser(loc, empty)[-1]).all()                      # says "not seen" on nothing
    loc.save(tmp_path / "loc.pt")
    back = SpikingLocaliser.load(tmp_path / "loc.pt")
    assert np.allclose(evaluate_localiser(back, sets[5]), evaluate_localiser(loc, sets[5]), equal_nan=True)


def test_checkpoint_keeps_the_validation_clip_names(tmp_path):
    loc = SpikingLocaliser(n_cells=4, ch=(2, 3), hidden=8, in_hw=(12, 16))
    loc.val_clips = ["sim_003", "sim_017"]
    loc.save(tmp_path / "l.pt")
    assert SpikingLocaliser.load(tmp_path / "l.pt").val_clips == ["sim_003", "sim_017"]


def test_summed_polarity_is_blind_to_an_on_off_swap(tmp_path):
    net = SpikingLocaliserNet(n_cells=6, ch=(3, 4), hidden=16, polarity="sum")
    frames = torch.rand(2, 2, 60, 80) * 5
    state = net.init_state(2, "cpu")
    a = net.step(frames, state)[0]
    b = net.step(frames.flip(1), state)[0]
    assert torch.allclose(a, b)
    loc = SpikingLocaliser(n_cells=6, ch=(3, 4), hidden=16, polarity="sum")
    loc.save(tmp_path / "s.pt")
    assert SpikingLocaliser.load(tmp_path / "s.pt").net.polarity == "sum"
    with pytest.raises(ValueError):
        SpikingLocaliserNet(polarity="mean")


def test_rotate_frames_turns_the_target_with_the_frame():
    from trajmem.snn_localise import rotate_frames

    frames = np.zeros((2, 2, 60, 80), np.float32)
    frames[0, 0, 30, 40] = 5.0                                    # centre-ish: stays in frame
    frames[1, 1, 30, 75] = 3.0                                    # far right: turned onto rows that are cropped away
    gt = np.array([[40 / 80, 30 / 60], [75 / 80, 30 / 60]])
    present = np.array([True, True])
    turned, g2, p2 = rotate_frames(frames, gt, present)
    assert turned.shape == frames.shape and turned.sum() == 5.0
    y, x = np.argwhere(turned[0, 0] > 0)[0]
    assert (round(g2[0, 0] * 80), round(g2[0, 1] * 60)) == (x, y)
    assert p2.tolist() == [True, False]


def test_fit_without_calibration_keeps_the_starting_weights_for_fine_tuning():
    sets = [a_frame_set(seed=i, duration_s=1.0) for i in range(2)]

    def weights_after(calibrate):
        loc = tiny()
        start = {k: v.clone() for k, v in loc.net.state_dict().items()}
        loc.fit(sets, epochs=1, batch=2, lr=0.0, augment=False, calibrate=calibrate)
        return all(torch.equal(start[k], v) for k, v in loc.net.state_dict().items())

    assert weights_after(calibrate=False)
    assert not weights_after(calibrate=True)
