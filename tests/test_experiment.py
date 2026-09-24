import _thesis_path  # noqa: F401  (side effect: adds the thesis repo to sys.path)
import numpy as np
import pytest

from trajmem.baseline import HarmonicFit
from trajmem.data import Clip
from trajmem.experiment import Trace, evaluate_clip, make_memory, score_trace
from trajmem.simulate import EVENT_DTYPE
from trajmem.trajectories import Deviation, TrajectorySpec, sample

RES = (640, 480)


def a_clip_of_events(spec, duration_s=8.0, per_window=30, window_us=5000, seed=0):
    """Events clustered on the analytic path, one clump per window, no simulator."""
    rng = np.random.default_rng(seed)
    n_win = int(duration_s * 1e6 / window_us)
    t_win = (np.arange(n_win) + 0.5) * window_us / 1e6
    centres = sample(spec, t_win) * RES
    xs = np.repeat(centres[:, 0], per_window) + rng.normal(0, 2, n_win * per_window)
    ys = np.repeat(centres[:, 1], per_window) + rng.normal(0, 2, n_win * per_window)
    ts = np.repeat(np.arange(n_win) * window_us, per_window) + rng.integers(0, window_us, n_win * per_window)
    ev = np.zeros(len(xs), dtype=EVENT_DTYPE)
    ev["x"], ev["y"], ev["timestamp"], ev["polarity"] = np.clip(xs, 0, 639), np.clip(ys, 0, 479), ts, 1
    ev = ev[np.argsort(ev["timestamp"], kind="stable")]
    return Clip(events=ev, duration_us=int(duration_s * 1e6), gt=lambda t: sample(spec, t),
                deviation_times=[d.at_t for d in spec.deviations], meta={"resolution": RES})


def a_spec(deviations=None):
    return TrajectorySpec(shape="circle", size=(0.2, 0.2), period_s=1.0, deviations=deviations or [])


def test_evaluate_clip_walks_the_clip_and_lines_up_truth_at_the_horizon():
    clip = a_clip_of_events(a_spec())
    trace = evaluate_clip(HarmonicFit(dt_s=0.005, warmup_s=3.0), clip, window_us=5000, horizon_s=0.1)
    assert isinstance(trace, Trace)
    assert len(trace.t) == 1600 and trace.pred.shape == (1600, 2) == trace.gt_ahead.shape
    assert trace.t[0] == pytest.approx(0.0025) and trace.horizon_s == 0.1
    assert np.allclose(trace.gt_ahead[:10], sample(a_spec(), trace.t[:10] + 0.1))
    assert np.isnan(trace.gt_ahead[-1]).all()          # t + h runs past the clip's end
    late = trace.t > 5.0
    err = np.hypot(*(trace.pred[late] - trace.gt_ahead[late]).T)
    assert np.nanmedian(err) < 0.01


def test_score_trace_reports_the_three_metrics_in_pixels():
    spec = a_spec(deviations=[Deviation(at_t=6.0, kind="shrink", params={"factor": 0.5})])
    clip = a_clip_of_events(spec)
    trace = evaluate_clip(HarmonicFit(dt_s=0.005, warmup_s=3.0), clip, window_us=5000, horizon_s=0.1)
    r = score_trace(trace, clip, tol_px=10.0, settle_s=4.0)
    assert r["fde_px"]["median"] < 6.0                 # pre-break steps only
    assert 0 < r["lock_on_s"] < 4.0
    assert r["deviation"]["auc"] > 0.95 and r["deviation"]["latency_s"] < 0.5


def test_score_trace_without_a_break_has_no_detection_numbers_but_a_false_alarm_rate():
    clip = a_clip_of_events(a_spec())
    trace = evaluate_clip(HarmonicFit(dt_s=0.005, warmup_s=3.0), clip, window_us=5000, horizon_s=0.1)
    r = score_trace(trace, clip, tol_px=10.0, settle_s=4.0, threshold=1e9)
    assert np.isnan(r["deviation"]["auc"]) and r["deviation"]["fp_per_min"] == 0.0


def test_make_memory_builds_the_named_baseline():
    assert isinstance(make_memory("harmonic", dt_s=0.005), HarmonicFit)
    assert isinstance(make_memory("harmonic", dt_s=0.005, checkpoint="x.pt"), HarmonicFit)   # SNN-only param ignored
    with pytest.raises(ValueError):
        make_memory("nonsense", dt_s=0.005)


def test_make_memory_loads_the_snn_from_a_checkpoint(tmp_path):
    from trajmem.snn import SpikingMemory

    SpikingMemory(dt_s=0.005, n_per_axis=4, n_fast=8, n_slow=4).save(tmp_path / "m.pt")
    m = make_memory("snn", dt_s=0.005, warmup_s=5.0, checkpoint=tmp_path / "m.pt")
    assert isinstance(m, SpikingMemory) and m.n_fast == 8
    with pytest.raises(ValueError):
        make_memory("snn", dt_s=0.001, checkpoint=tmp_path / "m.pt")


# --- clip sets and pooled runs ---------------------------------------------------

def test_load_set_reads_entries_with_optional_slices(tmp_path):
    from trajmem.experiment import load_set

    sets = tmp_path / "sets.yaml"
    sets.write_text("development:\n"
                    "  - {clip: a/b/c}\n"
                    "  - {clip: a/b/c, slice: [6.0, null], name: c/steady}\n"
                    "held_out: []\n")
    entries = load_set("development", sets)
    assert entries == [{"clip": "a/b/c", "name": "c", "slice": None},
                       {"clip": "a/b/c", "name": "c/steady", "slice": (6.0, None)}]
    assert load_set("held_out", sets) == []


def test_run_set_scores_each_clip_and_pools_the_medians():
    from trajmem.experiment import run_set

    spec = a_spec(deviations=[Deviation(at_t=6.0, kind="shrink", params={"factor": 0.5})])
    clips = [("one", a_clip_of_events(a_spec())), ("two", a_clip_of_events(spec, seed=1))]
    rows, pooled = run_set(lambda: HarmonicFit(dt_s=0.005, warmup_s=3.0), clips,
                           window_us=5000, horizon_s=0.1, tol_px=10.0, settle_s=4.0)
    assert [r["name"] for r in rows] == ["one", "two"]
    assert rows[0]["fde_px"]["median"] < 6.0 and rows[1]["deviation"]["auc"] > 0.9
    assert pooled["name"] == "pooled" and pooled["n_clips"] == 2
    assert pooled["fde_px"]["median"] == pytest.approx(
        np.median([rows[0]["fde_px"]["median"], rows[1]["fde_px"]["median"]]))
    assert pooled["deviation"]["auc"] == rows[1]["deviation"]["auc"]   # the only break clip


def test_load_set_expands_a_glob_entry(tmp_path):
    from trajmem.experiment import load_set

    for name in ("sim_001", "sim_000"):
        (tmp_path / f"{name}.npz").write_bytes(b"")
    sets = tmp_path / "sets.yaml"
    sets.write_text(f"sim:\n  - {{glob: '{tmp_path.name}/sim_*.npz'}}\n")   # root = its grandparent
    entries = load_set("sim", sets)
    assert [e["name"] for e in entries] == ["sim_000", "sim_001"]
    assert entries[0]["clip"].endswith("sim_000.npz") and entries[0]["slice"] is None


def test_pool_keeps_misses_and_counts_them():
    from trajmem.experiment import pool

    def row(name, latency, lock):
        return {"name": name, "fde_px": {"median": 1.0, "iqr": 0.1}, "lock_on_s": lock,
                "path_px": {"median": 2.0, "last": 1.5}, "path_lock_on_s": lock, "period_ratio": 1.0, "blank_px": np.nan,
                "deviation": {"auc": 0.9, "latency_s": latency, "fp_per_min": 0.0}, "unseen_fraction": 0.0}
    pooled = pool([row("a", 0.3, 1.0), row("b", np.inf, np.inf), row("c", 0.4, 2.0), row("d", np.inf, 3.0)])
    assert pooled["deviation"]["latency_s"] == np.inf            # half the breaks were missed
    assert pooled["deviation"]["missed"] == 2 and pooled["never_locked"] == 1
    assert pooled["lock_on_s"] == 2.5 == pooled["path_lock_on_s"]   # median of [1, inf, 2, 3]
    assert pooled["path_px"] == {"median": 2.0, "last": 1.5}


def test_load_set_glob_resolves_against_the_repo_and_refuses_an_empty_match(tmp_path, monkeypatch):
    from trajmem.experiment import load_set

    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "sim_000.npz").write_bytes(b"")
    sets = tmp_path / "corpus" / "sets.yaml"
    sets.write_text("sim:\n  - {glob: corpus/sim_*.npz}\nnone:\n  - {glob: corpus/nothing_*.npz}\n")
    monkeypatch.chdir(tmp_path / "corpus")                        # not the repo root
    assert [e["name"] for e in load_set("sim", sets)] == ["sim_000"]
    with pytest.raises(ValueError, match="matches no clip"):
        load_set("none", sets)


def test_subtracting_the_label_offset_removes_a_constant_labelling_bias():
    from trajmem.experiment import label_offset

    spec = a_spec()
    clip = a_clip_of_events(spec)
    shifted = Clip(events=clip.events, duration_us=clip.duration_us, deviation_times=[],
                   gt=lambda t: sample(spec, t) + np.array([0.0, 0.05]), meta=clip.meta)   # labels 24 px low
    trace = evaluate_clip(HarmonicFit(dt_s=0.005, warmup_s=3.0), shifted, window_us=5000, horizon_s=0.1)
    raw = score_trace(trace, shifted, tol_px=10.0, settle_s=4.0)
    fixed = score_trace(trace, shifted, tol_px=10.0, settle_s=4.0, subtract_offset=True)
    assert raw["fde_px"]["median"] > 20 and fixed["fde_px"]["median"] < 6
    assert abs(fixed["offset_px"][1] - 24) < 2 and abs(fixed["offset_px"][0]) < 2
    assert np.allclose(label_offset(trace, shifted, 4.0) * (640, 480), fixed["offset_px"])


# --- path shape ------------------------------------------------------------------

def test_reference_cycle_uses_the_spec_period_or_searches_for_it():
    from trajmem.experiment import clip_period, reference_cycle

    clip = a_clip_of_events(a_spec())
    assert clip_period(clip) == pytest.approx(1.0, abs=0.02)      # searched: the meta has no spec
    clip.meta["spec"] = a_spec()
    assert clip_period(clip) == 1.0
    cycle = reference_cycle(clip, n=64)
    truth = sample(a_spec(), np.linspace(0, 1.0, 64, endpoint=False))
    assert cycle.shape == (64, 2) and np.hypot(*(cycle - truth).T).max() < 0.01


def test_reference_cycle_fits_the_part_before_the_break():
    from trajmem.experiment import reference_cycle

    spec = a_spec(deviations=[Deviation(at_t=6.0, kind="shrink", params={"factor": 0.5})])
    clip = a_clip_of_events(spec)
    clip.meta["spec"] = spec
    cycle = reference_cycle(clip, n=64)
    assert np.hypot(*(cycle - np.array([0.5, 0.5])).T).mean() == pytest.approx(0.2, abs=0.01)


def test_evaluate_clip_scores_the_path_shape_per_step_and_score_trace_summarises_it():
    clip = a_clip_of_events(a_spec())
    trace = evaluate_clip(HarmonicFit(dt_s=0.005, warmup_s=3.0), clip, window_us=5000, horizon_s=0.1)
    assert trace.cycles.shape == (1600, 64, 2) and trace.period.shape == (1600,)
    assert np.isnan(trace.cycles[trace.t < 3.0]).all()            # no path before the warm-up
    assert np.isfinite(trace.cycles[trace.t > 3.1]).all()
    r = score_trace(trace, clip, tol_px=10.0, settle_s=4.0)
    assert r["path_px"]["median"] < 3.0 and r["path_px"]["last"] < 3.0
    assert r["period_ratio"] == pytest.approx(1.0, abs=0.02)
    assert 3.0 <= r["path_lock_on_s"] < 4.0


def test_remembered_path_covers_one_true_period_whatever_the_memory_thinks_the_period_is():
    from trajmem.experiment import remembered_path

    class Doubled:                                        # right curve, period taken as 2T
        def period(self):
            return 2.0

        def path_points(self, fractions):
            phi = 4 * np.pi * np.asarray(fractions)
            return np.column_stack([np.cos(phi), np.sin(phi)])

    path = remembered_path(Doubled(), period_s=1.0, n=64)
    phi = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    assert np.allclose(path, np.column_stack([np.cos(phi), np.sin(phi)]), atol=1e-9)

    class Nothing:
        def period(self):
            return np.nan

        def path_points(self, fractions):
            return None

    assert np.isnan(remembered_path(Nothing(), 1.0, n=8)).all()


def test_make_memory_picks_the_class_from_the_checkpoint(tmp_path):
    from trajmem.snn_lmu import LmuMemory

    LmuMemory(dt_s=0.005, q=4, n_per_dim=10, theta_s=1.0, n_per_axis=4, n_fast=8).save(tmp_path / "l.pt")
    assert isinstance(make_memory("snn", dt_s=0.005, checkpoint=tmp_path / "l.pt"), LmuMemory)


def test_evaluate_clip_can_hide_a_window_and_score_the_error_inside_it():
    clip = a_clip_of_events(a_spec())
    trace = evaluate_clip(HarmonicFit(dt_s=0.005, warmup_s=3.0), clip, window_us=5000, horizon_s=0.1,
                          blank=(5.0, 5.5))
    inside = (trace.t >= 5.0) & (trace.t < 5.5)
    assert np.isnan(trace.obs[inside]).all() and np.isfinite(trace.obs[~inside]).all()
    r = score_trace(trace, clip, tol_px=10.0, settle_s=4.0)
    assert 0 < r["blank_px"] < 10.0                                 # HarmonicFit extrapolates through it
    assert np.isnan(score_trace(evaluate_clip(HarmonicFit(dt_s=0.005, warmup_s=3.0), clip, 5000, 0.1),
                                clip, 10.0, 4.0)["blank_px"])


def test_evaluate_clip_with_a_frame_localiser_matches_the_centroid_path():
    from trajmem.experiment import make_localiser
    from trajmem.localise import FrameCentroid

    clip = a_clip_of_events(a_spec())
    ref = evaluate_clip(HarmonicFit(dt_s=0.005, warmup_s=3.0), clip, window_us=5000, horizon_s=0.1)
    via = evaluate_clip(HarmonicFit(dt_s=0.005, warmup_s=3.0), clip, window_us=5000, horizon_s=0.1,
                        localiser=FrameCentroid())
    assert np.allclose(via.t, ref.t)
    assert np.nanmedian(np.hypot(*(via.obs - ref.obs).T)) * 640 < 2.0        # same positions to a px or two
    assert make_localiser("centroid") is None and isinstance(make_localiser("frame_centroid"), FrameCentroid)
    with pytest.raises(ValueError):
        make_localiser("sideways")


def test_evaluate_clip_serves_several_horizons_from_one_pass():
    clip = a_clip_of_events(a_spec())
    many = evaluate_clip(HarmonicFit(dt_s=0.005, warmup_s=3.0), clip, 5000, (0.05, 0.1))
    assert [tr.horizon_s for tr in many] == [0.05, 0.1]
    for tr in many:
        one = evaluate_clip(HarmonicFit(dt_s=0.005, warmup_s=3.0), clip, 5000, tr.horizon_s)
        assert np.allclose(tr.pred, one.pred, equal_nan=True)
        assert np.allclose(tr.gt_ahead, one.gt_ahead, equal_nan=True)
        assert np.allclose(tr.score, one.score, equal_nan=True)
