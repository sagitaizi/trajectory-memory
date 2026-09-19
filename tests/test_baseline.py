import numpy as np
import pytest

from trajmem.baseline import HarmonicFit, PeriodicKalman
from trajmem.trajectories import Deviation, TrajectorySpec, sample

DT = 0.005


def a_track(duration_s=10.0, deviations=None, shape="ellipse"):
    spec = TrajectorySpec(shape=shape, size=(0.2, 0.12), period_s=1.3, center=(0.5, 0.5),
                          rotation=0.4, deviations=deviations or [])
    t = np.arange(0, duration_s, DT) + DT / 2
    return t, sample(spec, t), spec


def run(model, t, xy, horizon_s, start_s=0.0):
    """Feed the track; collect (t, prediction for t+h, score) from `start_s` on."""
    out = []
    for ti, (x, y) in zip(t, xy):
        model.observe(x, y)
        if ti >= start_s:
            out.append((ti, *model.predict(horizon_s), model.deviation_score()))
    return np.array(out)


@pytest.mark.parametrize("make", [lambda: PeriodicKalman(dt_s=DT), lambda: HarmonicFit(dt_s=DT)])
def test_predicts_a_clean_repetitive_path_ahead(make):
    t, xy, spec = a_track()
    out = run(make(), t, xy, horizon_s=0.1, start_s=5.0)
    truth = sample(spec, out[:, 0] + 0.1)
    err = np.hypot(*(out[:, 1:3] - truth).T)
    assert np.median(err) < 0.01                    # under 1 % of the frame, ~6 px


@pytest.mark.parametrize("make", [lambda: PeriodicKalman(dt_s=DT), lambda: HarmonicFit(dt_s=DT)])
def test_prediction_error_grows_with_horizon_but_stays_small(make):
    t, xy, spec = a_track()
    m = make()
    errs = {}
    for h in (0.05, 0.2):
        m.reset()
        out = run(m, t, xy, horizon_s=h, start_s=5.0)
        errs[h] = np.median(np.hypot(*(out[:, 1:3] - sample(spec, out[:, 0] + h)).T))
    assert errs[0.05] <= errs[0.2] + 1e-9 and errs[0.2] < 0.03


@pytest.mark.parametrize("make", [lambda: PeriodicKalman(dt_s=DT), lambda: HarmonicFit(dt_s=DT)])
def test_deviation_score_rises_after_a_break(make):
    dev = [Deviation(at_t=7.0, kind="shrink", params={"factor": 0.5})]
    t, xy, _ = a_track(deviations=dev)
    out = run(make(), t, xy, horizon_s=0.1, start_s=4.0)
    before = out[(out[:, 0] > 5.0) & (out[:, 0] < 7.0), 3]
    after = out[(out[:, 0] > 7.2) & (out[:, 0] < 8.0), 3]
    assert np.median(after) > 3 * np.percentile(before, 99)


@pytest.mark.parametrize("make", [lambda: PeriodicKalman(dt_s=DT), lambda: HarmonicFit(dt_s=DT)])
def test_unknown_observations_are_skipped_not_fatal(make):
    t, xy, spec = a_track()
    xy = xy.copy()
    xy[1200:1300] = np.nan                          # half a second unseen, after warm-up
    out = run(make(), t, xy, horizon_s=0.1, start_s=8.0)
    assert np.isfinite(out[:, 1:3]).all()
    err = np.hypot(*(out[:, 1:3] - sample(spec, out[:, 0] + 0.1)).T)
    assert np.median(err) < 0.01


@pytest.mark.parametrize("make", [lambda: PeriodicKalman(dt_s=DT), lambda: HarmonicFit(dt_s=DT)])
def test_before_warmup_it_predicts_the_last_position_and_scores_zero(make):
    m = make()
    m.observe(0.3, 0.6)
    assert m.predict(0.1) == (0.3, 0.6)
    assert m.deviation_score() == 0.0
    assert np.isnan(make().predict(0.1)).all()


@pytest.mark.parametrize("make", [lambda: PeriodicKalman(dt_s=DT), lambda: HarmonicFit(dt_s=DT)])
def test_reset_forgets_everything(make):
    t, xy, _ = a_track(duration_s=6.0)
    m = make()
    run(m, t, xy, horizon_s=0.1)
    m.reset()
    assert np.isnan(m.predict(0.1)).all()
    assert m.period_s is None


@pytest.mark.parametrize("make", [lambda: PeriodicKalman(dt_s=DT), lambda: HarmonicFit(dt_s=DT)])
def test_predicts_a_slow_path_whose_period_exceeds_the_warmup(make):
    spec = TrajectorySpec(shape="ellipse", size=(0.2, 0.12), period_s=4.0, rotation=0.4)
    t = np.arange(0, 16.0, DT) + DT / 2
    out = run(make(), t, sample(spec, t), horizon_s=0.1, start_s=8.0)
    err = np.hypot(*(out[:, 1:3] - sample(spec, out[:, 0] + 0.1)).T)
    assert np.median(err) < 0.01


@pytest.mark.parametrize("make", [lambda: PeriodicKalman(dt_s=DT), lambda: HarmonicFit(dt_s=DT)])
def test_predicts_a_figure8_whose_energy_sits_at_the_second_harmonic(make):
    spec = TrajectorySpec(shape="figure8", size=(0.1, 0.3), period_s=2.0, rotation=0.3)
    t = np.arange(0, 12.0, DT) + DT / 2
    out = run(make(), t, sample(spec, t), horizon_s=0.1, start_s=7.0)
    err = np.hypot(*(out[:, 1:3] - sample(spec, out[:, 0] + 0.1)).T)
    assert np.median(err) < 0.01


def test_surprise_smoothing_is_stable_for_windows_longer_than_its_time_constant():
    m = PeriodicKalman(dt_s=0.2, warmup_s=5.0)
    assert m.score_alpha <= 1.0


@pytest.mark.parametrize("make", [lambda: PeriodicKalman(dt_s=DT), lambda: HarmonicFit(dt_s=DT)])
def test_path_points_trace_the_learned_path_once_the_period_is_known(make):
    t, xy, spec = a_track()
    model = make()
    assert model.path_points([0.0]) is None and np.isnan(model.period())   # nothing learned yet
    run(model, t, xy, 0.1)
    assert model.period() == pytest.approx(spec.period_s, rel=0.02)
    cycle = model.path_points(np.arange(64) / 64)
    assert cycle.shape == (64, 2)
    truth = sample(spec, np.linspace(0, spec.period_s, 64, endpoint=False))
    d = np.hypot(*(cycle[:, None] - truth[None]).T)        # every point sits on the true path
    assert d.min(axis=0).max() < 0.005
