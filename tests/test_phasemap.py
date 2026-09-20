import _thesis_path  # noqa: F401  (side effect: adds the thesis repo to sys.path)
import numpy as np
import pytest

from trajmem.model import TrajectoryMemory
from trajmem.phasemap import PhaseMap
from trajmem.snn_phasemap import SpikingPhaseMap
from trajmem.trajectories import Deviation, TrajectorySpec, sample

DT = 0.005


def a_path(period_s=1.3, duration_s=10.0, deviations=None):
    spec = TrajectorySpec(shape="ellipse", size=(0.2, 0.12), period_s=period_s, center=(0.5, 0.5),
                          rotation=0.4, deviations=deviations or [])
    t = np.arange(0, duration_s, DT) + DT / 2
    return spec, t, sample(spec, t)


@pytest.mark.parametrize("make", [lambda: PhaseMap(dt_s=DT),
                                  lambda: SpikingPhaseMap(dt_s=DT, n_clock=2000, n_ring=100)])
def test_clock_and_map_satisfies_the_protocol_and_locks_onto_a_clean_path(make):
    m = make()
    assert isinstance(m, TrajectoryMemory)
    m.reset()
    assert np.isnan(m.period()) and m.path_points([0.0]) is None
    spec, t, xy = a_path()
    errs = []
    for i, (x, y) in enumerate(xy):
        m.observe(x, y)
        if t[i] > 7.0 and i + 20 < len(t):
            errs.append(np.hypot(*(np.array(m.predict(0.1)) - xy[i + 20])) * 640)
    assert m.period() == pytest.approx(1.3, rel=0.1)
    assert np.median(errs) < 30
    pts = m.path_points(np.arange(16) / 16)
    assert pts.shape == (16, 2) and np.isfinite(pts).all()
    assert np.hypot(*(pts.mean(0) - 0.5)) < 0.05                     # the map is centred on the path


def test_prototype_keeps_predicting_when_the_input_goes_blank():
    m = PhaseMap(dt_s=DT)
    spec, t, xy = a_path(duration_s=12.0)
    for x, y in xy:
        m.observe(x, y)
    errs = []
    for k in range(1, 121):                                          # 0.6 s blank
        m.observe(np.nan, np.nan)
        truth = sample(spec, t[-1] + k * DT + 0.1)
        errs.append(np.hypot(*(np.array(m.predict(0.1)) - truth)) * 640)
    assert np.median(errs) < 30


def test_deviation_score_rises_after_a_break_against_the_snapshot():
    m = PhaseMap(dt_s=DT)
    spec, t, xy = a_path(duration_s=20.0, deviations=[Deviation(at_t=15.0, kind="shrink", params={"factor": 0.5})])
    scores = []
    for x, y in xy:
        m.observe(x, y)
        scores.append(m.deviation_score())
    scores = np.array(scores)
    before, after = scores[(t > 12) & (t < 15)], scores[t > 15.5]
    assert np.median(after) > 3 * np.median(before)
