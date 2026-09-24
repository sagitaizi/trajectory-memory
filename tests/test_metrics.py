import numpy as np
import pytest

from trajmem.metrics import ade_fde, deviation_roc, displacement_error, lock_on_time


# --- prediction error -----------------------------------------------------------

def test_displacement_error_summarises_the_distance_per_step():
    pred = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 3.0], [np.nan, 0.0]])
    gt = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    r = displacement_error(pred, gt)
    assert np.allclose(r["errors"], [0.0, 1.0, 3.0, np.nan], equal_nan=True)
    assert r["median"] == 1.0 and r["mean"] == pytest.approx(4 / 3) and r["n"] == 3
    assert r["iqr"] == pytest.approx(np.percentile([0, 1, 3], 75) - np.percentile([0, 1, 3], 25))


def test_displacement_error_ignores_steps_with_unknown_ground_truth():
    pred = np.array([[1.0, 0.0], [1.0, 0.0]])
    gt = np.array([[0.0, 0.0], [np.nan, np.nan]])
    assert displacement_error(pred, gt)["n"] == 1


# --- lock-on --------------------------------------------------------------------

def test_lock_on_is_the_first_time_error_stays_under_tolerance():
    errors = np.array([5, 4, 3, 0.5, 2, 0.5, 0.4, 0.3, 0.2, 0.1])
    assert lock_on_time(errors, tol=1.0, dt=0.1) == pytest.approx(0.5)     # step 5, not step 3


def test_lock_on_is_infinite_when_it_never_settles():
    assert lock_on_time(np.array([5, 0.5, 5, 0.5, 5]), tol=1.0, dt=0.1) == np.inf


def test_lock_on_skips_unknown_steps():
    errors = np.array([5, np.nan, 0.5, np.nan, 0.5])
    assert lock_on_time(errors, tol=1.0, dt=0.1) == pytest.approx(0.2)


# --- deviation detection --------------------------------------------------------

def a_score_trace(dt=0.1, t_break=5.0, n=100, rise=3.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    s = rng.normal(0, 1, n) * noise + np.where(t >= t_break, rise, 0.0)
    return s, t


def test_deviation_roc_is_perfect_for_a_clean_step():
    s, t = a_score_trace()
    r = deviation_roc(s, t, [5.0], hold_n=3)
    assert r["auc"] == pytest.approx(1.0)
    assert r["latency_s"] == pytest.approx(0.2)        # three steps held above threshold
    assert r["fp_per_min"] == 0.0


def test_deviation_roc_is_chance_for_pure_noise():
    s, t = a_score_trace(rise=0.0, noise=1.0)
    r = deviation_roc(s, t, [5.0])
    assert 0.35 < r["auc"] < 0.65


def test_deviation_roc_counts_false_alarms_on_a_clip_without_a_break():
    s, t = a_score_trace(rise=0.0, noise=1.0, n=600)     # one minute
    s[100:110] = 10.0                                    # one sustained spurious burst
    r = deviation_roc(s, t, [], threshold=5.0, hold_n=3)
    assert np.isnan(r["auc"]) and np.isnan(r["latency_s"])
    assert r["fp_per_min"] == pytest.approx(1.0)


def test_deviation_roc_latency_is_unbounded_when_never_flagged():
    s, t = a_score_trace(rise=0.0)
    assert deviation_roc(s, t, [5.0], threshold=1.0)["latency_s"] == np.inf


def test_deviation_roc_default_threshold_sits_above_the_pre_break_scores():
    s, t = a_score_trace(noise=0.5, rise=5.0)
    r = deviation_roc(s, t, [5.0])
    assert r["threshold"] >= np.percentile(s[t < 5.0], 99) - 1e-9
    assert r["latency_s"] < 1.0


# --- path shape ------------------------------------------------------------------

def a_cycle(radius, n=64, phase=0.0, centre=(0.0, 0.0)):
    phi = np.linspace(0, 2 * np.pi, n, endpoint=False) + phase
    return np.column_stack([centre[0] + radius * np.cos(phi), centre[1] + radius * np.sin(phi)])


def test_path_shape_error_is_zero_for_the_same_cycle_at_another_phase():
    from trajmem.metrics import path_shape_error

    assert path_shape_error(a_cycle(1.0, phase=1.3), a_cycle(1.0)) < 2e-3   # chord error of the sub-point shifts


def test_path_shape_error_is_the_mean_point_distance_after_the_best_shift():
    from trajmem.metrics import path_shape_error

    assert path_shape_error(a_cycle(1.5), a_cycle(1.0)) == pytest.approx(0.5, abs=2e-3)
    assert path_shape_error(a_cycle(1.0, centre=(0.3, 0.0)), a_cycle(1.0)) == pytest.approx(0.3, abs=0.02)
    reversed_cycle = a_cycle(1.0)[::-1]                   # same shape, run the other way: not the same path
    assert path_shape_error(reversed_cycle, a_cycle(1.0)) > 0.5


def test_path_shape_error_needs_matching_point_counts():
    from trajmem.metrics import path_shape_error

    with pytest.raises(ValueError):
        path_shape_error(a_cycle(1.0, n=32), a_cycle(1.0, n=64))


def test_ratchet_threshold_tightens_but_never_loosens():
    from trajmem.metrics import ratchet_threshold

    t = np.arange(0, 30, 0.005)
    rng = np.random.default_rng(0)
    score = np.where(t < 8, 40.0, 10.0) + rng.normal(0, 1.0, len(t))     # settles after 8 s
    score[t > 20] += 60.0                                                # then a sustained break
    bar = ratchet_threshold(score, t, k=6.0)
    assert np.isinf(bar[t < 5]).all()                                    # no bar during warm-up
    settled, during_break = bar[(t > 9) & (t < 10)].max(), bar[t > 25].min()
    assert settled > 40                                                  # set while the score was still high
    assert during_break <= bar[(t > 15) & (t < 16)].min()                # never rises again
    assert np.all(np.diff(bar[t >= 5]) <= 1e-9)                          # one-way
    assert (score[t > 25] > bar[t > 25]).mean() > 0.9                    # the break stays flagged


def test_deviation_roc_takes_the_ratchet_rule():
    from trajmem.metrics import deviation_roc

    t = np.arange(0, 30, 0.005)
    score = np.where(t < 20, 10.0, 80.0) + np.random.default_rng(1).normal(0, 1.0, len(t))
    out = deviation_roc(score, t, [20.0], threshold="ratchet")
    assert out["latency_s"] < 0.5 and out["fp_per_min"] == 0 and out["auc"] > 0.99
    with pytest.raises(ValueError):
        deviation_roc(score, t, [20.0], threshold="sideways")


def test_ratchet_alarm_matches_the_batch_rule_step_by_step():
    from trajmem.metrics import RatchetAlarm, deviation_roc, ratchet_threshold

    t = np.arange(0, 30, 0.005)
    score = np.where(t < 8, 40.0, 10.0) + np.random.default_rng(0).normal(0, 1.0, len(t))
    score[t > 20] += 60.0
    alarm = RatchetAlarm()
    live = np.array([alarm.update(s, ti) for s, ti in zip(score, t)])
    batch = ratchet_threshold(score, t)
    assert np.allclose(alarm.bar, batch[-1])
    flagged = deviation_roc(score, t, [20.0], threshold=batch)
    assert live[t > 21].mean() > 0.9 and not live[t < 20].any()
    assert flagged["latency_s"] < 0.5


def test_ade_is_the_mean_over_horizons_and_fde_the_longest():
    r = ade_fde({0.05: 2.0, 0.1: 4.0, 0.2: 6.0})
    assert r["ade"] == 4.0 and r["fde"] == 6.0
    assert np.isnan(ade_fde({0.05: np.nan})["ade"])


def test_k_is_calibrated_per_input():
    from trajmem.metrics import RATCHET_K_BY_INPUT, k_for_input

    assert k_for_input("centroid") == RATCHET_K_BY_INPUT["centroid"] == 25.0
    assert k_for_input("snn") == 16.0                      # the learned localiser needs a lower bar
    assert k_for_input(None) == k_for_input("centroid")    # the default input
    assert k_for_input("something else") == 25.0
