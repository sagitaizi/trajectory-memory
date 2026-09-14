import numpy as np
import pytest

from trajmem.trajectories import Deviation, TrajectorySpec, sample


def circle(r=0.3, period=2.0, center=(0.5, 0.5), phase0=0.0, deviations=None):
    return TrajectorySpec(
        shape="circle", size=(r, r), period_s=period,
        center=center, phase0=phase0, deviations=deviations or [],
    )


def test_circle_radius_is_constant():
    spec = circle(r=0.3, center=(0.5, 0.5))
    t = np.linspace(0, spec.period_s, 200, endpoint=False)
    pos = sample(spec, t)
    d = np.hypot(pos[:, 0] - 0.5, pos[:, 1] - 0.5)
    assert np.allclose(d, 0.3, atol=1e-9)


def test_circle_starts_at_phase0_on_positive_x_axis():
    spec = circle(r=0.3, center=(0.5, 0.5), phase0=0.0)
    assert np.allclose(sample(spec, 0.0), [0.8, 0.5], atol=1e-9)


def test_circle_repeats_after_one_period():
    spec = circle(r=0.25, period=1.7)
    t = np.linspace(0.0, 5.0, 300)
    assert np.allclose(sample(spec, t), sample(spec, t + spec.period_s), atol=1e-9)


def test_ellipse_has_independent_semi_axes():
    spec = TrajectorySpec(shape="ellipse", size=(0.4, 0.2), period_s=2.0, center=(0.5, 0.5))
    t = np.linspace(0, spec.period_s, 400, endpoint=False)
    pos = sample(spec, t)
    assert np.isclose((pos[:, 0].max() - pos[:, 0].min()) / 2, 0.4, atol=1e-6)
    assert np.isclose((pos[:, 1].max() - pos[:, 1].min()) / 2, 0.2, atol=1e-6)


def test_figure8_is_at_centre_at_start_and_half_period():
    spec = TrajectorySpec(shape="figure8", size=(0.3, 0.2), period_s=2.0, center=(0.5, 0.5))
    assert np.allclose(sample(spec, 0.0), [0.5, 0.5], atol=1e-9)
    assert np.allclose(sample(spec, 1.0), [0.5, 0.5], atol=1e-9)


def test_figure8_extents_follow_gerono():
    spec = TrajectorySpec(shape="figure8", size=(0.3, 0.2), period_s=2.0, center=(0.5, 0.5))
    t = np.linspace(0, spec.period_s, 4000, endpoint=False)
    off = sample(spec, t) - np.array([0.5, 0.5])
    assert np.isclose(np.abs(off[:, 0]).max(), 0.3, atol=1e-4)   # x swings the full semi-axis
    assert np.isclose(np.abs(off[:, 1]).max(), 0.1, atol=1e-4)   # y swings sy/2


def test_lissajous_start_point_matches_closed_form():
    spec = TrajectorySpec(shape="lissajous", size=(0.3, 0.25), period_s=2.0,
                          center=(0.5, 0.5), lissajous=(3.0, 2.0, np.pi / 2))
    # u=0: x = sx*sin(delta) = 0.3, y = sy*sin(0) = 0
    assert np.allclose(sample(spec, 0.0), [0.8, 0.5], atol=1e-9)


def test_lissajous_closes_after_one_period_for_integer_ratio():
    spec = TrajectorySpec(shape="lissajous", size=(0.3, 0.3), period_s=2.0,
                          center=(0.5, 0.5), lissajous=(3.0, 2.0, np.pi / 2))
    t = np.linspace(0, 5, 250)
    assert np.allclose(sample(spec, t), sample(spec, t + spec.period_s), atol=1e-9)


def test_sample_scalar_t_returns_shape_2_array():
    assert sample(circle(), 0.3).shape == (2,)


def test_sample_array_t_returns_n_by_2_array():
    assert sample(circle(), np.linspace(0, 1, 17)).shape == (17, 2)


def test_no_deviations_matches_pure_analytic_path():
    plain = circle(r=0.3, period=2.0)
    withempty = circle(r=0.3, period=2.0, deviations=[])
    t = np.linspace(0, 4, 123)
    assert np.array_equal(sample(plain, t), sample(withempty, t))


def test_speed_change_scales_instantaneous_speed_after_event():
    dev = Deviation(at_t=1.0, kind="speed_change", params={"factor": 2.0})
    spec = circle(r=0.3, period=2.0, deviations=[dev])
    t = np.linspace(0, 2.0, 20001)
    pos = sample(spec, t)
    speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) / np.diff(t)
    tmid = 0.5 * (t[:-1] + t[1:])
    before = speed[(tmid > 0.3) & (tmid < 0.9)].mean()
    after = speed[(tmid > 1.1) & (tmid < 1.7)].mean()
    assert np.isclose(after / before, 2.0, rtol=1e-2)


def test_speed_change_keeps_path_continuous_at_event():
    dev = Deviation(at_t=1.0, kind="speed_change", params={"factor": 2.5})
    spec = circle(r=0.3, period=2.0, deviations=[dev])
    eps = 1e-6
    jump = np.linalg.norm(sample(spec, 1.0 + eps) - sample(spec, 1.0 - eps))
    assert jump < 1e-4


def test_shrink_scales_radius_after_event():
    dev = Deviation(at_t=1.0, kind="shrink", params={"factor": 0.5})
    spec = circle(r=0.4, period=2.0, deviations=[dev])
    before = sample(spec, np.linspace(0, 0.99, 500))
    after = sample(spec, np.linspace(1.01, 2.0, 500))
    r_before = np.hypot(before[:, 0] - 0.5, before[:, 1] - 0.5).max()
    r_after = np.hypot(after[:, 0] - 0.5, after[:, 1] - 0.5).max()
    assert np.isclose(r_before, 0.4, atol=1e-3)
    assert np.isclose(r_after, 0.2, atol=1e-3)


def test_drift_translates_centre_linearly_after_event():
    dev = Deviation(at_t=1.0, kind="drift", params={"vel": (0.1, -0.05)})
    spec = circle(r=0.2, period=2.0, deviations=[dev])
    base = circle(r=0.2, period=2.0)
    t = np.array([1.5, 2.0, 3.0])
    delta = sample(spec, t) - sample(base, t)
    expected = np.array([[0.1, -0.05]]) * (t - 1.0)[:, None]
    assert np.allclose(delta, expected, atol=1e-9)


def test_drift_has_no_effect_before_event():
    dev = Deviation(at_t=1.0, kind="drift", params={"vel": (0.3, 0.3)})
    spec = circle(r=0.2, period=2.0, deviations=[dev])
    base = circle(r=0.2, period=2.0)
    t = np.linspace(0, 0.999, 200)
    assert np.allclose(sample(spec, t), sample(base, t), atol=1e-12)


def test_switch_shape_changes_shape_after_event_with_continuous_phase():
    dev = Deviation(at_t=1.0, kind="switch_shape", params={"to": "figure8"})
    spec = circle(r=0.3, period=2.0, center=(0.5, 0.5), deviations=[dev])
    before = sample(spec, 0.5) - 0.5
    assert np.isclose(np.hypot(*before), 0.3, atol=1e-9)          # still on the circle
    assert np.allclose(sample(spec, 1.0), [0.5, 0.5], atol=1e-9)  # figure8 at phi=pi -> centre
    after = sample(spec, np.linspace(1.0, 2.0, 3000))
    assert np.isclose(np.abs(after[:, 1] - 0.5).max(), 0.15, atol=1e-3)  # y half-extent sy/2


def test_unknown_deviation_kind_is_rejected():
    spec = circle(deviations=[Deviation(at_t=1.0, kind="wobble", params={})])
    with pytest.raises(ValueError):
        sample(spec, 1.0)


def test_multiple_deviations_compose():
    spec = circle(r=0.4, period=2.0, deviations=[
        Deviation(at_t=1.0, kind="shrink", params={"factor": 0.5}),
        Deviation(at_t=1.0, kind="drift", params={"vel": (0.1, 0.0)}),
    ])
    p = sample(spec, 2.0)   # phi = 2*pi -> offset direction (1, 0); 1 s after both events
    assert np.allclose(p, [0.5 + 0.1 + 0.2, 0.5], atol=1e-6)


# --- rotation ----------------------------------------------------------------

def test_rotation_turns_a_flat_ellipse_into_a_tilted_sweep():
    spec = TrajectorySpec(shape="ellipse", size=(0.3, 0.0), period_s=2.0,
                          center=(0.5, 0.5), rotation=np.pi / 4)
    t = np.linspace(0, spec.period_s, 200, endpoint=False)
    off = sample(spec, t) - np.array([0.5, 0.5])
    assert np.allclose(off[:, 0], off[:, 1], atol=1e-9)           # on the diagonal
    assert np.isclose(np.hypot(*off.T).max(), 0.3, atol=1e-6)      # full semi-axis reached


def test_rotation_by_quarter_turn_swaps_the_axes():
    flat = TrajectorySpec(shape="ellipse", size=(0.4, 0.2), period_s=2.0)
    turned = TrajectorySpec(shape="ellipse", size=(0.4, 0.2), period_s=2.0, rotation=np.pi / 2)
    t = np.linspace(0, 2.0, 400, endpoint=False)
    a, b = sample(flat, t) - 0.5, sample(turned, t) - 0.5
    assert np.isclose(np.abs(b[:, 1]).max(), 0.4, atol=1e-6)
    assert np.isclose(np.abs(b[:, 0]).max(), 0.2, atol=1e-6)
    assert np.allclose(np.hypot(*a.T), np.hypot(*b.T), atol=1e-9)  # a rigid turn


def test_rotation_leaves_a_circle_a_circle():
    spec = circle(r=0.3)
    spec.rotation = 1.1
    t = np.linspace(0, 2.0, 100)
    assert np.allclose(np.hypot(*(sample(spec, t) - 0.5).T), 0.3, atol=1e-9)


# --- fitting an ellipse to marks ----------------------------------------------

def test_fit_ellipse_recovers_a_tilted_ellipse_from_sparse_marks():
    from trajmem.trajectories import fit_ellipse

    truth = TrajectorySpec(shape="ellipse", size=(0.13, 0.07), period_s=1.2,
                           center=(0.54, 0.52), phase0=0.8, rotation=0.4)
    t = np.arange(0.025, 25.0, 0.15)
    marks = [(ti, *sample(truth, ti)) for ti in t]

    fitted = fit_ellipse(marks, period_s=1.2)
    assert fitted.shape == "ellipse"
    check = np.linspace(0, 25.0, 500)
    assert np.allclose(sample(fitted, check), sample(truth, check), atol=1e-6)


def test_fit_ellipse_finds_the_period_when_not_given():
    from trajmem.trajectories import fit_ellipse

    truth = TrajectorySpec(shape="ellipse", size=(0.13, 0.07), period_s=1.2,
                           center=(0.54, 0.52), phase0=0.8)
    t = np.arange(0.0, 25.0, 0.15)
    marks = [(ti, *sample(truth, ti)) for ti in t]
    fitted = fit_ellipse(marks)
    assert np.isclose(fitted.period_s, 1.2, atol=0.005)


def test_fit_ellipse_ignores_marks_with_unknown_position():
    from trajmem.trajectories import fit_ellipse

    truth = TrajectorySpec(shape="ellipse", size=(0.1, 0.05), period_s=1.0)
    t = np.arange(0.0, 10.0, 0.1)
    marks = [(ti, *sample(truth, ti)) for ti in t] + [(3.05, np.nan, np.nan)]
    fitted = fit_ellipse(marks, period_s=1.0)
    assert np.allclose(sample(fitted, t), sample(truth, t), atol=1e-6)
