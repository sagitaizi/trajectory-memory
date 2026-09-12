import numpy as np
import pytest

from scripts.mark_breaks import break_clips, event_rate, nearest_mark
from trajmem.data import read_deviation_times, write_deviation_times
from trajmem.simulate import EVENT_DTYPE


def events_at(times_us):
    ev = np.zeros(len(times_us), dtype=EVENT_DTYPE)
    ev["timestamp"] = times_us
    return ev


def a_clip_dir(root, slug, times=None):
    d = root / slug
    d.mkdir(parents=True)
    (d / f"{slug}.aedat4").write_bytes(b"")
    if times is not None:
        write_deviation_times(d / f"{slug}.aedat4", times)
    return d


# --- the rate strip ----------------------------------------------------------

def test_event_rate_splits_the_clip_into_equal_slices():
    assert len(event_rate(events_at([0, 500_000]), 1.0, bins=10)) == 10


def test_event_rate_is_events_per_second_not_a_raw_count():
    """Ten events inside a 0.5 s slice is a rate of 20/s, not a count of 10."""
    rate = event_rate(events_at([0] * 10), 1.0, bins=2)
    assert rate[0] == pytest.approx(20.0)


def test_event_rate_finds_a_burst_in_the_second_half():
    ev = events_at([0] * 5 + [700_000] * 50)
    rate = event_rate(ev, 1.0, bins=2)
    assert rate[1] > rate[0] * 5


def test_event_rate_rejects_a_non_positive_bin_count():
    with pytest.raises(ValueError):
        event_rate(events_at([0]), 1.0, bins=0)


def test_event_rate_rejects_a_non_positive_duration():
    with pytest.raises(ValueError):
        event_rate(events_at([0]), 0.0)


# --- finding an existing mark ------------------------------------------------

def test_nearest_mark_finds_one_inside_the_tolerance():
    assert nearest_mark([2.0, 9.0], 2.01, tol=0.05) == 2.0


def test_nearest_mark_ignores_one_outside_the_tolerance():
    assert nearest_mark([2.0, 9.0], 5.0, tol=0.05) is None


def test_nearest_mark_picks_the_closer_of_two():
    assert nearest_mark([2.0, 2.4], 2.3, tol=1.0) == 2.4


def test_nearest_mark_of_nothing_is_none():
    assert nearest_mark([], 1.0, tol=1.0) is None


# --- the side-car ------------------------------------------------------------

def test_deviation_times_round_trip(tmp_path):
    clip = a_clip_dir(tmp_path, "wide_break") / "wide_break.aedat4"
    write_deviation_times(clip, [12.5])
    assert read_deviation_times(clip) == [12.5]


def test_deviation_times_are_sorted_on_write(tmp_path):
    clip = a_clip_dir(tmp_path, "wide_break") / "wide_break.aedat4"
    write_deviation_times(clip, [9.0, 2.0])
    assert read_deviation_times(clip) == [2.0, 9.0]


def test_an_unmarked_clip_reports_no_break_times(tmp_path):
    clip = a_clip_dir(tmp_path, "wide_break") / "wide_break.aedat4"
    assert read_deviation_times(clip) == []


# --- which clips need marking ------------------------------------------------

def test_break_clips_picks_deviation_clips_only(tmp_path):
    a_clip_dir(tmp_path, "small_01")
    a_clip_dir(tmp_path, "small_break")
    assert [c.stem for c in break_clips(tmp_path)] == ["small_break"]


def test_break_clips_finds_them_in_a_break_subdirectory(tmp_path):
    a_clip_dir(tmp_path / "break", "fan_brush_break_01")
    assert [c.stem for c in break_clips(tmp_path)] == ["fan_brush_break_01"]


def test_break_clips_skips_those_already_marked(tmp_path):
    a_clip_dir(tmp_path, "small_break", times=[4.0])
    assert break_clips(tmp_path) == []


def test_remark_includes_those_already_marked(tmp_path):
    a_clip_dir(tmp_path, "small_break", times=[4.0])
    assert [c.stem for c in break_clips(tmp_path, remark=True)] == ["small_break"]
