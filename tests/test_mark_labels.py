import numpy as np
import pytest

from scripts.mark_labels import (coverage, label_times, marked_instant, to_normalised,
                                 to_pixels)
from trajmem.data import read_labels, write_labels


def a_clip_dir(root, slug):
    d = root / slug
    d.mkdir(parents=True)
    (d / f"{slug}.aedat4").write_bytes(b"")
    return d


# --- which instants get asked about ------------------------------------------

def test_label_times_starts_at_zero_and_stays_inside_the_clip():
    times = label_times(5.0, every_s=0.5)
    assert times[0] == 0.0
    assert times[-1] < 5.0


def test_label_times_spaces_instants_by_the_interval():
    assert label_times(2.0, every_s=0.5) == pytest.approx([0.0, 0.5, 1.0, 1.5])


def test_label_times_rejects_a_non_positive_interval():
    with pytest.raises(ValueError):
        label_times(5.0, every_s=0.0)


def test_a_click_means_the_centre_of_the_displayed_window():
    assert marked_instant(2.0, 0.02) == pytest.approx(2.01)


# --- units -------------------------------------------------------------------

def test_clicks_are_normalised_by_the_sensor():
    assert to_normalised((320.0, 120.0), (640, 480)) == pytest.approx((0.5, 0.25))


def test_normalising_round_trips_back_to_pixels():
    assert to_pixels(to_normalised((301.0, 244.0), (640, 480)),
                     (640, 480)) == pytest.approx((301.0, 244.0))


# --- the side-car ------------------------------------------------------------

def test_write_labels_round_trips_through_read_labels(tmp_path):
    clip = a_clip_dir(tmp_path, "small_01") / "small_01.aedat4"
    write_labels(clip, [(0.0, 0.1, 0.2), (0.5, 0.3, 0.4)])
    assert read_labels(clip) == [(0.0, 0.1, 0.2), (0.5, 0.3, 0.4)]


def test_write_labels_sorts_marks_given_out_of_order(tmp_path):
    clip = a_clip_dir(tmp_path, "small_01") / "small_01.aedat4"
    write_labels(clip, [(0.5, 0.3, 0.4), (0.0, 0.1, 0.2)])
    assert [row[0] for row in read_labels(clip)] == [0.0, 0.5]


def test_write_labels_replaces_what_was_there(tmp_path):
    clip = a_clip_dir(tmp_path, "small_01") / "small_01.aedat4"
    write_labels(clip, [(0.0, 0.1, 0.2), (0.5, 0.3, 0.4)])
    write_labels(clip, [(1.0, 0.7, 0.8)])
    assert read_labels(clip) == [(1.0, 0.7, 0.8)]


def test_write_labels_rejects_an_empty_mark_list(tmp_path):
    clip = a_clip_dir(tmp_path, "small_01") / "small_01.aedat4"
    with pytest.raises(ValueError):
        write_labels(clip, [])


# --- progress reporting ------------------------------------------------------

def test_coverage_reports_the_largest_hole_between_marks():
    marks = {0.0: (0.1, 0.1), 0.5: (0.2, 0.2), 2.0: (0.3, 0.3)}
    assert "largest gap 1.50s" in coverage(marks, 3.0)


def test_coverage_of_nothing_says_so():
    assert coverage({}, 3.0) == "0 marks"
