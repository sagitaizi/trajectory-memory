import numpy as np
import pytest

from scripts.mark_anchors import accumulate, marked_instant, unmarked_clips
from trajmem.data import read_anchor, write_anchor
from trajmem.simulate import EVENT_DTYPE


def events_at(pixels, times_us=None):
    ev = np.zeros(len(pixels), dtype=EVENT_DTYPE)
    ev["x"] = [p[0] for p in pixels]
    ev["y"] = [p[1] for p in pixels]
    ev["timestamp"] = times_us if times_us is not None else [0] * len(pixels)
    return ev


def a_clip_dir(root, slug, anchor=None):
    d = root / slug
    d.mkdir(parents=True)
    (d / f"{slug}.aedat4").write_bytes(b"")
    if anchor is not None:
        write_anchor(d / f"{slug}.aedat4", *anchor)
    return d


# --- the anchor side-car -----------------------------------------------------

def test_write_anchor_round_trips_through_read_anchor(tmp_path):
    clip = a_clip_dir(tmp_path, "scan_pan_slow_01") / "scan_pan_slow_01.aedat4"
    write_anchor(clip, 301.5, 244.0)
    assert read_anchor(clip) == (301.5, 244.0, 0.0)


def test_write_anchor_round_trips_the_instant_it_was_marked_at(tmp_path):
    clip = a_clip_dir(tmp_path, "scan_pan_slow_01") / "scan_pan_slow_01.aedat4"
    write_anchor(clip, 301.5, 244.0, t_s=0.01)
    assert read_anchor(clip) == (301.5, 244.0, 0.01)


def test_read_anchor_defaults_the_instant_to_the_trimmed_start(tmp_path):
    """A side-car written before the instant was recorded still means t=0."""
    clip_dir = a_clip_dir(tmp_path, "scan_diag_01")
    (clip_dir / "scan_diag_01.anchor.json").write_text('{"x": 5.0, "y": 6.0}')
    assert read_anchor(clip_dir) == (5.0, 6.0, 0.0)


def test_marked_instant_is_the_centre_of_the_smear():
    """A target crossing the window smears across it; a click means the smear's centre."""
    window_s = 0.02
    times = np.arange(0, 20_000, 100)
    xs = 10 + times // 1000                      # 1 px per ms
    ev = events_at(list(zip(xs, [5] * len(xs))), times_us=times)

    img = accumulate(ev, 0.0, window_s, (40, 10))
    columns = img.sum(axis=0)
    smear_centre = float((columns * np.arange(40)).sum() / columns.sum())
    assert abs(smear_centre - (10 + marked_instant(window_s) * 1000)) < 1.0


def test_write_anchor_names_the_side_car_after_the_clip(tmp_path):
    clip = a_clip_dir(tmp_path, "scan_diag_02") / "scan_diag_02.aedat4"
    written = write_anchor(clip, 10.0, 20.0)
    assert written.name == "scan_diag_02.anchor.json"
    assert written.exists()


# --- the frame the marker draws on -------------------------------------------

def test_accumulate_lights_the_pixels_events_landed_on():
    img = accumulate(events_at([(3, 4)]), 0.0, 0.05, (8, 6))
    assert img.shape == (6, 8)
    assert img.dtype == np.uint8
    assert img[4, 3] > 0
    assert img[0, 0] == 0


def test_accumulate_is_brighter_where_events_are_denser():
    img = accumulate(events_at([(1, 1)] * 5 + [(6, 5)]), 0.0, 0.05, (8, 6))
    assert img[1, 1] > img[5, 6] > 0


def test_accumulate_ignores_events_outside_the_window():
    ev = events_at([(1, 1), (6, 5)], times_us=[0, 80_000])
    img = accumulate(ev, 0.0, 0.05, (8, 6))
    assert img[1, 1] > 0
    assert img[5, 6] == 0


def test_accumulate_starts_the_window_where_it_is_told():
    ev = events_at([(1, 1), (6, 5)], times_us=[0, 80_000])
    img = accumulate(ev, 0.06, 0.05, (8, 6))
    assert img[1, 1] == 0
    assert img[5, 6] > 0


def test_accumulate_brightens_a_sparse_frame_without_widening_the_window():
    """Dimness and smear are separate problems; gain fixes one without causing the other."""
    ev = events_at([(3, 4)])
    dim = accumulate(ev, 0.0, 0.02, (8, 6))
    bright = accumulate(ev, 0.0, 0.02, (8, 6), gain=4 * int(dim[4, 3]))
    assert bright[4, 3] > dim[4, 3]
    assert bright[0, 0] == 0


def test_accumulate_returns_a_blank_frame_when_no_events_land():
    img = accumulate(events_at([]), 0.0, 0.05, (8, 6))
    assert img.shape == (6, 8)
    assert not img.any()


def test_accumulate_drops_events_outside_the_sensor():
    """Corrupt coordinates must not wrap round to a valid pixel."""
    img = accumulate(events_at([(99, 99), (3, 4)]), 0.0, 0.05, (8, 6))
    assert img[4, 3] > 0
    assert img.sum() == img[4, 3]


# --- which clips still need marking ------------------------------------------

def test_unmarked_clips_finds_every_clip_in_the_group(tmp_path):
    a_clip_dir(tmp_path, "scan_pan_slow_01")
    a_clip_dir(tmp_path, "scan_diag_02")
    assert [p.stem for p in unmarked_clips(tmp_path)] == ["scan_diag_02", "scan_pan_slow_01"]


def test_unmarked_clips_skips_clips_that_already_have_an_anchor(tmp_path):
    a_clip_dir(tmp_path, "scan_pan_slow_01", anchor=(320.0, 240.0))
    a_clip_dir(tmp_path, "scan_diag_02")
    assert [p.stem for p in unmarked_clips(tmp_path)] == ["scan_diag_02"]


def test_unmarked_clips_can_return_the_marked_ones_too(tmp_path):
    a_clip_dir(tmp_path, "scan_pan_slow_01", anchor=(320.0, 240.0))
    a_clip_dir(tmp_path, "scan_diag_02")
    found = unmarked_clips(tmp_path, remark=True)
    assert [p.stem for p in found] == ["scan_diag_02", "scan_pan_slow_01"]


def test_unmarked_clips_reaches_into_a_break_subdirectory(tmp_path):
    a_clip_dir(tmp_path, "fan_brush_slow_01")
    a_clip_dir(tmp_path / "break", "fan_brush_break_01")
    assert [p.stem for p in unmarked_clips(tmp_path)] == [
        "fan_brush_break_01", "fan_brush_slow_01"]


def test_unmarked_clips_rejects_a_group_that_is_not_there(tmp_path):
    with pytest.raises(FileNotFoundError):
        unmarked_clips(tmp_path / "nope")
