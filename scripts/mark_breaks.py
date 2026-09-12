"""Mark the instant a clip's motion stops matching its repeating path.

    python scripts/mark_breaks.py corpus/real/pendulum/wide_break
    python scripts/mark_breaks.py --group fan

Plays the clip and records the instants you call a break, into <slug>.deviation.json.
Phase D scores detection latency against these, so what matters is *when*, not where --
which is why this is a player with fine stepping rather than a click-per-frame tool.

Under the picture is the event rate over the whole clip with the playhead on it. A
break usually shows there as a step or spike (scan_pan_slow_break_01 runs ~2.5x its
base rate at the break), so the strip says roughly where to look before you look.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import _thesis_path  # noqa: F401,E402  (adds the thesis repo to sys.path)
import numpy as np  # noqa: E402

from scripts.mark_anchors import accumulate  # noqa: E402
from scripts.replay_gt import frame_times, to_pixels  # noqa: E402
from trajmem.data import (load_recording, read_deviation_times,  # noqa: E402
                          write_deviation_times)

_VIEW_BRIGHTNESS = 40
_VIEW_WINDOW_S = 0.02
_WINDOW = "mark breaks"
_STRIP_H = 70
_HELP = ("space=play/pause  ,/.=step  </>=jump 1s  m=mark break  u=unmark  "
         "[ ]=view window  - +=view brightness  s=save  q=quit")


class _Quit(Exception):
    """The operator asked to stop marking."""


def event_rate(events, duration_s: float, bins: int = 240) -> np.ndarray:
    """Events per second in each of `bins` equal slices of the clip."""
    if bins <= 0:
        raise ValueError("bins must be positive")
    if duration_s <= 0:
        raise ValueError("duration must be positive")
    edges = np.linspace(0, duration_s * 1e6, bins + 1)
    counts, _ = np.histogram(events["timestamp"], bins=edges)
    return counts / (duration_s / bins)


def nearest_mark(marks, t: float, tol: float):
    """The marked instant closest to `t`, or None if none is within `tol`."""
    if not marks:
        return None
    best = min(marks, key=lambda m: abs(m - t))
    return best if abs(best - t) <= tol else None


def strip_image(rate: np.ndarray, width: int, height: int, t: float,
                duration_s: float, marks) -> np.ndarray:
    """The rate trace as an image, with the playhead and any marks drawn on it."""
    import cv2

    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (28, 28, 28)
    if rate.max() > 0:
        ys = height - 1 - (rate / rate.max() * (height - 6)).astype(int)
        xs = (np.linspace(0, width - 1, len(rate))).astype(int)
        cv2.polylines(img, [np.stack([xs, ys], axis=1)], False, (90, 200, 90), 1,
                      cv2.LINE_AA)

    for m in marks:
        x = int(m / duration_s * (width - 1))
        cv2.line(img, (x, 0), (x, height - 1), (0, 0, 255), 1)

    x = int(t / duration_s * (width - 1))
    cv2.line(img, (x, 0), (x, height - 1), (0, 220, 255), 1)
    return img


def _draw(img, marks, view_zoom, label, t, duration_s, view_window_s, view_brightness,
          rate):
    import cv2

    z = view_zoom
    view = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    view = cv2.resize(view, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)

    on_mark = nearest_mark(marks, t, tol=view_window_s)
    hud = (f"{label}   t={t:6.2f}s / {duration_s:.1f}s   "
           f"view: {view_window_s * 1000:.0f}ms bright {view_brightness}   "
           f"{len(marks)} break{'' if len(marks) == 1 else 's'}")
    cv2.putText(view, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 220, 255), 1, cv2.LINE_AA)
    if on_mark is not None:
        cv2.putText(view, f"BREAK MARKED @ {on_mark:.2f}s", (8, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    if marks:
        cv2.putText(view, "  ".join(f"{m:.2f}" for m in sorted(marks)), (8, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(view, _HELP, (8, view.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.40, (190, 190, 190), 1, cv2.LINE_AA)

    strip = strip_image(rate, view.shape[1], _STRIP_H, t, duration_s, marks)
    return np.vstack([view, strip])


def mark(clip, label: str, fps: float, view_window_s: float, view_zoom: float,
         view_brightness: int) -> list:
    """Play one clip, returning the instants marked as breaks. Raises _Quit."""
    import cv2

    res = clip.meta["resolution"]
    duration_s = clip.duration_us / 1e6
    times = frame_times(duration_s, fps)
    rate = event_rate(clip.events, duration_s)
    marks = list(read_deviation_times(
        pathlib.Path(clip.source) / f"{clip.meta['slug']}.aedat4"))

    i, playing = 0, True
    delay = max(1, int(1000 / fps))
    cv2.namedWindow(_WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            t = times[i]
            img = accumulate(clip.events, t, view_window_s, res, gain=view_brightness)
            cv2.imshow(_WINDOW, _draw(img, marks, view_zoom, label, t, duration_s,
                                      view_window_s, view_brightness, rate))
            if cv2.getWindowProperty(_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                raise _Quit
            key = cv2.waitKeyEx(delay if playing else 20)

            if key in (ord("q"), 27):
                raise _Quit
            elif key == ord("s"):
                break
            elif key == ord(" "):
                playing = not playing
            elif key == ord("m"):
                playing = False
                if nearest_mark(marks, t, tol=view_window_s) is None:
                    marks.append(round(float(t), 3))
            elif key == ord("u"):
                near = nearest_mark(marks, t, tol=1.0)
                if near is not None:
                    marks.remove(near)
            elif key == ord("."):
                playing, i = False, min(i + 1, len(times) - 1)
            elif key == ord(","):
                playing, i = False, max(i - 1, 0)
            elif key == ord(">"):
                playing, i = False, min(i + int(fps), len(times) - 1)
            elif key == ord("<"):
                playing, i = False, max(i - int(fps), 0)
            elif key == ord("["):
                view_window_s = max(0.002, view_window_s / 1.5)
            elif key == ord("]"):
                view_window_s = min(0.5, view_window_s * 1.5)
            elif key == ord("-"):
                view_brightness = max(1, int(view_brightness / 1.5))
            elif key in (ord("+"), ord("=")):
                view_brightness = min(400, int(view_brightness * 1.5) + 1)
            elif key == ord("r"):
                i, playing = 0, True
            elif playing:
                i += 1
                if i >= len(times):
                    i, playing = len(times) - 1, False
    finally:
        cv2.destroyAllWindows()
    return sorted(marks)


def break_clips(group_dir, remark: bool = False) -> list[pathlib.Path]:
    """Deviation clips under `group_dir` still needing a break time."""
    from trajmem.data import describe_clip

    group_dir = pathlib.Path(group_dir)
    if not group_dir.is_dir():
        raise FileNotFoundError(group_dir)
    clips = [c for c in sorted(group_dir.rglob("*.aedat4"))
             if describe_clip(c)["is_break"]]
    return clips if remark else [c for c in clips if not read_deviation_times(c)]


def _targets(args) -> list[pathlib.Path]:
    if args.clip:
        p = pathlib.Path(args.clip)
        return [p if p.suffix == ".aedat4" else p / f"{p.name}.aedat4"]
    if args.group:
        return break_clips(pathlib.Path(args.corpus) / args.group, args.remark)
    corpus = pathlib.Path(args.corpus)
    return [c for g in sorted(p for p in corpus.iterdir() if p.is_dir())
            for c in break_clips(g, args.remark)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clip", nargs="?", help="one clip directory or .aedat4")
    p.add_argument("--group", help="corpus group to walk, e.g. fan")
    p.add_argument("--corpus", default="corpus/real")
    p.add_argument("--remark", action="store_true",
                   help="include clips whose break time is already marked")
    p.add_argument("--fps", type=float, default=30.0, help="frames played per second")
    p.add_argument("--view-window", type=float, default=_VIEW_WINDOW_S, metavar="S",
                   help="display only: seconds of events summed into each shown frame")
    p.add_argument("--view-brightness", type=int, default=_VIEW_BRIGHTNESS, metavar="N",
                   help="display only: event count -> grey level")
    p.add_argument("--view-zoom", type=float, default=2.0, metavar="Z",
                   help="display only: window magnification")
    args = p.parse_args()

    clips = _targets(args)
    if not clips:
        print("nothing to mark")
        return

    for i, path in enumerate(clips, 1):
        clip = load_recording(path)
        label = f"{path.stem}   {i}/{len(clips)}"
        try:
            marks = mark(clip, label, args.fps, args.view_window, args.view_zoom,
                         args.view_brightness)
        except _Quit:
            print("stopped")
            break
        if not marks:
            print(f"{path.stem}: no break marked, nothing written")
            continue
        out = write_deviation_times(path, marks)
        print(f"{path.stem}: {', '.join(f'{m:.2f}s' for m in marks)} -> {out}")


if __name__ == "__main__":
    main()
