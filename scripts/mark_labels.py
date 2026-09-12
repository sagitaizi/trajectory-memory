"""Hand-label a target's path through a clip, one click per instant.

    python scripts/mark_labels.py corpus/real/pendulum/small_01
    python scripts/mark_labels.py --group wall --every 0.4

Steps through the clip at a fixed interval, showing accumulated events. Click the
**centre of the target** and it records that instant and advances. Marks are written
to <slug>.labels.csv and `data.label_gt` interpolates between them, which is how the
fan, pendulum and 4b clips get their ground truth.

Pick an interval fine enough that the path is near-straight between marks: linear
interpolation is exact on a straight leg and cuts the corner on a curve.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import _thesis_path  # noqa: F401,E402  (adds the thesis repo to sys.path)
import numpy as np  # noqa: E402

from scripts.mark_anchors import accumulate  # noqa: E402
from trajmem.data import load_recording, read_labels, write_labels  # noqa: E402

_VIEW_BRIGHTNESS = 40
_VIEW_WINDOW_S = 0.02
_WINDOW = "mark labels"
_HELP = ("CLICK THE TARGET'S CENTRE   n/space=skip  b=back  u=unmark  "
         "[ ]=view window  - +=view brightness  s=save  q=quit")


class _Quit(Exception):
    """The operator asked to stop marking."""


def label_times(duration_s: float, every_s: float) -> np.ndarray:
    """The instants to ask about, from zero up to the clip's end."""
    if every_s <= 0:
        raise ValueError("interval must be positive")
    return np.arange(0.0, duration_s, every_s)


def marked_instant(t: float, view_window_s: float) -> float:
    """A click means the centre of the displayed window, where the smear is centred."""
    return t + view_window_s / 2


def to_normalised(point, resolution) -> tuple[float, float]:
    """Clicks arrive in pixels; labels are stored normalised, as every gt is."""
    w, h = resolution
    return point[0] / w, point[1] / h


def to_pixels(point, resolution) -> tuple[float, float]:
    w, h = resolution
    return point[0] * w, point[1] * h


def existing_marks(clip) -> dict:
    """Labels already on disk, keyed by the instant they were marked at."""
    labels = read_labels(pathlib.Path(clip.source) / f"{clip.meta['slug']}.aedat4")
    return {round(t, 4): (x, y) for t, x, y in (labels or [])}


def resume_at(times, marks, view_window_s: float) -> int:
    """First instant with no mark yet, so a part-done clip picks up where it stopped."""
    for i, t in enumerate(times):
        if round(marked_instant(t, view_window_s), 4) not in marks:
            return i
    return 0                                    # all done: start over to review


def coverage(marks: dict, duration_s: float) -> str:
    if not marks:
        return "0 marks"
    ts = sorted(marks)
    gap = max(np.diff(ts)) if len(ts) > 1 else 0.0
    return (f"{len(marks)} marks  {ts[0]:.1f}-{ts[-1]:.1f}s  "
            f"largest gap {gap:.2f}s")


def _draw(img, marks, current, resolution, view_zoom, label, t, view_window_s,
          view_brightness, n_done, n_total):
    import cv2

    z = view_zoom
    view = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    view = cv2.resize(view, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)

    pts = [to_pixels(marks[k], resolution) for k in sorted(marks)]
    if len(pts) > 1:
        poly = np.array([[int(round(x * z)), int(round(y * z))] for x, y in pts])
        cv2.polylines(view, [poly], False, (140, 90, 0), 1, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(view, (int(round(x * z)), int(round(y * z))), 2, (255, 150, 0), -1)

    if current is not None:
        x, y = int(round(current[0] * z)), int(round(current[1] * z))
        cv2.line(view, (x - 14, y), (x + 14, y), (0, 255, 0), 1)
        cv2.line(view, (x, y - 14), (x, y + 14), (0, 255, 0), 1)
        cv2.circle(view, (x, y), 16, (0, 255, 0), 1)

    hud = (f"{label}   t={t:6.2f}s  {n_done}/{n_total}   "
           f"view: {view_window_s * 1000:.0f}ms bright {view_brightness}")
    cv2.putText(view, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(view, coverage(marks, 0), (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 150, 0), 1, cv2.LINE_AA)
    cv2.putText(view, _HELP, (8, view.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (190, 190, 190), 1, cv2.LINE_AA)
    return view


def mark(clip, label: str, every_s: float, view_window_s: float, view_zoom: float,
         view_brightness: int) -> dict:
    """Walk one clip's instants, returning marks keyed by instant. Raises _Quit."""
    import cv2

    res = clip.meta["resolution"]
    times = label_times(clip.duration_us / 1e6, every_s)
    marks = existing_marks(clip)
    state = {"click": None}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x / view_zoom, y / view_zoom)

    cv2.namedWindow(_WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(_WINDOW, on_mouse)
    i = resume_at(times, marks, view_window_s)
    try:
        while i < len(times):
            t = times[i]
            key_t = round(marked_instant(t, view_window_s), 4)
            img = accumulate(clip.events, t, view_window_s, res, gain=view_brightness)
            current = to_pixels(marks[key_t], res) if key_t in marks else None
            cv2.imshow(_WINDOW, _draw(img, marks, current, res, view_zoom, label, t,
                                      view_window_s, view_brightness, i + 1, len(times)))
            if cv2.getWindowProperty(_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                raise _Quit
            key = cv2.waitKeyEx(20)

            if state["click"] is not None:
                marks[key_t] = to_normalised(state["click"], res)
                state["click"] = None
                i += 1                              # a click means done with this one
            elif key in (ord("n"), ord(" ")):
                i += 1
            elif key == ord("b"):
                i = max(0, i - 1)
            elif key == ord("u"):
                marks.pop(key_t, None)
            elif key == ord("["):
                view_window_s = max(0.002, view_window_s / 1.5)
            elif key == ord("]"):
                view_window_s = min(0.5, view_window_s * 1.5)
            elif key == ord("-"):
                view_brightness = max(1, int(view_brightness / 1.5))
            elif key in (ord("+"), ord("=")):
                view_brightness = min(400, int(view_brightness * 1.5) + 1)
            elif key == ord("s"):
                break
            elif key in (ord("q"), 27):
                raise _Quit
    finally:
        cv2.destroyAllWindows()
    return marks


def _targets(args) -> list[pathlib.Path]:
    if args.clip:
        p = pathlib.Path(args.clip)
        return [p if p.suffix == ".aedat4" else p / f"{p.name}.aedat4"]
    if not args.group:
        raise SystemExit("give a clip path or --group")
    clips = sorted((pathlib.Path(args.corpus) / args.group).rglob("*.aedat4"))
    return clips if args.relabel else [c for c in clips if read_labels(c) is None]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clip", nargs="?", help="one clip directory or .aedat4")
    p.add_argument("--group", help="corpus group to walk, e.g. pendulum")
    p.add_argument("--corpus", default="corpus/real")
    p.add_argument("--relabel", action="store_true",
                   help="include clips that already have labels")
    p.add_argument("--every", type=float, default=0.5, metavar="S",
                   help="seconds between the instants you are asked to mark")
    p.add_argument("--view-window", type=float, default=_VIEW_WINDOW_S, metavar="S",
                   help="display only: seconds of events summed into each shown frame")
    p.add_argument("--view-brightness", type=int, default=_VIEW_BRIGHTNESS, metavar="N",
                   help="display only: event count -> grey level")
    p.add_argument("--view-zoom", type=float, default=2.0, metavar="Z",
                   help="display only: window magnification")
    args = p.parse_args()

    clips = _targets(args)
    if not clips:
        print("nothing to label")
        return

    for i, path in enumerate(clips, 1):
        clip = load_recording(path)
        label = f"{path.stem}   {i}/{len(clips)}"
        try:
            marks = mark(clip, label, args.every, args.view_window, args.view_zoom,
                         args.view_brightness)
        except _Quit:
            print("stopped")
            break
        if not marks:
            print(f"{path.stem}: no marks, nothing written")
            continue
        out = write_labels(path, [(t, x, y) for t, (x, y) in sorted(marks.items())])
        print(f"{path.stem}: {coverage(marks, clip.duration_us / 1e6)} -> {out}")


if __name__ == "__main__":
    main()
