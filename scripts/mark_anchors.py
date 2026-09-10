"""Mark the anchor pixel each Setup 4a clip's ground truth is referenced to.

    python scripts/mark_anchors.py --group wall
    python scripts/mark_anchors.py corpus/real/wall/scan_pan_slow_01

Shows a window of accumulated events from the start of the *trimmed* clip and writes
the clicked pixel to <slug>.anchor.json. Click the **centre of the target**: a corner
on one clip and a centre on another would offset their ground truths by half a target.

The window smears the target across itself, so a click means the window's centre
instant, not its first -- that instant goes into the side-car and `ego_gt` reads the
encoders there. Widening the window is therefore safe; it costs sharpness, not accuracy.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import _thesis_path  # noqa: F401,E402  (adds the thesis repo to sys.path)
import numpy as np  # noqa: E402

from trajmem.data import load_recording, read_anchor, write_anchor  # noqa: E402

# A slow sweep fires few events and looks dim; widening the window would brighten it
# but smear the target, which is exactly the precision the anchor needs. So brightness
# is its own control.
_GAIN = 40  # counts -> grey level
_WINDOW = "mark anchor"
_HELP = ("CLICK THE TARGET'S CENTRE   arrows/hjkl=nudge  [ ]=window  - +=brightness  "
         "r=reset  enter=save  s=skip  q=quit")


class _Quit(Exception):
    """The operator asked to stop marking."""


def accumulate(events, start_s: float, window_s: float, resolution,
               gain: int = _GAIN) -> np.ndarray:
    """Events in [start_s, start_s + window_s) as a uint8 image, brighter where denser.

    Assumes `events` is sorted by timestamp, as every Clip's is.
    """
    w, h = resolution
    lo = np.searchsorted(events["timestamp"], int(start_s * 1e6))
    hi = np.searchsorted(events["timestamp"], int((start_s + window_s) * 1e6))
    ev = events[lo:hi]

    xs, ys = ev["x"].astype(np.int64), ev["y"].astype(np.int64)
    on_sensor = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    counts = np.bincount(ys[on_sensor] * w + xs[on_sensor], minlength=w * h)
    return np.clip(counts.reshape(h, w) * gain, 0, 255).astype(np.uint8)


def marked_instant(window_s: float) -> float:
    """The instant a click refers to: the centre of the displayed window.

    The window renders [0, window_s), across which a moving target smears, so the
    pixel a human picks out is where the target was halfway through it.
    """
    return window_s / 2


def unmarked_clips(group_dir, remark: bool = False) -> list[pathlib.Path]:
    """Clip files under `group_dir` still needing an anchor, in a stable order."""
    group_dir = pathlib.Path(group_dir)
    if not group_dir.is_dir():
        raise FileNotFoundError(group_dir)
    clips = sorted(group_dir.rglob("*.aedat4"))
    return clips if remark else [c for c in clips if read_anchor(c) is None]


def _view(img, point, scale, label, window_s, gain):
    import cv2

    view = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    if point is not None:
        x, y = int(round(point[0] * scale)), int(round(point[1] * scale))
        cv2.line(view, (x - 14, y), (x + 14, y), (0, 255, 0), 1)
        cv2.line(view, (x, y - 14), (x, y + 14), (0, 255, 0), 1)
        cv2.circle(view, (x, y), 16, (0, 255, 0), 1)
        cv2.putText(view, f"{point[0]:.1f}, {point[1]:.1f}", (x + 20, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(view, f"{label}   window={window_s * 1000:.0f}ms  "
                f"t={marked_instant(window_s) * 1000:.0f}ms  gain={gain}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(view, _HELP, (8, view.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1, cv2.LINE_AA)
    return view


_NUDGE = {2424832: (-1, 0), 2555904: (1, 0), 2490368: (0, -1), 2621440: (0, 1),
          ord("h"): (-1, 0), ord("l"): (1, 0), ord("k"): (0, -1), ord("j"): (0, 1)}


def mark(clip, label: str, window_s: float, scale: float):
    """Run the marking window for one clip.

    Returns `(x, y, t_s)` -- the clicked pixel and the instant it holds at -- or None
    if the operator skipped the clip.
    """
    import cv2

    state = {"point": None}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["point"] = (x / scale, y / scale)

    gain = _GAIN
    cv2.namedWindow(_WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(_WINDOW, on_mouse)
    img = accumulate(clip.events, 0.0, window_s, clip.meta["resolution"], gain)
    try:
        while True:
            cv2.imshow(_WINDOW, _view(img, state["point"], scale, label, window_s, gain))
            key = cv2.waitKeyEx(20)
            if key == -1:
                continue
            if key in (ord("q"), 27):
                raise _Quit
            if key == ord("s"):
                return None
            if key in (13, 10) and state["point"] is not None:
                return (*state["point"], marked_instant(window_s))
            if key == ord("r"):
                state["point"] = None
            elif key in (ord("["), ord("]")):
                window_s = float(np.clip(window_s * (1 / 1.5 if key == ord("[") else 1.5),
                                         0.002, 2.0))
                img = accumulate(clip.events, 0.0, window_s, clip.meta["resolution"], gain)
            elif key in (ord("-"), ord("="), ord("+")):
                gain = int(np.clip(gain // 2 if key == ord("-") else gain * 2, 1, 255))
                img = accumulate(clip.events, 0.0, window_s, clip.meta["resolution"], gain)
            elif key in _NUDGE and state["point"] is not None:
                dx, dy = _NUDGE[key]
                state["point"] = (state["point"][0] + dx, state["point"][1] + dy)
    finally:
        cv2.destroyWindow(_WINDOW)


def _targets(args) -> list[pathlib.Path]:
    if args.clip:
        p = pathlib.Path(args.clip)
        return [p if p.suffix == ".aedat4" else p / f"{p.name}.aedat4"]
    if not args.group:
        raise SystemExit("give a clip path or --group")
    return unmarked_clips(pathlib.Path(args.corpus) / args.group, args.remark)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clip", nargs="?", help="one clip directory or .aedat4")
    p.add_argument("--group", help="corpus group to walk, e.g. wall")
    p.add_argument("--corpus", default="corpus/real")
    p.add_argument("--remark", action="store_true", help="include already-marked clips")
    p.add_argument("--window", type=float, default=0.02, help="accumulation window (s)")
    p.add_argument("--scale", type=float, default=2.0, help="display magnification")
    args = p.parse_args()

    clips = _targets(args)
    if not clips:
        print("nothing to mark")
        return

    marked = skipped = 0
    for i, path in enumerate(clips, 1):
        clip = load_recording(path)
        if not clip.meta["has_motors"]:
            print(f"{path.stem}: no encoder side-car, not a 4a clip - skipping")
            skipped += 1
            continue
        label = f"{path.stem}   {i}/{len(clips)}"
        try:
            point = mark(clip, label, args.window, args.scale)
        except _Quit:
            print("stopped")
            break
        if point is None:
            print(f"{path.stem}: skipped")
            skipped += 1
            continue
        x, y, t_s = point
        sidecar = write_anchor(path, x, y, t_s)
        marked += 1
        print(f"{path.stem}: anchor ({x:.1f}, {y:.1f}) at t={t_s * 1000:.0f}ms "
              f"-> {sidecar.name}")

    print(f"marked {marked}, skipped {skipped}, of {len(clips)} clip(s)")


if __name__ == "__main__":
    main()
