"""Replay a clip with its ground truth drawn on top, to see whether the two agree.

    python scripts/replay_gt.py corpus/real/wall/scan_pan_slow_01
    python scripts/replay_gt.py corpus/real/wall/scan_pan_slow_01 --gt-scale 1.307
    python scripts/replay_gt.py corpus/real/wall/scan_pan_slow_01 --save

Plays accumulated event frames with a crosshair at gt(t), a fading trail of where the
ground truth has just been, the marked anchor, and a HUD. The point is to catch a
ground truth whose *shape* is right but whose scale or sign is not: a marker that
drifts off the target over a sweep is a calibration error, not a labelling one.

Needs a clip that has ground truth -- a marked anchor (Setup 4a) or hand-labels.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import _thesis_path  # noqa: F401,E402  (adds the thesis repo to sys.path)
import numpy as np  # noqa: E402

from scripts.mark_anchors import accumulate  # noqa: E402
from trajmem.data import load_recording  # noqa: E402

# Everything prefixed `view_` is display only: it changes what you see, never the
# events, the ground truth or anything written to disk. Only --gt-scale does that.
_VIEW_BRIGHTNESS = 40
_VIEW_WINDOW_S = 0.02
_WINDOW = "replay gt"
_TRAIL_S = 2.0
_HELP = ("space=play/pause  ,/.=step  </>=jump 1s  [ ]=view window  "
         "- +=view brightness  t=trail  r=restart  q=quit")


def frame_times(duration_s: float, fps: float) -> np.ndarray:
    """Instants to render, from 0 up to the clip's end."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    return np.arange(0, duration_s, 1.0 / fps)


def trail_points(gt, t: float, trail_s: float, n: int = 48) -> np.ndarray:
    """gt over the `trail_s` before `t`, oldest first, clipped at the clip's start."""
    if t <= 0 or trail_s <= 0:
        return np.asarray(gt(max(t, 0.0)), dtype=float).reshape(1, 2)
    start = max(0.0, t - trail_s)
    return np.atleast_2d(np.asarray(gt(np.linspace(start, t, n)), dtype=float))


def to_pixels(points, resolution) -> np.ndarray:
    """gt returns normalised coordinates; the display works in pixels."""
    w, h = resolution
    return np.atleast_2d(np.asarray(points, dtype=float)) * np.array([w, h], dtype=float)


def scaled_ticks_per_radian(ticks_per_radian, factor: float) -> tuple[float, float]:
    """Fewer ticks per radian means a wider swept angle, so a bigger pixel sweep.

    `factor` is measured as true-shift / predicted-shift, so dividing by it is the
    correction. See --gt-scale.
    """
    if factor <= 0:
        raise ValueError("gt scale must be positive")
    pan, tilt = ticks_per_radian
    return pan / factor, tilt / factor


def rescale_gt(clip, factor: float):
    """A copy of `clip` whose ground truth uses a corrected encoder scale.

    Rebuilds ego_gt from the clip's own side-cars rather than stretching pixels, so
    the correction goes through the same projection the real ground truth does.
    """
    from recording.player import MotorTrack
    from recording.recorder import motor_sidecar_path

    from dataclasses import replace

    from trajmem.data import Anchor, _ticks_per_radian, ego_gt
    from trajmem.simulate import load_intrinsics

    anchor = clip.meta.get("anchor")
    if anchor is None:
        raise SystemExit("--gt-scale only applies to encoder ground truth (Setup 4a)")

    anchor = Anchor(*anchor)
    d, slug = pathlib.Path(clip.source), clip.meta["slug"]
    track = MotorTrack.from_csv(motor_sidecar_path(d / f"{slug}.aedat4"))
    tpr = scaled_ticks_per_radian(_ticks_per_radian(d, slug), factor)
    gt = ego_gt((anchor.x, anchor.y), track, load_intrinsics(),
                clip.meta["t0_device_us"], tpr, anchor_t_s=anchor.t_s)
    return replace(clip, gt=gt, meta={**clip.meta, "gt_scale": factor})


def _ticks_at(track, t0_us: int, t: float):
    if track is None:
        return None
    return track.position_at(int(t0_us + t * 1e6))


def _draw(img, gt_px, trail_px, anchor, view_zoom, label, t, view_window_s,
          view_brightness, ticks, gt_scale=1.0):
    import cv2

    z = view_zoom
    view = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    view = cv2.resize(view, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)

    for i, (x, y) in enumerate(trail_px):
        fade = (i + 1) / len(trail_px)          # oldest dimmest
        cv2.circle(view, (int(round(x * z)), int(round(y * z))), 2,
                   (0, int(90 + 130 * fade), int(200 * fade)), -1)

    if anchor is not None:
        ax, ay = int(round(anchor[0] * z)), int(round(anchor[1] * z))
        cv2.drawMarker(view, (ax, ay), (200, 120, 255), cv2.MARKER_TILTED_CROSS, 14, 1)

    x, y = int(round(gt_px[0] * z)), int(round(gt_px[1] * z))
    cv2.line(view, (x - 14, y), (x + 14, y), (0, 255, 0), 1)
    cv2.line(view, (x, y - 14), (x, y + 14), (0, 255, 0), 1)
    cv2.circle(view, (x, y), 16, (0, 255, 0), 1)
    cv2.putText(view, f"{gt_px[0]:.1f}, {gt_px[1]:.1f}", (x + 20, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    hud = (f"{label}   t={t:6.2f}s  view: {view_window_s * 1000:.0f}ms "
           f"bright {view_brightness}")
    if ticks is not None:
        hud += f"   pan={ticks[0]:.0f} tilt={ticks[1]:.0f}"
    if gt_scale != 1.0:
        hud += f"   gt x{gt_scale:g}"
    cv2.putText(view, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(view, _HELP, (8, view.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (190, 190, 190), 1, cv2.LINE_AA)
    return view


def _motor_track(clip):
    """The encoder side-car, for the HUD's tick readout. None when the clip has none."""
    if not clip.meta.get("has_motors"):
        return None
    from recording.player import MotorTrack
    from recording.recorder import motor_sidecar_path

    d = pathlib.Path(clip.source)
    return MotorTrack.from_csv(motor_sidecar_path(d / f"{clip.meta['slug']}.aedat4"))


def render(clip, t: float, view_window_s: float, view_brightness: int, trail_s: float,
           view_zoom: float, label: str, track=None):
    """One overlaid frame: events in [t, t+view_window) with the ground truth on top.

    The marker is sampled at the window's *centre*, not its start: the displayed
    events smear the target across the whole window, so its midpoint is the instant
    the picture actually shows. This is the convention `mark_anchors` marks under.
    """
    res = clip.meta["resolution"]
    mid = t + view_window_s / 2
    img = accumulate(clip.events, t, view_window_s, res, gain=view_brightness)
    gt_px = to_pixels(clip.gt(mid), res)[0]
    trail_px = to_pixels(trail_points(clip.gt, mid, trail_s), res)
    anchor = clip.meta.get("anchor")
    return _draw(img, gt_px, trail_px, anchor[:2] if anchor else None, view_zoom,
                 label, t, view_window_s, view_brightness,
                 _ticks_at(track, clip.meta["t0_device_us"], mid),
                 clip.meta.get("gt_scale", 1.0))


def save(clip, path, fps: float, view_window_s: float, view_brightness: int,
         trail_s: float, view_zoom: float, label: str) -> pathlib.Path:
    import cv2

    track = _motor_track(clip)
    times = frame_times(clip.duration_us / 1e6, fps)
    first = render(clip, times[0], view_window_s, view_brightness, trail_s, view_zoom,
                   label, track)
    h, w = first.shape[:2]

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        writer.write(first)
        for t in times[1:]:
            writer.write(render(clip, t, view_window_s, view_brightness, trail_s,
                                view_zoom, label, track))
    finally:
        writer.release()
    return path


def play(clip, fps: float, view_window_s: float, view_brightness: int, trail_s: float,
         view_zoom: float, label: str) -> None:
    import cv2

    track = _motor_track(clip)
    times = frame_times(clip.duration_us / 1e6, fps)
    i, playing = 0, True
    delay = max(1, int(1000 / fps))

    cv2.namedWindow(_WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            cv2.imshow(_WINDOW, render(clip, times[i], view_window_s, view_brightness,
                                       trail_s, view_zoom, label, track))
            if cv2.getWindowProperty(_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break                           # the human closed the window
            key = cv2.waitKeyEx(delay if playing else 20)
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                playing = not playing
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
            elif key == ord("t"):
                trail_s = 0.0 if trail_s else _TRAIL_S
            elif key == ord("r"):
                i, playing = 0, True
            elif playing:
                i += 1
                if i >= len(times):
                    i, playing = len(times) - 1, False
    finally:
        cv2.destroyAllWindows()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clip", help="a clip directory or .aedat4")
    p.add_argument("--fps", type=float, default=30.0, help="frames rendered per second")
    p.add_argument("--view-window", type=float, default=_VIEW_WINDOW_S, metavar="S",
                   help="display only: seconds of events summed into each shown "
                        "frame. Events are sparse, so some window is needed to see "
                        "anything; a wide one smears a moving target")
    p.add_argument("--view-brightness", type=int, default=_VIEW_BRIGHTNESS, metavar="N",
                   help="display only: event count -> grey level")
    p.add_argument("--view-zoom", type=float, default=2.0, metavar="Z",
                   help="display only: window magnification")
    p.add_argument("--trail", type=float, default=_TRAIL_S, help="trail length (s)")
    p.add_argument("--gt-scale", type=float, default=1.0, metavar="F",
                   help="the one flag that changes the ground truth: divides "
                        "ticks_per_radian by F, widening the predicted sweep F-fold. "
                        "Nothing on disk is written either way")
    p.add_argument("--save", nargs="?", const="", metavar="PATH",
                   help="write a video instead of opening a window "
                        "(default runs/replay/<slug>.mp4)")
    args = p.parse_args()

    clip = load_recording(pathlib.Path(args.clip))
    if clip.gt is None:
        raise SystemExit(f"{clip.meta['slug']}: no ground truth "
                         "(mark an anchor or add hand-labels first)")
    if args.gt_scale != 1.0:
        clip = rescale_gt(clip, args.gt_scale)

    label = clip.meta["slug"]
    if args.save is None:
        play(clip, args.fps, args.view_window, args.view_brightness, args.trail,
             args.view_zoom, label)
        return

    out = (pathlib.Path(args.save) if args.save
           else pathlib.Path("runs/replay") / f"{label}.mp4")
    print("wrote", save(clip, out, args.fps, args.view_window, args.view_brightness,
                        args.trail, args.view_zoom, label))


if __name__ == "__main__":
    main()
