"""Visual test bench: every clip in corpus/bench.yaml through three pipelines side by
side -- the classical baseline, the classical localiser feeding the spiking memory, and
the all-spiking chain -- so one iteration of the SNN can be eyeballed against the last.

    python scripts/bench.py --tag pol_e2 --localiser-checkpoint runs/localiser/snn_pol_e2.pt
    python scripts/bench.py --tag pol_e2 --only loop                 # yaml clips whose name contains "loop"
    python scripts/bench.py --live                                   # play instead: n=next clip, q=quit
    python scripts/bench.py --live --clip corpus/real/fan/fan_string_01

Rendering writes runs/bench/<tag>/<clip>.mp4 at true speed (the pipelines are stepped
offline at their real 5 ms window) plus an index.md. Live mode plays the same three
panels in a window at whatever speed the CPU allows, with the speed shown.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import _thesis_path  # noqa: F401,E402  (adds the thesis repo to sys.path)
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from scripts.mark_anchors import accumulate  # noqa: E402
from scripts.replay_gt import (LiveOverlay, _draw, _window_events, frame_times,  # noqa: E402
                               open_clip, to_pixels, trail_points)
from trajmem.metrics import RatchetAlarm  # noqa: E402

BENCH = pathlib.Path("corpus/bench.yaml")
PIPELINES = (("centroid -> Kalman", "kalman", "centroid"),
             ("centroid -> SNN memory", "snn_phasemap", "centroid"),
             ("SNN localiser -> SNN memory", "snn_phasemap", "snn"))
_HEADER_PX = 30
_WINDOW = "bench"


def screen_width(fallback: int = 1600) -> int:
    """Usable window width in pixels, so three panels are not cut off the screen."""
    try:
        import ctypes

        return int(ctypes.windll.user32.GetSystemMetrics(0)) - 16
    except Exception:
        return fallback


def load_bench(path=BENCH) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(yaml.safe_load(fh)["clips"])


def sim_status(clip_name: str, localiser) -> str:
    """"trained on" / "not trained on" for a simulated clip: the checkpoint's own
    validation list, or the shared split over the frame sets on disk for older ones."""
    val = getattr(localiser, "val_clips", None)
    if val is None:
        from trajmem.localise import split_frame_sets

        paths = sorted(pathlib.Path("corpus/sim/frames_8x_5000us").glob("sim_*.npz"))
        val = [p.stem for p in split_frame_sets(paths)[1]]
    return "not trained on" if clip_name in val else "trained on"


def make_overlays(clip, localiser, window_us: int, horizon_s: float, warmup_s: float,
                  pipelines=PIPELINES) -> list[LiveOverlay]:
    from trajmem.experiment import make_memory

    out = []
    for label, memory, loc in pipelines:
        mem = make_memory(memory, dt_s=window_us / 1e6, warmup_s=warmup_s)
        out.append(LiveOverlay(mem, clip, window_us, horizon_s, label, localiser if loc == "snn" else None))
    return out


class Panels:
    """The three pipelines stepped together, one tiled frame per instant."""

    def __init__(self, clip, overlays, header: str, view_window_s: float, view_brightness: int,
                 trail_s: float, zoom: float, max_width: int = 0):
        self.clip, self.overlays, self.header = clip, overlays, header
        self.view_window_s, self.view_brightness, self.trail_s, self.zoom = view_window_s, view_brightness, trail_s, zoom
        self.max_width = max_width
        self.errors = [[] for _ in overlays]
        self.alarms = [RatchetAlarm() for _ in overlays]        # the same rule the scoring uses
        self.flagged = [False] * len(overlays)
        # The pipelines are independent and torch drops the GIL, so stepping them in
        # parallel costs about the slowest one instead of the sum.
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(overlays))

    def frame(self, t: float, note: str = "") -> np.ndarray:
        import cv2

        clip, res = self.clip, self.clip.meta["resolution"]
        mid = t + self.view_window_s / 2
        img = accumulate(_window_events(clip, t, self.view_window_s), 0.0, self.view_window_s, res,
                         gain=self.view_brightness)
        if clip.gt is None:
            gt_px, trail_px = np.array([np.nan, np.nan]), np.empty((0, 2))
        else:
            gt_px = to_pixels(clip.gt(mid), res)[0]
            trail_px = to_pixels(trail_points(clip.gt, mid, self.trail_s), res)
        panels = []
        steps = list(self.pool.map(lambda ov: ov.at(mid), self.overlays))
        for k, (ov, step) in enumerate(zip(self.overlays, steps)):
            if np.isfinite(step["err_px"]):
                self.errors[k].append(step["err_px"])
            self.flagged[k] = self.alarms[k].update(step.get("score", np.nan), mid)
            view = _draw(img, gt_px, trail_px, None, self.zoom, ov.name, t, self.view_window_s,
                         self.view_brightness, None, model={"step": step, "name": ov.memory.__class__.__name__},
                         footer="")
            if self.errors[k]:
                cv2.putText(view, f"median err {np.median(self.errors[k]):5.1f} px", (view.shape[1] - 190, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
            if self.flagged[k]:                                  # the memory says: not the remembered path
                cv2.rectangle(view, (2, 2), (view.shape[1] - 3, view.shape[0] - 3), (0, 0, 255), 4)
                cv2.putText(view, "DEVIATION", (view.shape[1] // 2 - 80, 64), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 2, cv2.LINE_AA)
            bar = f"alarm bar {self.alarms[k].bar:5.1f} px" if np.isfinite(self.alarms[k].bar) else "alarm settling"
            cv2.putText(view, bar, (8, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (120, 200, 255), 1, cv2.LINE_AA)
            panels.append(view)
        tiled = np.hstack(panels)
        header = np.zeros((_HEADER_PX, tiled.shape[1], 3), np.uint8)
        cv2.putText(header, f"{self.header}   t={t:6.2f}s   {note}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        out = np.vstack([header, tiled])
        if self.max_width and out.shape[1] > self.max_width:
            f = self.max_width / out.shape[1]
            out = cv2.resize(out, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        return out


def render_clip(panels: Panels, out: pathlib.Path, fps: float) -> pathlib.Path:
    import cv2

    times = frame_times(panels.clip.duration_us / 1e6, fps)
    first = panels.frame(times[0])
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (first.shape[1], first.shape[0]))
    try:
        writer.write(first)
        for t in times[1:]:
            writer.write(panels.frame(t))
    finally:
        writer.release()
    return out


def play_clip(panels: Panels, fps: float) -> str:
    """Play in a window; returns "next" or "quit". Space pauses, ,/. step, n next, q quit."""
    import cv2

    times = frame_times(panels.clip.duration_us / 1e6, fps)
    i, playing, speed = 0, True, 0.0
    cv2.namedWindow(_WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            t0 = time.perf_counter()
            cv2.imshow(_WINDOW, panels.frame(times[i], f"x{speed:.2f} real time   space=pause ,/.=step n=next q=quit"))
            if cv2.getWindowProperty(_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return "quit"
            spent = (time.perf_counter() - t0) * 1000.0
            key = cv2.waitKeyEx(max(1, int(1000.0 / fps - spent)) if playing else 20)
            speed = min(1.0, (1000.0 / fps) / max(spent, 1e-3)) if playing else 0.0
            if key in (ord("q"), 27):
                return "quit"
            if key == ord("n"):
                return "next"
            if key == ord(" "):
                playing = not playing
            elif key == ord("."):
                playing, i = False, min(i + 1, len(times) - 1)
            elif key == ord(","):
                playing, i = False, max(i - 1, 0)
            elif playing:
                i += 1
                if i >= len(times):
                    return "next"
    finally:
        cv2.destroyWindow(_WINDOW)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", default=str(BENCH), help="the clip list")
    p.add_argument("--only", metavar="TEXT", help="only yaml clips whose name contains TEXT")
    p.add_argument("--clip", metavar="PATH", help="one clip instead of the yaml (any clip)")
    p.add_argument("--live", action="store_true", help="play in a window instead of writing videos")
    p.add_argument("--tag", default="latest", help="output folder name under runs/bench/")
    p.add_argument("--localiser-checkpoint", metavar="PATH",
                   help="spiking localiser weights (default runs/localiser/snn.pt)")
    p.add_argument("--horizon", type=float, default=0.1, help="prediction horizon (s)")
    p.add_argument("--window-us", type=int, default=5000)
    p.add_argument("--warmup", type=float, default=5.0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--view-window", type=float, default=0.02, metavar="S",
                   help="display only: seconds of events per shown frame")
    p.add_argument("--view-brightness", type=int, default=40)
    p.add_argument("--view-zoom", type=float, default=1.0)
    p.add_argument("--trail", type=float, default=2.0)
    p.add_argument("--max-width", type=int, default=0, metavar="PX",
                   help="scale the tiled frame to this width (live: the screen's width by default)")
    p.add_argument("--pipelines", default="all",
                   help="which panels to show: all, or a comma-separated pick of kalman, snn_memory, snn_full "
                        "(fewer panels play faster)")
    args = p.parse_args(argv)

    from trajmem.experiment import make_localiser

    localiser = make_localiser("snn", args.localiser_checkpoint)
    ckpt = pathlib.Path(args.localiser_checkpoint or "runs/localiser/snn.pt").name
    entries = [{"clip": args.clip, "pool": ""}] if args.clip else load_bench(args.bench)
    if args.only:
        entries = [e for e in entries if args.only in str(e["clip"])]
    if not entries:
        raise SystemExit("no clips selected")

    names = {"kalman": PIPELINES[0], "snn_memory": PIPELINES[1], "snn_full": PIPELINES[2]}
    if args.pipelines == "all":
        pipelines = PIPELINES
    else:
        pipelines = tuple(names[n.strip()] for n in args.pipelines.split(","))
    max_width = args.max_width or (screen_width() if args.live else 0)

    out_dir = pathlib.Path("runs/bench") / args.tag
    index = []
    for e in entries:
        clip, name = open_clip(e["clip"])
        pool = e.get("pool", "")
        if pool.startswith("sim"):
            pool = f"{pool}, {sim_status(name, localiser)}"
        overlays = make_overlays(clip, localiser, args.window_us, args.horizon, args.warmup, pipelines)
        header = f"{name}   [{pool}]   localiser {ckpt}"
        panels = Panels(clip, overlays, header, args.view_window, args.view_brightness, args.trail, args.view_zoom,
                        max_width)
        if args.live:
            print(f"playing {name}", flush=True)
            if play_clip(panels, args.fps) == "quit":
                break
            continue
        t0 = time.time()
        path = render_clip(panels, out_dir / f"{name.replace('/', '_')}.mp4", args.fps)
        med = ["n/a" if not errs else f"{np.median(errs):.1f}" for errs in panels.errors]
        index.append(f"| {name} | {pool} | {' | '.join(med)} |")
        print(f"wrote {path}  ({time.time() - t0:.0f} s)  median err px: {', '.join(med)}", flush=True)
    if index:
        head = ("| clip | pool | " + " | ".join(lbl for lbl, _, _ in pipelines) + " |\n"
                + "|---|---" * (1 + len(pipelines)) + "|\n")
        (out_dir / "index.md").write_text(
            f"# bench {args.tag}  (localiser {ckpt}; median prediction error px at +{args.horizon * 1000:.0f} ms, "
            f"labelled clips)\n\n" + head + "\n".join(index) + "\n", encoding="utf-8")
        print("index", out_dir / "index.md")


if __name__ == "__main__":
    main()
