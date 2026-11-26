"""Replay a development pendulum run with the simulated eye drawn on the events: the reaction
alone beside the eye with the clock memory, its remembered path and its deviation alarm.

    python scripts/cogsci/replay.py small_01 --live                  # window: space, ,/. step, q quit
    python scripts/cogsci/replay.py small_01 --blank 15 17.4         # hide the target then (s from start)
    python scripts/cogsci/replay.py wide_break                       # -> runs/cogsci/replay/wide_break.mp4
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_eye import DT, ROOT, RUNS, real_run  # noqa: E402
from pursuit.clock import ClockMemory  # noqa: E402
from pursuit.loop import follow  # noqa: E402
from pursuit.measures import ALARM_K  # noqa: E402
from scripts.bench import play_clip, render_clip  # noqa: E402
from scripts.mark_anchors import accumulate  # noqa: E402
from scripts.replay_gt import _window_events, open_clip, to_pixels, trail_points  # noqa: E402
from trajmem.metrics import RatchetAlarm  # noqa: E402

CLIPS = {clip: (positions, start) for clip, positions, start in RUNS + [("wide_break", "wide_break", 0.0)]}
OUT = ROOT / "runs/cogsci/replay"
BLUE, ORANGE, PATH, TARGET, GREY = (214, 120, 42), (52, 104, 235), (0, 200, 255), (0, 220, 0), (170, 170, 170)
VIEW_S, BRIGHTNESS, TRAIL_S, EYE_TRAIL_S, PATH_POINTS = 0.02, 40, 1.0, 0.4, 60


class Drawn(ClockMemory):
    """The clock memory, also keeping the path it expects over the next cycle at chosen steps."""

    def __init__(self, dt_s, keep_every, **kw):
        super().__init__(dt_s, **kw)
        self.keep_every = keep_every

    def reset(self):
        super().reset()
        self.n, self.paths = 0, {}

    def observe(self, position, velocity):
        super().observe(position, velocity)
        self.n += 1
        lap = self.period()
        if self.n % self.keep_every or not (np.isfinite(lap) and np.isfinite(position).all()
                                            and np.isfinite(self.velocity(0.0)).all()):
            return
        da = lap / PATH_POINTS
        v = np.array([self.velocity(a) for a in np.arange(PATH_POINTS) * da])
        self.paths[self.n - 1] = position + np.vstack([np.zeros(2), np.cumsum(v * da, axis=0)])


class Replay:
    """What `bench.play_clip` / `bench.render_clip` need: `clip` and `frame(t, note)`."""

    def __init__(self, name, blank, fps):
        positions, self.start_s = CLIPS[name]
        self.clip, _ = open_clip(ROOT / "corpus/real/pendulum" / name)
        txy = real_run(name, positions, self.start_s)[0]
        self.t0, self.name = txy[0, 0], name
        self.blank = None if blank is None else (self.t0 + blank[0], self.t0 + blank[1])
        self.memory = Drawn(DT, keep_every=max(1, int(round(1 / (fps * DT)))))
        self.mine = follow(self.memory, txy, DT, blank=self.blank)
        self.alone = follow(None, txy, DT, blank=self.blank)
        rule = RatchetAlarm(k=ALARM_K, floor_px=0.0, dt_s=DT)
        self.flagged = np.array([rule.update(s, t - self.t0) for s, t in zip(self.mine.surprise, self.mine.t)])
        self.path_steps = np.array(sorted(self.memory.paths))

    def frame(self, t_clip, note=""):
        import cv2

        res = self.clip.meta["resolution"]
        mid = t_clip + VIEW_S / 2
        t = mid - self.start_s
        img = accumulate(_window_events(self.clip, t_clip, VIEW_S), 0.0, VIEW_S, res, gain=BRIGHTNESS)
        hidden = self.blank is not None and self.blank[0] <= t < self.blank[1]
        base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if hidden:
            base = (base * 0.35).astype(np.uint8)
        if self.clip.gt is not None:
            for k, (x, y) in enumerate(to_pixels(trail_points(self.clip.gt, mid, TRAIL_S), res)):
                if np.isfinite(x):
                    cv2.circle(base, (int(x), int(y)), 1, TARGET, -1)
        i = int(np.clip(np.searchsorted(self.mine.t, t), 0, len(self.mine.t) - 1))
        inside = self.mine.t[0] <= t <= self.mine.t[-1]
        views = []
        for label, tr, colour in (("reaction alone", self.alone, ORANGE), ("with the clock memory", self.mine, BLUE)):
            view = base.copy()
            if inside:
                if tr is self.mine:
                    self._memory(view, i)
                trail = tr.gaze[max(0, i - int(EYE_TRAIL_S / DT)):i + 1]
                cv2.polylines(view, [np.round(trail).astype(np.int32).reshape(-1, 1, 2)], False, colour, 1, cv2.LINE_AA)
                cv2.circle(view, tuple(int(v) for v in np.round(tr.gaze[i])), 7, colour, 2, cv2.LINE_AA)
            cv2.putText(view, f"eye, {label}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
            if hidden:
                cv2.putText(view, "TARGET HIDDEN", (view.shape[1] // 2 - 85, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREY, 2,
                            cv2.LINE_AA)
            views.append(view)
        tiled = np.hstack(views)
        header = np.zeros((30, tiled.shape[1], 3), np.uint8)
        cv2.putText(header, f"{self.name}   t={t - self.t0:6.2f}s   green: target   {note}", (8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([header, tiled])

    def _memory(self, view, i):
        import cv2

        k = self.path_steps[np.searchsorted(self.path_steps, i, side="right") - 1] if len(self.path_steps) and \
            self.path_steps[0] <= i else None
        if k is not None:
            pts = np.round(self.memory.paths[k]).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(view, [pts], False, PATH, 1, cv2.LINE_AA)
        period, surprise = self.mine.period[i], self.mine.surprise[i]
        hud = ("choosing a clock" if not np.isfinite(period) else f"period {period:.2f} s") + \
              ("   surprise: not yet" if not np.isfinite(surprise) else f"   surprise {surprise:5.0f} px/s")
        cv2.putText(view, hud, (8, view.shape[0] - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, PATH, 1, cv2.LINE_AA)
        cv2.putText(view, "orange-yellow line: the path the memory expects next", (8, view.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, PATH, 1, cv2.LINE_AA)
        if self.flagged[i]:
            cv2.rectangle(view, (2, 2), (view.shape[1] - 3, view.shape[0] - 3), (0, 0, 255), 4)
            cv2.putText(view, "DEVIATION", (view.shape[1] // 2 - 80, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
                        cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", choices=sorted(CLIPS))
    p.add_argument("--blank", nargs=2, type=float, metavar=("FROM", "TO"), help="hide the target, s from the start")
    p.add_argument("--live", action="store_true", help="play in a window instead of writing a video")
    p.add_argument("--fps", type=float, default=30.0)
    a = p.parse_args()
    replay = Replay(a.run, a.blank, a.fps)
    if a.live:
        play_clip(replay, a.fps)
    else:
        tag = "" if a.blank is None else f"__hidden_{a.blank[0]:g}-{a.blank[1]:g}"
        print(render_clip(replay, OUT / f"{a.run}{tag}.mp4", a.fps), flush=True)


if __name__ == "__main__":
    main()
