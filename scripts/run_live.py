"""The system on the event camera, live: events -> 5 ms count frames -> spiking localiser ->
spiking memory, drawn the way bench.py draws a recorded clip -- the remembered path, the
prediction 100 ms ahead, and the deviation mark.

    python scripts/run_live.py
    python scripts/run_live.py --record runs/live/live_01.aedat4 --log runs/live/live_01.csv
    python scripts/run_live.py --duration 60

The sensor runs at its default settings and the events take a recorded clip's path; the one
difference is that the stuck-pixel cut sees only the past. The localiser runs in its own
process, overlapping the memory -- the configuration scripts/time_live.py found real time.
Run on mains power. When the camera side falls more than 100 ms behind, windows are not
localised until it catches up -- the memory still steps every 5 ms, seeing nothing -- and the
skips are counted on screen and in the log. q quits, r makes the memory forget the path.
--record writes the run as a clip (aedat4 + params side-car) that replays like any other.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import queue
import sys
import time
from collections import deque

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import _thesis_path  # noqa: F401,E402  (adds the thesis repo to sys.path)
import numpy as np  # noqa: E402

WINDOW_US = 5000
SEND_EVERY = 4                                                # windows per message to the memory (20 ms)
MEMORY_THREADS, LOCALISER_THREADS = 1, 2                     # small nets: more threads only fight each other
PATH_EVERY_S = 0.5                                            # redraw the remembered path: 5 ms each, a slow drift
MAX_LAG_S = 0.1                                               # behind by more than the horizon: skip localising
LOCALISER = ROOT / "runs/localiser/pend_ft.pt"
_WINDOW = "live"


def camera_batches(capture, writer=None, stop=None):
    """Event batches from the camera as EVENT_DTYPE arrays, each also written to `writer`
    (with the IMU, as the recorder writes them) when recording."""
    import multiprocessing as mp

    from trajmem.data import to_event_array

    parent = mp.parent_process()                              # a killed main must not leave the camera running
    while capture.isRunning() and not (stop is not None and stop.is_set()):
        if parent is not None and not parent.is_alive():
            return
        if writer is not None:
            imu = capture.getNextImuBatch()
            if imu is not None and len(imu) > 0:
                writer.writeImuPacket(imu)
        events = capture.getNextEventBatch()
        if events is None or events.size() == 0:
            time.sleep(0.001)
            continue
        if writer is not None:
            writer.writeEvents(events)
        yield to_event_array(events.numpy())


class Lag:
    """How far behind the camera: wall time minus camera time, less the smallest that gap
    has been. Never negative, and it rises only when the chain falls behind."""

    def __init__(self, clock=time.perf_counter):
        self.clock, self.best = clock, np.inf

    def __call__(self, t: float) -> float:
        gap = self.clock() - t
        self.best = min(self.best, gap)
        return gap - self.best


def replay_batches(player, stop=None):
    """A recorded clip's batches at the pace the camera delivered them, in place of the camera."""
    from trajmem.data import to_event_array

    start = None
    while player.isRunning() and not (stop is not None and stop.is_set()):
        events = player.getNextEventBatch()
        if events is None or len(events) == 0:
            time.sleep(0.001)
            continue
        ev = to_event_array(events.numpy())
        start = start or (time.perf_counter(), int(ev["timestamp"][0]))
        wait = (int(ev["timestamp"][-1]) - start[1]) / 1e6 - (time.perf_counter() - start[0])
        if wait > 0:
            time.sleep(wait)
        yield ev


def live_positions(batches, localiser, window_us: int, resolution, max_lag_s: float | None = None,
                   clock=time.perf_counter):
    """(t, x, y, window events, skipped) per window: experiment.positions for a stream of
    batches. With `max_lag_s`, a window that arrives more than that behind the camera is not
    localised: its position is NaN, which the memory takes as "not seen" and runs on through."""
    from trajmem.experiment import FRAME_DOWNSAMPLE
    from trajmem.frontend import LiveFrames

    frames = LiveFrames(resolution, window_us, downsample=FRAME_DOWNSAMPLE)
    localiser.reset()
    lag = Lag(clock)
    for batch in batches:
        for t0, ev, frame in frames.push(batch):
            t = (t0 + window_us / 2) / 1e6
            if max_lag_s is not None:
                if lag(t) > max_lag_s:
                    yield t, np.nan, np.nan, ev, True
                    continue
            x, y = localiser.locate(frame)
            yield t, float(x), float(y), ev, False


def _camera_process(cfg: dict, pos_q, view_q, stop) -> None:
    """Child process: camera + localiser. Positions go on `pos_q`; a display image every
    1/fps of camera time goes on `view_q`, dropped when the window is behind."""
    from trajmem.snn import _prefer_performance_cores

    _prefer_performance_cores()
    pos_q.cancel_join_thread()                                # exit at once on stop; the main has stopped reading
    view_q.cancel_join_thread()
    writer = batches = stream = None
    sent = []
    try:
        import dv_processing as dv

        from scripts.mark_anchors import accumulate
        from trajmem.experiment import make_localiser

        import torch

        localiser = make_localiser("snn", cfg["checkpoint"])  # before the camera, so no backlog builds up meanwhile
        torch.set_num_threads(cfg["threads"])
        if cfg["replay"]:
            from recording.player import Player

            capture = Player(cfg["replay"])
            res = tuple(capture.getEventResolution())
            batches = replay_batches(capture, stop)
        else:
            capture = dv.io.camera.open()
            res = tuple(capture.getEventResolution())
            if cfg["record"]:
                pathlib.Path(cfg["record"]).parent.mkdir(parents=True, exist_ok=True)
                writer = dv.io.MonoCameraWriter(str(cfg["record"]), capture)
            batches = camera_batches(capture, writer, stop)
        pos_q.put(("ready", res, writer is not None))
        shown = deque(maxlen=max(1, round(cfg["view_window_s"] * 1e6 / WINDOW_US)))
        next_view = 0.0
        stream = live_positions(batches, localiser, WINDOW_US, res, MAX_LAG_S)
        for t, x, y, ev, skipped in stream:
            sent.append((t, x, y, skipped))
            if len(sent) == SEND_EVERY:                       # a few windows per message: less queue overhead
                pos_q.put(sent)
                sent = []
            shown.append(ev)
            if t >= next_view:
                start_s = t - WINDOW_US / 2e6 - (len(shown) - 1) * WINDOW_US / 1e6
                img = accumulate(np.concatenate(shown), start_s, cfg["view_window_s"], res, gain=cfg["brightness"])
                try:
                    view_q.put_nowait((t, img))
                except queue.Full:
                    pass
                next_view = t + 1.0 / cfg["fps"]
    except Exception as exc:                                  # the main process reports it
        pos_q.put(("error", repr(exc)))
    finally:
        if sent:
            pos_q.put(sent)
        for gen in (stream, batches):                         # a paused generator would keep the writer alive
            if gen is not None:
                gen.close()
        del writer                                            # the writer finalises the aedat4 when destroyed
        pos_q.put(None)


class LiveView:
    """The memory and its alarm, stepped per window, drawn as bench.py draws a panel."""

    def __init__(self, resolution, horizon_s: float = 0.1, warmup_s: float = 5.0, k: float | None = None,
                 window_us: int = WINDOW_US, name: str = "SNN localiser -> SNN memory"):
        from trajmem.experiment import make_memory
        from trajmem.metrics import k_for_input

        self.resolution, self.horizon_s, self.warmup_s, self.window_us, self.name = (
            resolution, horizon_s, warmup_s, window_us, name)
        self.k = k_for_input("snn", LOCALISER) if k is None else k
        self.memory = make_memory("snn_phasemap", dt_s=window_us / 1e6, warmup_s=warmup_s)
        self.reset()

    def reset(self) -> None:
        """Forget the path: a fresh memory state and alarm, both warming up again from now."""
        from scripts.replay_gt import LiveOverlay
        from trajmem.data import Clip
        from trajmem.metrics import RatchetAlarm
        from trajmem.simulate import EVENT_DTYPE

        no_truth = Clip(events=np.empty(0, EVENT_DTYPE), duration_us=0, gt=None,
                        meta={"resolution": self.resolution})
        self.overlay = LiveOverlay(self.memory, no_truth, self.window_us, self.horizon_s, self.name, windows=())
        self.overlay.cycle_every_s = PATH_EVERY_S
        self.alarm = RatchetAlarm(k=self.k, warmup_s=self.warmup_s, dt_s=self.window_us / 1e6)
        self.flagged, self.t, self.t_origin = False, np.nan, None

    @property
    def learning(self) -> bool:
        """Still inside the warm-up since the start or the last reset."""
        return self.t_origin is None or self.t - self.t_origin < self.warmup_s

    @property
    def step_out(self) -> dict:
        return self.overlay.step

    def step(self, t: float, x: float, y: float) -> dict:
        if self.t_origin is None:
            self.t_origin = t - self.window_us / 2e6
        self.t = t
        step = self.overlay.observe(t, x, y)
        self.overlay.stepped.clear()                          # only bench's `at` reads it; live it would only grow
        self.flagged = self.alarm.update(step["score"], t - self.t_origin)
        return step

    def frame(self, img, lag_ms: float, view_window_s: float = 0.02, brightness: int = 40,
              zoom: float = 1.0, skipped: int = 0) -> np.ndarray:
        import cv2

        from scripts.bench import draw_alarm
        from scripts.replay_gt import _draw

        t = 0.0 if self.t_origin is None else self.t - self.t_origin
        model = None if self.learning else {"step": self.overlay.step, "name": self.memory.__class__.__name__}
        view = _draw(img, None, np.empty((0, 2)), None, zoom, f"{self.name}   q=quit r=re-learn", t,
                     view_window_s, brightness, None, model=model, footer="")
        if self.learning:                                     # what the memory says before warm-up is not meant to be read
            cv2.putText(view, f"learning the path... {self.warmup_s - t:3.1f} s", (8, view.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1, cv2.LINE_AA)
        else:
            draw_alarm(view, self.alarm, self.flagged)
        cv2.putText(view, f"behind real time {lag_ms:5.0f} ms", (view.shape[1] - 230, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
        if skipped:
            cv2.putText(view, f"localiser skipped {skipped}", (view.shape[1] - 230, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return view


def _write_sidecar(path) -> None:
    """The recorder's params side-car, saying no flicker detection ran."""
    from config.runtime import load_runtime_config
    from recording.recorder import write_params_sidecar

    params = load_runtime_config()
    params.flicker_autodetect_enabled = False
    write_params_sidecar(pathlib.Path(path), params)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--localiser-checkpoint", default=str(LOCALISER), metavar="PATH")
    p.add_argument("--record", metavar="AEDAT4", help="also write the run as a clip")
    p.add_argument("--replay", metavar="AEDAT4", help="a recorded clip at camera pace instead of the camera")
    p.add_argument("--log", metavar="CSV", help="per-window t, position, prediction, score, flag")
    p.add_argument("--duration", type=float, metavar="S", help="stop after this many seconds")
    p.add_argument("--horizon", type=float, default=0.1, help="prediction horizon (s)")
    p.add_argument("--warmup", type=float, default=5.0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--view-window", type=float, default=0.02, metavar="S",
                   help="display only: seconds of events per shown frame")
    p.add_argument("--view-brightness", type=int, default=40)
    p.add_argument("--view-zoom", type=float, default=1.0)
    args = p.parse_args(argv)
    if args.record and args.replay:
        p.error("--record writes the camera; a replay is already a recording")

    import multiprocessing as mp

    import cv2

    from trajmem.metrics import k_for_input
    from trajmem.snn import _prefer_performance_cores

    _prefer_performance_cores()
    pos_q, view_q, stop = mp.Queue(), mp.Queue(maxsize=2), mp.Event()
    cfg = {"checkpoint": args.localiser_checkpoint, "record": args.record, "replay": args.replay, "threads": LOCALISER_THREADS,
           "fps": args.fps,
           "view_window_s": args.view_window, "brightness": args.view_brightness}
    proc = mp.Process(target=_camera_process, args=(cfg, pos_q, view_q, stop), daemon=True)
    proc.start()
    first = pos_q.get(timeout=60)
    if first is None or first[0] == "error":
        proc.join(5)
        raise SystemExit(f"camera process failed: {first}")
    res, recording = first[1], first[2]

    view = LiveView(res, args.horizon, args.warmup, k_for_input("snn", args.localiser_checkpoint))
    import torch

    torch.set_num_threads(MEMORY_THREADS)
    log_fh = open(args.log, "w", newline="", encoding="utf-8") if args.log else None
    log = csv.writer(log_fh) if log_fh else None
    if log:
        log.writerow(["t_s", "x", "y", "pred_x", "pred_y", "score", "flagged", "skipped"])
    print(f"live on {res[0]}x{res[1]}, localiser {pathlib.Path(args.localiser_checkpoint).name}"
          + (f", recording to {args.record}" if args.record else ""), flush=True)

    lag, lag_ms, n, n_skipped, error = Lag(), 0.0, 0, 0, None
    cv2.namedWindow(_WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        running = True
        while running:
            redraw_at = time.perf_counter() + 1.0 / args.fps   # the window answers keys even when behind
            while running and time.perf_counter() < redraw_at:
                try:
                    item = pos_q.get(timeout=max(1e-3, redraw_at - time.perf_counter()))
                except queue.Empty:
                    break
                if item is None or item[0] == "error":
                    error, running = (item[1] if item else None), False
                    break
                for t, x, y, skipped in item:
                    step = view.step(t, x, y)
                    n += 1
                    n_skipped += skipped
                    if log:
                        px, py = step["pred_px"] / np.array(res, dtype=float)
                        log.writerow([f"{t:.4f}", x, y, px, py, step["score"], int(view.flagged), int(skipped)])
                    if args.duration is not None and t >= args.duration:
                        running = False
                        break
                lag_ms = 1000.0 * lag(t)
            img = None
            while True:
                try:
                    _, img = view_q.get_nowait()
                except queue.Empty:
                    break
            if img is not None:
                cv2.imshow(_WINDOW, view.frame(img, lag_ms, args.view_window, args.view_brightness, args.view_zoom,
                                               n_skipped))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                running = False
            elif key == ord("r"):
                print(f"re-learning from t={view.t:.2f} s", flush=True)
                view.reset()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        proc.join(10)
        if proc.is_alive():
            proc.terminate()
        cv2.destroyAllWindows()
        if log_fh:
            log_fh.close()
    if error:
        print(f"camera process stopped: {error}", flush=True)
    if recording and pathlib.Path(args.record).exists():
        _write_sidecar(args.record)
        print(f"recorded -> {args.record}", flush=True)
    if n:
        print(f"{n} windows ({n * WINDOW_US / 1e6:.1f} s of camera time), ended {lag_ms:.0f} ms behind real time, "
              f"localiser skipped {n_skipped} ({100 * n_skipped / n:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
