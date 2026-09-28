"""How fast the full pipeline runs: spiking localiser + spiking memory, per 5 ms frame.

    python scripts/time_live.py corpus/real/pendulum/small_03 corpus/real/pendulum/wide_01
    python scripts/time_live.py --set held_out

Two ways per clip: `sequential` (localise, then remember, one frame at a time) and
`overlapped` (the localiser in its own process, already on frame t+1 while the memory
takes frame t). Both give identical outputs; overlapped is the live configuration.
Time on mains power: battery saving slows the CPU by a third.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
WINDOW_US = 5000


def _setup() -> None:
    sys.path.insert(0, str(ROOT))
    import _thesis_path  # noqa: F401  (adds the thesis repo to sys.path)
    from trajmem.snn import _prefer_performance_cores
    _prefer_performance_cores()


def _open(entry):
    """The clip, sliced; unlabelled clips too, since timing needs no labels."""
    from trajmem.data import load_recording, slice_clip
    clip = load_recording(ROOT / entry["clip"])
    if entry.get("slice") is not None:
        t0, t1 = entry["slice"]
        clip = slice_clip(clip, t0 or 0.0, clip.duration_us / 1e6 if t1 is None else t1)
    return clip


def _frames(clip):
    from trajmem.experiment import FRAME_DOWNSAMPLE
    from trajmem.frontend import to_frames
    return (f for _, f in to_frames(clip, WINDOW_US, kind="count", downsample=FRAME_DOWNSAMPLE))


def _step(mem, xy) -> None:
    mem.observe(*xy)
    mem.predict(0.1)
    mem.deviation_score()


def _localise_into(entry, q, go) -> None:
    _setup()
    from trajmem.experiment import make_localiser
    clip = _open(entry)
    loc = make_localiser("snn")
    loc.reset()
    q.put("ready")
    go.wait()
    for f in _frames(clip):
        q.put(loc.locate(f))
    q.put(None)


def overlapped(entry) -> tuple[int, float]:
    from trajmem.experiment import make_memory
    q, go = mp.Queue(maxsize=64), mp.Event()
    p = mp.Process(target=_localise_into, args=(entry, q, go))
    p.start()
    mem = make_memory("snn_phasemap", dt_s=WINDOW_US / 1e6)
    mem.reset()
    if q.get() != "ready":
        raise RuntimeError("localiser process failed to start")
    go.set()
    t0, n = time.perf_counter(), 0
    while (xy := q.get()) is not None:
        _step(mem, xy)
        n += 1
    wall = time.perf_counter() - t0
    p.join()
    return n, wall


def sequential(entry) -> tuple[int, float]:
    from trajmem.experiment import make_localiser, make_memory
    clip = _open(entry)
    loc = make_localiser("snn")
    mem = make_memory("snn_phasemap", dt_s=WINDOW_US / 1e6)
    loc.reset()
    mem.reset()
    t0, n = time.perf_counter(), 0
    for f in _frames(clip):
        _step(mem, loc.locate(f))
        n += 1
    return n, time.perf_counter() - t0


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clips", nargs="*", help="clip directories")
    p.add_argument("--set", help="a set from corpus/sets.yaml instead of clip paths")
    args = p.parse_args(argv)
    _setup()
    if args.set:
        from trajmem.experiment import load_set
        entries = load_set(args.set)
    else:
        entries = [{"clip": c, "slice": None, "name": pathlib.Path(c).name} for c in args.clips]
    if not entries:
        p.error("give clip paths or --set")
    for entry in entries:
        for name, fn in (("sequential", sequential), ("overlapped", overlapped)):
            n, wall = fn(entry)
            span = n * WINDOW_US / 1e6
            print(f"{entry['name']:24s} {name:10s} {wall / n * 1e3:5.2f} ms/frame  x{span / wall:.2f} real time "
                  f"({span:.1f} s of events in {wall:.1f} s)")


if __name__ == "__main__":
    main()
