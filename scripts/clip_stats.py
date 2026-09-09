"""Sanity stats for a recorded aedat4 (direct dv_processing read, no recording.player).

    python scripts/clip_stats.py corpus/real/fan_brush_slow_01.aedat4
"""
import sys

import dv_processing as dv


def main() -> int:
    path = sys.argv[1]
    rec = dv.io.MonoCameraRecording(path)
    streams = [s for s, ok in (
        ("events", rec.isEventStreamAvailable()),
        ("frames", rec.isFrameStreamAvailable()),
        ("imu", rec.isImuStreamAvailable()),
    ) if ok]
    print(f"camera     : {rec.getCameraName()}")
    print(f"streams    : {', '.join(streams)}")
    print(f"resolution : {rec.getEventResolution()}")

    n = 0
    t0 = t1 = None
    while rec.isRunning():
        batch = rec.getNextEventBatch()
        if batch is None:
            continue
        n += len(batch)
        if t0 is None:
            t0 = batch.getLowestTime()
        t1 = batch.getHighestTime()

    span = (t1 - t0) / 1e6 if t0 is not None else 0.0
    print(f"events     : {n:,}")
    print(f"duration   : {span:.3f} s")
    print(f"mean rate  : {n / span / 1e3:,.0f} k ev/s" if span else "mean rate  : n/a")
    ok = n > 0 and span > 1.0
    print("OK" if ok else "SUSPECT — empty or very short")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
