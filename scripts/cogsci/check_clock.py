"""Does the clock memory lock on, and is the velocity it learns clean enough? The development
pendulum runs, and clean sinusoids for the lags of the chain."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_eye import DT, RES, ROOT, RUNS, real_run, sinusoid  # noqa: E402
from pursuit.clock import ClockMemory  # noqa: E402
from pursuit.loop import follow  # noqa: E402
from pursuit.measures import gain_lag  # noqa: E402
from trajmem.data import read_deviation_times  # noqa: E402

SETTLE_S = 10.0


def report(name, txy, velocity, period_s, memory, until=np.inf):
    tr = follow(memory, txy, DT)
    since = tr.t - tr.t[0]
    ok = (since > SETTLE_S) & (tr.t < until)
    true = np.atleast_2d(velocity(tr.t))
    ratio = tr.period / period_s
    has = np.flatnonzero(np.isfinite(tr.surprise))
    slow_at = since[has[0]] if len(has) else np.nan
    first = since[np.isfinite(ratio)][0] if np.isfinite(ratio).any() else np.nan
    near = np.abs(ratio - 1) > 0.1
    near[~np.isfinite(ratio)] = True
    locked = since[near & (tr.t < until)].max() if (near & (tr.t < until)).any() else 0.0
    cells = [f"{name:10s} {period_s:5.2f} {first:5.1f} {locked:6.1f} {np.nanmedian(ratio[ok]):6.3f} {slow_at:6.1f}"]
    for v in (tr.memory_velocity, tr.eye_velocity):
        g, lag = gain_lag(tr.t[ok], v[ok], true[ok], period_s)
        rms = np.sqrt(np.mean(np.sum((v[ok] - true[ok]) ** 2, 1)) / np.mean(np.sum(true[ok] ** 2, 1)))
        cells.append(f"{g:5.2f} {lag * 1e3:6.0f} {rms:5.2f}")
    s = tr.surprise[(since > max(SETTLE_S, slow_at + 1.0)) & (tr.t < until)]
    speed = np.sqrt(np.mean(np.sum(true[ok] ** 2, 1)))
    cells.append(f"{np.median(s) / speed:5.2f} {s.max() / speed:5.2f}" if len(s) else "    -     -")
    print(" | ".join(cells), flush=True)


if __name__ == "__main__":
    print(f"{'run':10s} {'T s':>5s} {'first':>5s} {'locked':>6s} {'T/true':>6s} {'slow@':>6s} | memory: gain, lag ms, err/v | eye: gain, lag ms, err/v | surprise/speed: median, max")
    m = ClockMemory(DT)
    for clip, positions, start in RUNS + [("wide_break", "wide_break", 0.0)]:
        txy, velocity, period = real_run(clip, positions, start)
        breaks = read_deviation_times(ROOT / "corpus/real/pendulum" / clip)
        until = min(breaks) - start - 0.5 if breaks else np.inf
        if breaks:
            from check_eye import search_period
            before = txy[txy[:, 0] < until]
            period = search_period(before[::4, 0], before[::4, 1:])
        report(clip, txy, velocity, period, m, until)
    for T in (0.8, 1.2, 2.0, 3.0, 4.0):
        report("sinusoid", *sinusoid(T, max(20.0, 12 * T)), m)
