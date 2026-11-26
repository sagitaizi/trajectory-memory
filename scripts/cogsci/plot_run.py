"""Pictures of the model on development pendulum runs: the eye with the clock memory against the
reaction alone, at the start, through a gap with the target hidden, and at a break in the path.

    python scripts/cogsci/plot_run.py      # -> runs/cogsci/plots/{gap,break}.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_eye import DT, ROOT, real_run  # noqa: E402
from pursuit.clock import ClockMemory  # noqa: E402
from pursuit.loop import follow  # noqa: E402
from pursuit.measures import alarms  # noqa: E402
from trajmem.data import read_deviation_times  # noqa: E402

OUT = ROOT / "runs/cogsci/plots"
INK, MUTED, GRID, BAND = "#2b2b29", "#8a8984", "#e4e3de", "#efeee9"
CLOCK, REACTION = "#2a78d6", "#eb6834"
GAP_S = (15.0, 17.4)

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def runs(clip, positions, start_s, blank=None):
    txy, velocity, period_s = real_run(clip, positions, start_s)
    t0 = txy[0, 0]
    gap = None if blank is None else (t0 + blank[0], t0 + blank[1])
    with_clock = follow(ClockMemory(DT), txy, DT, blank=gap)
    alone = follow(None, txy, DT, blank=gap)
    t = with_clock.t - t0
    return t, np.atleast_2d(velocity(with_clock.t))[:, 0], with_clock, alone, period_s, t0


def speeds(ax, t, target, with_clock, alone, window, hidden=None):
    ok = (t >= window[0]) & (t <= window[1])
    if hidden is not None:
        ax.axvspan(*hidden, color=BAND, lw=0, zorder=0)
        ax.text(np.mean(hidden), 0.97, "target hidden", transform=ax.get_xaxis_transform(), ha="center",
                va="top", color=MUTED)
        seen = ok & ~((t >= hidden[0]) & (t < hidden[1]))
        ax.plot(t[ok], np.where(seen[ok], np.nan, target[ok]), color=INK, lw=1.0, ls=(0, (2, 2)))
        ax.plot(t[ok], np.where(seen[ok], target[ok], np.nan), color=INK, lw=1.2, label="target")
    else:
        ax.plot(t[ok], target[ok], color=INK, lw=1.2, label="target")
    ax.plot(t[ok], alone.eye_velocity[ok, 0], color=REACTION, lw=1.5, label="eye, reaction alone")
    ax.plot(t[ok], with_clock.eye_velocity[ok, 0], color=CLOCK, lw=1.5, label="eye, with the clock memory")
    ax.set_xlim(*window)
    ax.set_ylabel("horizontal speed (px/s)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.axhline(0, color=GRID, lw=0.8)


def gap_figure():
    t, target, with_clock, alone, period_s, _ = runs("small_01", "small_01", 0.0, blank=GAP_S)
    fig, (a, b) = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
    speeds(a, t, target, with_clock, alone, (0, 6))
    a.set_title("Start of the motion", loc="left", color=INK)
    speeds(b, t, target, with_clock, alone, (12, 21), hidden=GAP_S)
    b.set_title(f"Target hidden for two cycles (period {period_s:.2f} s): the memory carries the eye on and fades",
                loc="left", color=INK)
    b.set_xlabel("time from the start of the run (s)")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=3, frameon=False)
    return fig


def break_figure():
    t, target, with_clock, alone, period_s, t0 = runs("wide_break", "wide_break", 0.0)
    pushed = min(read_deviation_times(ROOT / "corpus/real/pendulum/wide_break")) - t0
    raised = alarms(t, with_clock.surprise, DT)
    fig, (a, b) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
    speeds(a, t, target, with_clock, alone, (8, 20))
    a.set_title("The pendulum is pushed off its swing", loc="left", color=INK)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=3, frameon=False)
    b.plot(t, with_clock.surprise, color=CLOCK, lw=1.5)
    b.set_ylabel("surprise (px/s)")
    b.grid(axis="y", color=GRID, lw=0.8)
    for x in raised:
        b.axvline(x, color=INK, lw=1.0)
    first = raised[0] - pushed if raised else None
    when = "no alarm" if first is None else f"first {abs(first):.2f} s {'before' if first < 0 else 'after'} the marked push"
    b.set_title(f"Surprise, with alarms (solid lines): {when}", loc="left", color=INK)
    b.set_xlabel("time from the start of the run (s)")
    for ax in (a, b):
        ax.axvline(pushed, color=MUTED, lw=1.0, ls=(0, (3, 2)))
    a.text(pushed, 0.97, " push", transform=a.get_xaxis_transform(), va="top", color=MUTED)
    return fig


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, make in (("gap", gap_figure), ("break", break_figure)):
        make().savefig(OUT / f"{name}.png", dpi=150)
        print(OUT / f"{name}.png", flush=True)
