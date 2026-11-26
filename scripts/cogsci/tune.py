"""The adjusted settings (D6): the map's learning-rate schedule, then the alarm's k and floor.

    python scripts/cogsci/tune.py map      # -> runs/cogsci/tune_map.csv
    python scripts/cogsci/tune.py alarm    # -> runs/cogsci/tune_alarm.csv, with the map settings chosen

Map: the eye's gain and lag on the simulated 4 s swings against the human values, relative to
the target as the localiser reports it (its own lag is measured and taken out). Alarm: no
false alarm on the steady development runs and the same swings, over several neuron builds.
"""
import csv
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_eye import DT, ROOT, RUNS, real_run  # noqa: E402
from pursuit.clock import ClockMemory  # noqa: E402
from pursuit.loop import follow  # noqa: E402
from pursuit.measures import alarms, gain_lag  # noqa: E402
from trajmem.data import read_deviation_times  # noqa: E402

OUT = ROOT / "runs/cogsci"
SIMS = ("steady_4s_a", "steady_4s_b", "steady_4s_c")
HUMAN_GAIN, HUMAN_GAIN_SD, LAG_UNIT_S = 1.03, 0.09, 0.02     # Brostek et al. 2017; lag zero, counted in 20 ms
SETTLED_S = 20.0
GRID = {"eta_start": (0.1, 0.3, 0.6), "eta": (0.01, 0.02, 0.05), "eta_tau_s": (1.0, 2.0, 4.0)}
PUBLISHED = {"eta_start": 0.3, "eta": 0.02, "eta_tau_s": 2.0}
K_GRID, FLOOR_GRID = (3, 4, 6, 8, 10, 12, 16, 20, 25, 30), (0.0, 50.0, 100.0)
SEEDS = (1, 2, 3)


def sim(name):
    z = np.load(OUT / "sims" / f"{name}.npz")
    return z["txy"], np.gradient(z["true"], DT, axis=0), float(z["period_s"])


def steady_input(name):
    """Positions of a steady run: a simulated swing, or a development run up to its break."""
    if name in SIMS:
        return sim(name)[0]
    runs = {clip: (positions, start) for clip, positions, start in RUNS + [("wide_break", "wide_break", 0.0)]}
    positions, start = runs[name]
    txy = real_run(name, positions, start)[0]
    breaks = read_deviation_times(ROOT / "corpus/real/pendulum" / name)
    return txy[txy[:, 0] < min(breaks) - start - 0.5] if breaks else txy


def map_job(job):
    settings, name = job
    txy, true, period_s = sim(name)
    tr = follow(ClockMemory(DT, **settings), txy, DT)
    ok = tr.t > SETTLED_S
    (gain, lag), (m_gain, m_lag), (l_gain, l_lag) = (
        gain_lag(tr.t[ok], v[ok], true[ok], period_s) for v in (tr.eye_velocity, tr.memory_velocity, tr.target_velocity))
    gain, lag, m_gain, m_lag = gain / l_gain, lag - l_lag, m_gain / l_gain, m_lag - l_lag
    return {**settings, "clip": name, "gain": gain, "lag_ms": lag * 1e3, "memory_gain": m_gain, "memory_lag_ms": m_lag * 1e3,
            "localiser_gain": l_gain, "localiser_lag_ms": l_lag * 1e3,
            "distance": abs(gain - HUMAN_GAIN) / HUMAN_GAIN_SD + abs(lag) / LAG_UNIT_S}


def alarm_job(job):
    settings, name, seed = job
    txy = steady_input(name)
    tr = follow(ClockMemory(DT, seed=seed, **settings), txy, DT)
    t = tr.t - tr.t[0]
    speed = np.sqrt(np.nanmean(np.sum(tr.memory_velocity[t > 10] ** 2, 1)))
    return [{"clip": name, "seed": seed, "k": k, "floor": f, "false_alarms": len(alarms(t, tr.surprise, DT, k, f)),
             "score_median": float(np.nanmedian(tr.surprise)), "score_max": float(np.nanmax(tr.surprise)), "speed": float(speed)}
            for k in K_GRID for f in FLOOR_GRID]


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def mean_distance(rows):
    """[(distance, settings)] over the clips, closest to people first."""
    key = lambda r: tuple(float(r[k]) for k in GRID)  # noqa: E731
    by = {}
    for r in rows:
        by.setdefault(key(r), []).append(float(r["distance"]))
    return sorted(((float(np.mean(d)), dict(zip(GRID, k))) for k, d in by.items()), key=lambda x: x[0])


if __name__ == "__main__":
    with ProcessPoolExecutor(7) as pool:
        if sys.argv[1] == "map":
            grid = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
            rows = []
            for r in pool.map(map_job, [(g, n) for g in grid for n in SIMS]):
                rows.append(r)
                print(" ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()), flush=True)
            write(OUT / "tune_map.csv", rows)
            for d, settings in mean_distance(rows):
                print(f"{d:.3f} {settings}", "<- published" if settings == PUBLISHED else "", flush=True)
        else:
            settings = {k: float(v) for k, v in (a.split("=") for a in sys.argv[2:])} or PUBLISHED
            jobs = [(settings, n, s) for n in SIMS + ("small_01", "wide_02", "wide_break") for s in SEEDS]
            rows = [r for part in pool.map(alarm_job, jobs) for r in part]
            write(OUT / "tune_alarm.csv", rows)
            for name in dict.fromkeys(r["clip"] for r in rows):
                quiet = [k for k in K_GRID if not any(r["false_alarms"] for r in rows if (r["clip"], r["k"], r["floor"]) == (name, k, 0.0))]
                print(f"{name}: smallest k with no false alarm on any build: {min(quiet, default=None)}", flush=True)
