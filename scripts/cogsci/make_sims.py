"""Simulated pendulum clips through the camera model and the frozen localiser, kept as positions.

    python scripts/cogsci/make_sims.py steady_4s_a steady_4s_b steady_4s_c

Each writes runs/cogsci/sims/<name>.npz: `txy` (s, px) as the localiser saw it, `true` (px) at
the same times, `period_s`. The target's looks are drawn from the sim ranges by seed.
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import _thesis_path  # noqa: E402,F401
from make_sim_dataset import random_sim_cfg  # noqa: E402
from trajmem.data import load_sim, sim_params  # noqa: E402
from trajmem.experiment import make_localiser, positions  # noqa: E402
from trajmem.trajectories import TrajectorySpec  # noqa: E402

OUT = ROOT / "runs/cogsci/sims"
LOCALISER = ROOT / "runs/localiser/pend_ft.pt"
WINDOW_US, FPS, ARM_PX = 5000, 650, 380.0

# name: (period s, swing rad, duration s, seed)
SIMS = {"steady_4s_a": (4.0, 0.40, 40.0, 1), "steady_4s_b": (4.0, 0.50, 40.0, 2), "steady_4s_c": (4.0, 0.32, 40.0, 3)}


def make(name: str) -> None:
    period_s, swing, duration_s, seed = SIMS[name]
    base = sim_params()
    base["duration_s"], base["fps"], base["kind"] = duration_s, FPS, "pendulum"
    cfg = random_sim_cfg(np.random.default_rng([2027, seed]), base)
    w, h = base["resolution"]
    spec = TrajectorySpec(shape="pendulum", size=(swing, swing), period_s=period_s, center=(0.5, -0.3),
                          arm=(ARM_PX / w, ARM_PX / h), phase0=0.0)
    started = time.time()
    clip = load_sim(spec, cfg, seed=seed)
    txy = np.array(list(positions(clip, WINDOW_US, make_localiser("snn", LOCALISER))))
    true = np.atleast_2d(clip.gt(txy[:, 0])) * (w, h)
    txy[:, 1:] *= (w, h)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{name}.npz", txy=txy, true=true, period_s=period_s)
    seen = np.isfinite(txy[:, 1])
    err = np.hypot(*(txy[seen, 1:] - true[seen]).T)
    print(f"{name}: {len(clip.events):,} events, unseen {1 - seen.mean():.1%}, localiser error median {np.median(err):.1f} px, "
          f"95% {np.percentile(err, 95):.1f} px, {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    for name in sys.argv[1:] or SIMS:
        make(name)
