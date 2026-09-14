"""Batch-generate simulated pretraining clips: random repetitive paths through the
matched camera model, each with exact ground truth.

    python scripts/make_sim_dataset.py --n 100 --duration 15 --out corpus/sim

Every clip draws its own path (shape, size, period, position, angle; half get one
scripted deviation in the middle third) and its own camera settings from the
`sim.randomise` ranges in params.yaml. Clip i depends only on (--seed, i), so a run
that is killed resumes by skipping the files already written. Watch a clip with
`python scripts/replay_gt.py corpus/sim/sim_000.npz`.
"""
from __future__ import annotations

import argparse
import copy
import csv
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import _thesis_path  # noqa: F401,E402  (adds the thesis repo to sys.path)
import numpy as np  # noqa: E402

from trajmem.data import load_clip, load_sim, save_clip, sim_params  # noqa: E402
from trajmem.trajectories import Deviation, TrajectorySpec, sample  # noqa: E402

MANIFEST_FIELDS = ("name", "shape", "period_s", "deviation", "deviation_t", "seed",
                   "pos_thres", "sigma_thres", "shot_noise_rate_hz", "fg_intensity", "radius_px")
_SWITCH_TARGETS = ("circle", "ellipse", "figure8", "lissajous")


def random_spec(rng: np.random.Generator, path_cfg: dict, duration_s: float) -> TrajectorySpec:
    """A random repetitive path that stays inside the frame for the whole clip.

    "sweep" is an ellipse with a zero minor axis: a straight line at any angle, the
    pendulum and wall-target motions. Specs that leave the frame are redrawn.
    """
    lo, hi = path_cfg["semi_axis"]
    margin = path_cfg["margin"]
    for _ in range(200):
        shape = rng.choice(path_cfg["shapes"])
        a = rng.uniform(lo, hi)
        b = 0.0 if shape == "sweep" else rng.uniform(lo, hi)
        if shape == "circle":
            b = a
        spec = TrajectorySpec(
            shape="ellipse" if shape == "sweep" else str(shape),
            size=(float(a), float(b)),
            period_s=float(rng.uniform(*path_cfg["period_s"])),
            center=(float(rng.uniform(margin, 1 - margin)), float(rng.uniform(margin, 1 - margin))),
            phase0=float(rng.uniform(0, 2 * np.pi)),
            rotation=float(rng.uniform(0, np.pi)),
            deviations=_random_deviations(rng, path_cfg, duration_s, shape),
        )
        pos = sample(spec, np.linspace(0, duration_s, int(200 * duration_s) + 1))
        if pos.min() >= margin and pos.max() <= 1 - margin:
            return spec
    raise RuntimeError("could not draw a path that stays inside the frame")


def _random_deviations(rng, path_cfg, duration_s, shape) -> list[Deviation]:
    if rng.uniform() >= path_cfg["deviation_fraction"]:
        return []
    at_t = float(rng.uniform(duration_s / 3, 2 * duration_s / 3))
    kind = rng.choice(["shrink", "speed_change", "drift", "switch_shape"])
    if kind == "shrink":
        params = {"factor": float(rng.uniform(0.4, 0.7))}
    elif kind == "speed_change":
        params = {"factor": float(rng.choice([rng.uniform(0.5, 0.8), rng.uniform(1.25, 2.0)]))}
    elif kind == "drift":
        angle, speed = rng.uniform(0, 2 * np.pi), rng.uniform(0.01, 0.04)
        params = {"vel": (float(speed * np.cos(angle)), float(speed * np.sin(angle)))}
    else:
        to = rng.choice([s for s in _SWITCH_TARGETS if s != shape])
        params = {"to": str(to)}
    return [Deviation(at_t=at_t, kind=str(kind), params=params)]


def random_sim_cfg(rng: np.random.Generator, sim_cfg: dict) -> dict:
    """A copy of `sim_cfg` with the `randomise` ranges drawn and dropped."""
    cfg = copy.deepcopy(sim_cfg)
    ranges = cfg.pop("randomise")
    for key in ("pos_thres", "sigma_thres", "shot_noise_rate_hz"):
        cfg["v2e"][key] = float(rng.uniform(*ranges[key]))
    cfg["v2e"]["neg_thres"] = cfg["v2e"]["pos_thres"]
    cfg["blob"]["fg_intensity"] = float(rng.uniform(*ranges["fg_intensity"]))
    cfg["blob"]["radius_px"] = int(rng.integers(ranges["radius_px"][0], ranges["radius_px"][1] + 1))
    return cfg


def manifest_row(name: str, clip) -> dict:
    spec, cfg = clip.meta["spec"], clip.meta["sim_cfg"]
    dev = spec.deviations[0] if spec.deviations else None
    return {
        "name": name,
        "shape": "sweep" if spec.size[1] == 0 else spec.shape,
        "period_s": f"{spec.period_s:.4f}",
        "deviation": dev.kind if dev else "",
        "deviation_t": f"{dev.at_t:.3f}" if dev else "",
        "seed": clip.meta["seed"],
        "pos_thres": f"{cfg['v2e']['pos_thres']:.4f}",
        "sigma_thres": f"{cfg['v2e']['sigma_thres']:.4f}",
        "shot_noise_rate_hz": f"{cfg['v2e']['shot_noise_rate_hz']:.4f}",
        "fg_intensity": f"{cfg['blob']['fg_intensity']:.1f}",
        "radius_px": cfg["blob"]["radius_px"],
    }


def _write_manifest(out: pathlib.Path, rows: dict[str, dict]) -> None:
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows[k] for k in sorted(rows))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="corpus/sim", help="output directory")
    parser.add_argument("--n", type=int, default=100, help="number of clips")
    parser.add_argument("--duration", type=float, default=15.0, help="clip length (s)")
    parser.add_argument("--fps", type=int, default=650, help="blob render rate; keep >= 21 x cutoff_hz")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-distortion", action="store_true", help="skip the lens model")
    args = parser.parse_args(argv)

    base = sim_params(args.params)
    base["duration_s"], base["fps"] = args.duration, args.fps
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = {}
    for i in range(args.n):
        name = f"sim_{i:03d}"
        path = out / f"{name}.npz"
        if path.exists():
            rows[name] = manifest_row(name, load_clip(path))
            continue
        rng = np.random.default_rng([args.seed, i])
        cfg = random_sim_cfg(rng, base)
        spec = random_spec(rng, base["randomise"]["path"], args.duration)
        started = time.time()
        clip = load_sim(spec, cfg, seed=args.seed * 10_000 + i, distort=not args.no_distortion)
        clip.meta["sim_cfg"] = cfg
        save_clip(clip, path)
        rows[name] = manifest_row(name, clip)
        _write_manifest(out, rows)
        print(f"{name}  {rows[name]['shape']:9} T={spec.period_s:.2f}s  "
              f"dev={rows[name]['deviation'] or '-':12} {len(clip.events):>9,} ev  "
              f"{time.time() - started:5.0f} s", flush=True)
    _write_manifest(out, rows)
    print(f"{len(rows)} clips in {out}/")


if __name__ == "__main__":
    main()
