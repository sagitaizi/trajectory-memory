"""Batch-generate simulated pretraining clips: random repetitive paths through the
matched camera model, each with exact ground truth.

    python scripts/make_sim_dataset.py --n 100 --duration 15 --out corpus/sim

Every clip draws its own path (shape, size, period, position, angle; half get one
scripted deviation in the middle third; most get small smooth imperfections, some
stay perfect), its own target (aspect, angle, texture, an optional string to a pivot)
and its own camera settings (thresholds, noise, an optional photoreceptor lag), all
from the `sim.randomise` ranges in params.yaml. Clip i depends only on (--seed, i), so a run
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
from trajmem.trajectories import Deviation, TrajectorySpec, Wobble, sample  # noqa: E402

MANIFEST_FIELDS = ("name", "shape", "period_s", "deviation", "deviation_t", "wobble", "seed",
                   "pos_thres", "sigma_thres", "shot_noise_rate_hz", "cutoff_hz", "fg_intensity",
                   "radius_px", "aspect", "string", "texture_depth")
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
            wobble=_random_wobble(rng, path_cfg["wobble"]),
        )
        t = np.linspace(0, duration_s, int(200 * duration_s) + 1)
        pos = sample(spec, t)
        if pos.min() >= margin and pos.max() <= 1 - margin and _deviation_shows(spec, t, pos):
            return spec
    raise RuntimeError("could not draw a path that stays inside the frame")


def _deviation_shows(spec, t, pos, min_shift=0.01) -> bool:
    """A break must move the path: a circle switched to an ellipse is the same path."""
    if not spec.deviations:
        return True
    plain = sample(TrajectorySpec(**{**spec.__dict__, "deviations": []}), t)
    after = t >= spec.deviations[0].at_t
    return bool(np.abs(pos[after] - plain[after]).max() > min_shift)


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


def _random_wobble(rng, cfg) -> Wobble | None:
    """None for a perfect path; otherwise every term drawn on its own, so the spread
    runs from barely imperfect to clearly wobbly."""
    if rng.uniform() < cfg["perfect_fraction"]:
        return None
    u = lambda key: float(rng.uniform(*cfg[key]))  # noqa: E731
    return Wobble(
        amp_depth=u("amp_depth"), amp_period_s=u("amp_period_s"),
        drift=(u("drift"), u("drift")), drift_period_s=u("drift_period_s"),
        phase_depth=u("phase_depth"), phase_period_s=u("phase_period_s"),
        growth=u("growth"),
    )


def random_sim_cfg(rng: np.random.Generator, sim_cfg: dict) -> dict:
    """A copy of `sim_cfg` with the `randomise` ranges drawn and dropped."""
    cfg = copy.deepcopy(sim_cfg)
    ranges = cfg.pop("randomise")
    for key in ("pos_thres", "sigma_thres", "shot_noise_rate_hz"):
        cfg["v2e"][key] = float(rng.uniform(*ranges[key]))
    cfg["v2e"]["neg_thres"] = cfg["v2e"]["pos_thres"]
    cfg["v2e"]["cutoff_hz"] = (float(rng.uniform(*ranges["cutoff_hz"]))
                               if rng.uniform() < ranges["lag_fraction"] else 0.0)
    blob = cfg["blob"]
    blob["fg_intensity"] = float(rng.uniform(*ranges["fg_intensity"]))
    blob["radius_px"] = int(rng.integers(ranges["radius_px"][0], ranges["radius_px"][1] + 1))
    blob["aspect"] = float(rng.uniform(*ranges["aspect"]))
    blob["angle"] = float(rng.uniform(0, np.pi))
    blob["texture"] = {"depth": float(rng.uniform(*ranges["texture_depth"])),
                       "scale_px": float(rng.uniform(*ranges["texture_scale_px"])),
                       "seed": int(rng.integers(0, 2**31))}
    blob["string"] = None
    if sim_cfg.get("kind") == "sheet":                       # the wall-target setup: outlines only
        blob["kind"] = "sheet"
        blob["target_px"] = [int(rng.integers(50, 200)), int(rng.integers(25, 100))]
        blob["sheet_px"] = [int(rng.integers(280, 560)), int(rng.integers(180, 400))]
        blob["sheet_offset"] = [float(rng.uniform(0.15, 0.85)), float(rng.uniform(0.15, 0.85))]
        blob["thickness_px"] = int(rng.integers(2, 5))
        blob["sheet_contrast"] = float(rng.uniform(0.2, 0.5))
        blob["fg_intensity"] = float(rng.uniform(60, 130))
        blob["angle"] = float(rng.uniform(-0.15, 0.15))
        blob["texture"] = {"depth": 0.0, "scale_px": 4.0, "seed": 0}
        return cfg
    if rng.uniform() < ranges["string_fraction"]:
        lo, hi = ranges["string_thickness_px"]
        blob["string"] = {                       # the pivot sits above the frame
            "pivot": [float(rng.uniform(0.1, 0.9)), float(rng.uniform(-0.6, -0.05))],
            "thickness_px": int(rng.integers(lo, hi + 1)),
            "intensity": float(rng.uniform(*ranges["string_intensity"])),
        }
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
        "wobble": "" if spec.wobble is None else "yes",
        "seed": clip.meta["seed"],
        "pos_thres": f"{cfg['v2e']['pos_thres']:.4f}",
        "sigma_thres": f"{cfg['v2e']['sigma_thres']:.4f}",
        "shot_noise_rate_hz": f"{cfg['v2e']['shot_noise_rate_hz']:.4f}",
        "cutoff_hz": f"{cfg['v2e']['cutoff_hz']:.0f}",
        "fg_intensity": f"{cfg['blob']['fg_intensity']:.1f}",
        "radius_px": cfg["blob"]["radius_px"],
        "aspect": f"{cfg['blob']['aspect']:.2f}",
        "string": "yes" if cfg["blob"].get("string") else ("sheet" if cfg["blob"].get("kind") == "sheet" else ""),
        "texture_depth": f"{cfg['blob']['texture']['depth']:.2f}",
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
    parser.add_argument("--start", type=int, default=0, help="first clip index (continue a corpus)")
    parser.add_argument("--kind", default="blob", choices=("blob", "sheet"),
                        help="blob (the fan / pendulum targets) or sheet (the wall target on its sheet)")
    args = parser.parse_args(argv)

    base = sim_params(args.params)
    base["duration_s"], base["fps"] = args.duration, args.fps
    if args.kind == "sheet":
        base["kind"] = "sheet"
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = {}
    if (out / "manifest.csv").exists():                        # keep the rows of clips made earlier
        with open(out / "manifest.csv", encoding="utf-8") as fh:
            rows = {r["name"]: r for r in csv.DictReader(fh)}
    for i in range(args.start, args.start + args.n):
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
