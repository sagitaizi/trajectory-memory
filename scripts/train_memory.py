"""Pretrain the spiking trajectory memory on the simulated tracks and save a checkpoint.

    python scripts/train_memory.py                                   # -> runs/memory/snn.pt
    python scripts/train_memory.py --epochs 60 --out runs/memory/snn_long.pt
    python scripts/train_memory.py --augment flips,shift:2,scale:2,stretch:2 --include-locked

Reads <corpus>/tracks.npz + manifest.csv (scripts/make_tracks.py). Clips whose measured
track sits further than --max-obs-err-px (median) from the truth are left out: the
memory learns from positions, and on those clips the centroid is on the string, not
the target (PLAN.md section A). --include-locked keeps them with the true position
(plus 3 px of noise) standing in for the centroid. --augment applies trajmem.augment to
the training clips only. The per-epoch log goes next to the checkpoint as .csv.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time
from dataclasses import replace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import _thesis_path  # noqa: F401,E402  (adds the thesis repo to sys.path)
import numpy as np  # noqa: E402

from trajmem.augment import augment  # noqa: E402
from trajmem.data import load_tracks  # noqa: E402
from trajmem.snn import SpikingMemory  # noqa: E402


def usable(tracks, max_obs_err_px: float, resolution):
    keep = []
    for tr in tracks:
        err = np.hypot(*((tr.obs - tr.gt) * resolution).T)
        if np.nanmedian(err) <= max_obs_err_px:
            keep.append(tr)
    return keep


def with_truth_standing_in(tracks, seed: int = 1, noise_px: float = 3.0):
    rng = np.random.default_rng(seed)
    return [replace(tr, obs=tr.gt + rng.normal(0, noise_px / 640, tr.gt.shape)) for tr in tracks]


def split(tracks, val_fraction: float, seed: int):
    """The same held-back clips `SpikingMemory.fit` would pick on its own."""
    order = np.random.default_rng(seed).permutation(len(tracks))
    n_val = max(1, round(len(tracks) * val_fraction)) if len(tracks) > 1 else 0
    return [tracks[i] for i in order[n_val:]], [tracks[i] for i in order[:n_val]]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default="corpus/sim")
    p.add_argument("--out", default="runs/memory/snn.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--chunk-s", type=float, default=5.0, help="BPTT chunk (s); longer than the longest cycle")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--path-weight", type=float, default=1.0, help="weight of the path-head loss")
    p.add_argument("--path-loss", default="geometric", choices=("geometric", "mse"),
                   help="px of the drawn cycle + phase + period (geometric), or MSE on the raw numbers")
    p.add_argument("--anchor-tau-s", type=float, default=0.0, help="smoothing of the anchor position (s)")
    p.add_argument("--heads-from", default="fast", choices=("fast", "both", "split"),
                   help="which layer the horizon heads read; split = 25/50 ms fast, 100/200 ms slow")
    p.add_argument("--arch", default="two_layer", choices=("two_layer", "lmu"),
                   help="learned slow layer (two_layer) or the fixed LMU window memory (lmu)")
    p.add_argument("--blank-prob", type=float, default=0.5, help="lmu: fraction of chunks with a hidden stretch")
    p.add_argument("--lmu-q", type=int, default=24, help="lmu: Legendre order per axis")
    p.add_argument("--lmu-n", type=int, default=200, help="lmu: neurons per state dimension")
    p.add_argument("--lmu-theta", type=float, default=4.0, help="lmu: window (s)")
    p.add_argument("--path-from", default="lmu", choices=("lmu", "both", "state"), help="lmu: what the path head reads")
    p.add_argument("--path-readout", default="linear", choices=("linear", "mlp"), help="lmu: path head form")
    p.add_argument("--path-pretrain-steps", type=int, default=0,
                   help="lmu: fit the path head offline on the recorded window for this many steps, then freeze it")
    p.add_argument("--schedule", default="none", choices=("none", "cosine"), help="learning-rate schedule")
    p.add_argument("--max-obs-err-px", type=float, default=15.0)
    p.add_argument("--include-locked", action="store_true", help="string-locked clips too, truth standing in")
    p.add_argument("--augment", default="", help='e.g. "flips,shift:2,scale:2,stretch:2" (training clips only)')
    p.add_argument("--n-per-axis", type=int, default=32)
    p.add_argument("--n-fast", type=int, default=384)
    p.add_argument("--n-slow", type=int, default=128)
    p.add_argument("--device", default=None, help="cpu (default) or cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    tracks = load_tracks(args.corpus)
    dt_s = float(np.median(np.diff(tracks[0].t)))
    kept = usable(tracks, args.max_obs_err_px, (640, 480))
    train, val = split(kept, args.val_fraction, args.seed)
    if args.include_locked:
        used = {tr.name for tr in kept}
        train += with_truth_standing_in([tr for tr in tracks if tr.name not in used])
    train = augment(train, args.augment, args.seed)
    print(f"{len(tracks)} tracks, {len(kept)} with the centroid within {args.max_obs_err_px} px of the "
          f"truth; {len(train)} training after --include-locked/--augment, {len(val)} validation; "
          f"dt {dt_s * 1e3:.1f} ms; {len(tracks[0].t)} steps each")
    if args.arch == "lmu":
        from trajmem.snn_lmu import LmuMemory

        memory = LmuMemory(dt_s=dt_s, device=args.device, n_per_axis=args.n_per_axis, n_fast=args.n_fast,
                           anchor_tau_s=args.anchor_tau_s, q=args.lmu_q, n_per_dim=args.lmu_n,
                           theta_s=args.lmu_theta, path_from=args.path_from, path_readout=args.path_readout,
                           seed=args.seed)
        memory.set_radii_from(train)
        fit_extra = {"blank_prob": args.blank_prob, "path_pretrain_steps": args.path_pretrain_steps}
    else:
        memory = SpikingMemory(dt_s=dt_s, device=args.device, n_per_axis=args.n_per_axis,
                               n_fast=args.n_fast, n_slow=args.n_slow, anchor_tau_s=args.anchor_tau_s,
                               heads_from=args.heads_from, seed=args.seed)
        fit_extra = {}

    t0 = time.time()

    out = pathlib.Path(args.out)

    def show(e):
        print(f"epoch {e['epoch']:3d}  train {e['train_loss']:7.3f}  val {e['val_loss']:7.3f} "
              f"(heads {e['val_heads']:6.3f} path {e['val_path']:6.3f})  val 100 ms {e['val_px']:6.1f} px  "
              f"path {e['val_path_px']:5.1f} px  "
              f"rates {e['rate_fast']:.2f}/{e['rate_slow']:.2f}  {time.time() - t0:5.0f} s", flush=True)
        memory.save(out.with_suffix(".partial.pt"))        # latest weights, should the run be cut short

    log = memory.fit(train, val_tracks=val, epochs=args.epochs, chunk_s=args.chunk_s, batch=args.batch,
                     lr=args.lr, path_weight=args.path_weight, seed=args.seed, schedule=args.schedule,
                     path_loss=args.path_loss, log_fn=show, **fit_extra)
    if getattr(memory, "path_pretrain_val_px", None) is not None:
        print(f"path head pretrained offline: validation path {memory.path_pretrain_val_px:.1f} px")
    out = memory.save(out)
    out.with_suffix(".partial.pt").unlink(missing_ok=True)
    best = min((e for e in log if np.isfinite(e["val_loss"])), key=lambda e: e["val_loss"])
    print(f"best epoch {best['epoch']} (val loss {best['val_loss']:.3f}): val 100 ms {best['val_px']:.1f} px, "
          f"path {best['val_path_px']:.1f} px; deviation scales "
          f"{memory.scales[0]:.1f} / {memory.scales[1]:.1f} px; wrote {out}")
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)


if __name__ == "__main__":
    main()
