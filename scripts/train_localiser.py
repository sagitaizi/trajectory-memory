"""Train the spiking localiser on the simulated frame sets.

    python scripts/train_localiser.py                       # -> runs/localiser/snn.pt
    python scripts/train_localiser.py --epochs 30 --batch 8 --out runs/localiser/snn_e30.pt

Frame sets come from scripts/make_frames.py (<corpus>/frames_8x_5000us/sim_*.npz); a
tenth of the clips validate. Augmentation (stuck pixels, noise floor, flips, blank
stretches) is applied to the training clips only. The best epoch by validation loss is
saved with a per-epoch log next to it.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import _thesis_path  # noqa: F401,E402  (adds the thesis repo to sys.path)
import numpy as np  # noqa: E402

from trajmem.localise import load_frame_set  # noqa: E402
from trajmem.snn_localise import SpikingLocaliser  # noqa: E402


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default="corpus/sim")
    p.add_argument("--out", default="runs/localiser/snn.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--chunk-s", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--n-cells", type=int, default=32)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--ch", type=int, nargs=2, default=(8, 16), metavar=("C1", "C2"), help="conv channels")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    d = pathlib.Path(args.corpus) / "frames_8x_5000us"
    paths = sorted(d.glob("sim_*.npz"))
    if not paths:
        raise SystemExit(f"no frame sets in {d}; run scripts/make_frames.py first")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(paths))
    n_val = max(1, round(len(paths) * args.val_fraction))
    val = [load_frame_set(paths[i], 5000, 8) for i in order[:n_val]]
    train = [paths[i] for i in order[n_val:]]                     # loaded per batch
    print(f"{len(paths)} frame sets: {len(train)} train, {len(val)} validation", flush=True)

    loc = SpikingLocaliser(n_cells=args.n_cells, ch=tuple(args.ch), hidden=args.hidden, seed=args.seed, device=args.device)
    out = pathlib.Path(args.out)
    t0 = time.time()

    def show(e):
        print(f"epoch {e['epoch']:3d}  train {e['train_loss']:6.3f}  val {e['val_loss']:6.3f} "
              f"(pos {e['val_pos']:5.3f} present {e['val_present']:5.3f})  val px {e['val_px']:6.1f}  "
              f"present acc {e['val_present_acc']:.3f}  {time.time() - t0:5.0f} s", flush=True)
        loc.save(out.with_suffix(".partial.pt"))

    log = loc.fit(train, val_sets=val, epochs=args.epochs, chunk_s=args.chunk_s, batch=args.batch, lr=args.lr,
                  augment=not args.no_augment, seed=args.seed, log_fn=show)
    loc.save(out)
    out.with_suffix(".partial.pt").unlink(missing_ok=True)
    best = min((e for e in log if np.isfinite(e["val_loss"])), key=lambda e: e["val_loss"])
    print(f"best epoch {best['epoch']}: val px {best['val_px']:.1f}, present acc {best['val_present_acc']:.3f}; wrote {out}")
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)


if __name__ == "__main__":
    main()
