"""Frame datasets for the localiser: per clip, ON/OFF count images per window, block-
downsampled, as uint8, with the true position per window.

    python scripts/make_frames.py --corpus corpus/sim --window-us 5000 --downsample 8

Writes <corpus>/frames_<downsample>x_<window_us>us/<name>.npz with `frames`
(N, 2, H/d, W/d) uint8, `t` (s) and `gt` (N, 2, normalised). A 15 s clip at 5 ms and 8x
is ~29 MB; the window and factor are the localiser's input choices, so pass them.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import _thesis_path  # noqa: F401,E402  (adds the thesis repo to sys.path)
import numpy as np  # noqa: E402

from trajmem.data import load_clip  # noqa: E402
from trajmem.frontend import to_frames  # noqa: E402


def frames_of(clip, window_us: int, downsample: int) -> dict:
    ts, frames = [], []
    for t0, frame in to_frames(clip, window_us, kind="count", downsample=downsample):
        ts.append((t0 + window_us / 2) / 1e6)
        frames.append(np.minimum(frame, 255).astype(np.uint8))
    t = np.array(ts)
    return {"t": t, "frames": np.stack(frames), "gt": np.atleast_2d(clip.gt(t))}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default="corpus/sim")
    p.add_argument("--window-us", type=int, default=5000)
    p.add_argument("--downsample", type=int, default=8)
    args = p.parse_args(argv)

    corpus = pathlib.Path(args.corpus)
    out_dir = corpus / f"frames_{args.downsample}x_{args.window_us}us"
    out_dir.mkdir(exist_ok=True)
    for path in sorted(corpus.glob("sim_*.npz")):
        target = out_dir / path.name
        if target.exists():
            continue
        np.savez_compressed(target, **frames_of(load_clip(path), args.window_us, args.downsample))
        print(path.stem, flush=True)
    print(f"wrote {out_dir}/")


if __name__ == "__main__":
    main()
