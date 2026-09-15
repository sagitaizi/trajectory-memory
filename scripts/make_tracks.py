"""Position tracks for every simulated clip: what the frontend measured and the truth,
one row per window, into one file the memory core can pretrain from.

    python scripts/make_tracks.py --corpus corpus/sim --window-us 5000

Writes <corpus>/tracks.npz with, per clip `name`: `<name>/t` (window centres, s),
`<name>/obs` (measured, NaN = unseen), `<name>/gt` (true position at t), and
`<name>/deviation_t` (the break, or NaN). Positions are normalised.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import _thesis_path  # noqa: F401,E402  (adds the thesis repo to sys.path)
import numpy as np  # noqa: E402

from trajmem.data import load_clip  # noqa: E402
from trajmem.frontend import to_position  # noqa: E402


def track_of(clip, window_us: int) -> dict:
    rows = np.array(list(to_position(clip, window_us)), dtype=float)
    t = rows[:, 0]
    return {"t": t, "obs": rows[:, 1:3], "gt": np.atleast_2d(clip.gt(t)),
            "deviation_t": float(min(clip.deviation_times)) if clip.deviation_times else np.nan}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default="corpus/sim")
    p.add_argument("--window-us", type=int, default=5000)
    args = p.parse_args(argv)

    corpus = pathlib.Path(args.corpus)
    out = {}
    for path in sorted(corpus.glob("sim_*.npz")):
        for key, value in track_of(load_clip(path), args.window_us).items():
            out[f"{path.stem}/{key}"] = value
        print(path.stem, flush=True)
    np.savez_compressed(corpus / "tracks.npz", window_us=args.window_us, **out)
    print(f"wrote {corpus / 'tracks.npz'}: {len(out) // 4} clips")


if __name__ == "__main__":
    main()
