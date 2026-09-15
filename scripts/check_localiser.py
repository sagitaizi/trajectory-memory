"""How well the classical centroid (frontend.to_position) matches the hand-labels.

    python scripts/check_localiser.py --set development
    python scripts/check_localiser.py corpus/real/fan/fan_brush_slow_02 --window-us 10000

Per clip: median and 90th-percentile distance to the labels in px, and the median
*signed* offset, which tells a labelling-convention gap (a constant offset) from
tracking noise. This is the floor under every prediction error scored against labels.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import _thesis_path  # noqa: F401,E402  (adds the thesis repo to sys.path)
import numpy as np  # noqa: E402

from scripts.run_experiment import open_clip  # noqa: E402
from trajmem.experiment import open_set  # noqa: E402
from trajmem.frontend import to_position  # noqa: E402


def localiser_error(clip, window_us: int) -> dict:
    w, h = clip.meta["resolution"]
    track = np.array(list(to_position(clip, window_us)))
    seen = np.isfinite(track[:, 1])
    gt = np.atleast_2d(clip.gt(track[seen, 0]))
    ok = np.isfinite(gt[:, 0])
    d = (track[seen][ok, 1:] - gt[ok]) * (w, h)
    err = np.hypot(*d.T)
    return {"median": float(np.median(err)), "p90": float(np.percentile(err, 90)),
            "dx": float(np.median(d[:, 0])), "dy": float(np.median(d[:, 1])),
            "unseen": float(1 - seen.mean()), "n": int(ok.sum())}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clip", nargs="?")
    p.add_argument("--set", metavar="NAME")
    p.add_argument("--window-us", type=int, default=5000)
    args = p.parse_args(argv)

    clips = open_set(args.set) if args.set else [(args.clip, open_clip(args.clip))]
    print(f"{'clip':26} {'median':>7} {'p90':>7} {'dx':>7} {'dy':>7} {'unseen':>7}   (px, window {args.window_us} us)")
    for name, clip in clips:
        r = localiser_error(clip, args.window_us)
        print(f"{name:26} {r['median']:7.1f} {r['p90']:7.1f} {r['dx']:+7.1f} {r['dy']:+7.1f} {r['unseen']:7.1%}")


if __name__ == "__main__":
    main()
