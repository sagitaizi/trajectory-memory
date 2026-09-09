"""CLI over experiment.run."""
from __future__ import annotations

import argparse

import _thesis_path  # noqa: F401  (adds the thesis repo to sys.path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--sim", default="corpus/sim", help="pretraining clips")
    parser.add_argument("--real", default="corpus/real", help="held-out evaluation clips")
    parser.add_argument("--out", default="runs", help="results directory")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
