"""Batch-generate simulated pretraining clips from trajectory specs."""
from __future__ import annotations

import argparse

import _thesis_path  # noqa: F401  (adds the thesis repo to sys.path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="corpus/sim", help="output directory")
    parser.add_argument("--n", type=int, default=20, help="number of clips")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
