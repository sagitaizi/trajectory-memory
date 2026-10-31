"""Energy per 5 ms step, from operation counts: the spiking pipeline, the same networks run
conventionally, and the classical pipeline (centroid + periodic Kalman).

    python scripts/energy.py          # reads runs/paper/spike_counts.csv -> runs/paper/energy.csv

Per-operation energies are 45 nm, 32-bit float (Horowitz, ISSCC 2014): an accumulate
(one synaptic operation) 0.9 pJ, a multiply-accumulate 4.6 pJ. Arithmetic only, no memory
access. Every LIF update and every non-spiking step is charged as a multiply-accumulate.
"""
from __future__ import annotations

import csv
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
IN = ROOT / "runs/paper/spike_counts.csv"
OUT = ROOT / "runs/paper/energy.csv"
E_AC, E_MAC = 0.9e-12, 4.6e-12
KALMAN_N, COORDS, HORIZONS = 7, 2, 4
CENTROID_OPS_PER_EVENT = 6                    # cell index (2), bincount, keep test, two sums
CENTROID_CELLS = 40 * 30


def kalman_macs() -> int:
    n, c = KALMAN_N, COORDS
    step = n * n * c + 2 * n ** 3 + n * c + n * n + n + n + n * c + n * n + n * n + n * n
    return step + HORIZONS * (n * n * c + n * c)


def main() -> None:
    d = pd.read_csv(IN)
    rows = []
    for run, g in d.groupby("run", sort=False):
        def pick(part, text, col):
            return g[(g.part == part) & g["item"].str.contains(text, regex=False)][col].sum()
        events = pick("localiser", "input (events)", "spikes_per_step")
        lif_syn = pick("localiser", "LIF", "synops_per_step")
        loc = g[g.part == "localiser"]
        dense = loc[~loc["item"].str.contains("active pixels")].dense_ops_per_step.sum()
        updates = g[g.part == "neuron updates"].set_index("item").dense_ops_per_step
        loc_updates, mem_updates = updates.iloc[0], updates.iloc[1]
        mem_syn = pick("memory", "factored", "synops_per_step") + pick("memory", "ring", "synops_per_step")
        mem_dense = g[g.part == "memory"].dense_ops_per_step.sum()
        mem_other = g[g.part == "memory, non-spiking"].dense_ops_per_step.sum()
        for input_as in ("events", "active pixels"):
            syn = pick("localiser", f"input ({input_as}", "synops_per_step") + lif_syn
            spiking_loc = syn * E_AC + loc_updates * E_MAC
            spiking_mem = mem_syn * E_AC + (mem_updates + mem_other) * E_MAC
            classical = events * CENTROID_OPS_PER_EVENT * E_AC + CENTROID_CELLS * E_AC + kalman_macs() * E_MAC
            rows.append({
                "run": run, "input_counted_as": input_as,
                "localiser_spiking_uJ": spiking_loc * 1e6,
                "localiser_conventional_uJ": (dense + loc_updates) * E_MAC * 1e6,
                "memory_spiking_uJ": spiking_mem * 1e6,
                "memory_conventional_uJ": (mem_dense + mem_other) * E_MAC * 1e6,
                "classical_pipeline_uJ": classical * 1e6})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"kalman MACs per step {kalman_macs()}; wrote {OUT}")


if __name__ == "__main__":
    main()
