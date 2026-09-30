"""Spikes and synaptic operations per 5 ms step of the full spiking pipeline (the `pend_ft`
localiser and the spiking clock-and-map), on the pendulum development runs.

    python scripts/spike_counts.py            # -> runs/paper/spike_counts.csv

A synaptic operation (SynOp) is one accumulate at one synapse a spike reaches. Conv
fan-out is exact per position (kernel, stride, padding, output channels). The memory's
Nengo populations are counted two ways: factored (a spike reaches the D decoded
dimensions; each neuron then encodes D values per LIF substep) and as the full
neuron-to-neuron weight matrix a neuromorphic chip would store. The run is checked
against the paper's cached positions and traces: counting must not change any output.
"""
from __future__ import annotations

import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import _thesis_path  # noqa: F401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as fn  # noqa: E402

from trajmem.experiment import evaluate_clip, load_set, make_localiser, make_memory, open_one, positions  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEV = ROOT / "runs/paper/development"
OUT = ROOT / "runs/paper/spike_counts.csv"
LOCALISER = ROOT / "runs/localiser/pend_ft.pt"
RUNS = ("small_01", "wide_02/steady", "wide_break")
HORIZONS = (0.025, 0.05, 0.1, 0.2)
WINDOW_US = 5000
STEADY_S = 6.0


def conv_fanout(conv: torch.nn.Conv2d, in_hw) -> torch.Tensor:
    """Synapses each input position of `conv` reaches: output channels x output positions."""
    x = torch.ones(1, 1, *in_hw, requires_grad=True)
    k = torch.ones(1, 1, *conv.kernel_size)
    fn.conv2d(x, k, stride=conv.stride, padding=conv.padding).sum().backward()
    return x.grad[0, 0] * conv.out_channels


class CountingLocaliser:
    """Wraps the localiser: records the input and each LIF layer's spikes, per frame."""

    def __init__(self, localiser):
        self.loc, net = localiser, localiser.net
        c1, c2 = net.hw
        self.fan_in = conv_fanout(net.conv1, localiser.in_hw)
        self.fan_1 = conv_fanout(net.conv2, c1[1:])
        self.rows, self._spk = [], {}
        for name in ("lif1", "lif2", "lif3"):
            getattr(net, name).register_forward_hook(self._hook(name))

    def _hook(self, name):
        def hook(_mod, _inp, out):
            self._spk[name] = out[0].detach()[0]
        return hook

    def reset(self):
        self.loc.reset()

    def locate(self, frame):
        xy = self.loc.locate(frame)
        f = torch.as_tensor(np.asarray(frame, dtype=np.float32))
        s1, s2, s3 = self._spk["lif1"], self._spk["lif2"], self._spk["lif3"]
        net = self.loc.net
        self.rows.append({
            "in_events": float(f.sum()), "in_pixels": float((f > 0).sum()),
            "in_synops_event": float((f * self.fan_in).sum()), "in_synops_pixel": float(((f > 0) * self.fan_in).sum()),
            "s1": float(s1.sum()), "s2": float(s2.sum()), "s3": float(s3.sum()),
            "syn1": float((s1 * self.fan_1).sum()), "syn2": float(s2.sum()) * net.fc.out_features,
            "syn3": float(s3.sum()) * net.read.out_features})
        return xy


def count_memory(mem):
    """Counts the LIF spikes of the clock and ring populations, per `observe` step."""
    counts, rows = {"clock": 0.0, "ring": 0.0}, []
    for key, pop in (("clock", mem.clock), ("ring", mem.ring)):
        lif = pop._lif

        def counted(j, v, r, lif=lif, key=key):
            spk, v, r = lif(j, v, r)
            counts[key] += float((spk > 0).sum())
            return spk, v, r
        pop._lif = counted
    observe = mem.observe

    def counted_observe(x, y):
        before = dict(counts)
        observe(x, y)
        rows.append({"clock": counts["clock"] - before["clock"], "ring": counts["ring"] - before["ring"],
                     "snapshot": float(mem.snapshot is not None)})
    mem.observe = counted_observe
    return rows


def localiser_dense_macs(net) -> dict:
    (c1, h1, w1), (c2, h2, w2) = net.hw
    return {"conv1": c1 * h1 * w1 * net.conv1.in_channels * 25, "conv2": c2 * h2 * w2 * c1 * 25,
            "fc": net.fc.in_features * net.fc.out_features, "read": net.read.in_features * net.read.out_features}


def memory_conventional_ops(mem, n_horizons: int) -> dict:
    """Approximate per-step op counts of the memory's non-spiking steps (after election)."""
    B, N, D = mem.B, mem.n_ring, 2
    read_one = N * D + 4 * N + D * N          # ring tuning curves at a phase (encode, LIF rate) + map
    return {"map readout (w_map . ring rates)": B * 2 * N,
            "PES map update (normalised LMS)": B * 2 * N * 3 + B * N * 2,
            "slow copy update": B * 2 * N * 2,
            "slow copy readout (deviation score)": B * 2 * N,
            "prediction reads (per horizon)": n_horizons * read_one,
            "input gate (expected position)": read_one,
            "clock rate update, phase normalisation, axis/drive, centre and gain, scores": B * 20 + 40}


def run_one(entry, cached_txy, cached_pred):
    clip = open_one(entry)
    loc = CountingLocaliser(make_localiser("snn", LOCALISER))
    txy = np.array(list(positions(clip, WINDOW_US, loc)), dtype=float)
    same_pos = np.array_equal(txy, cached_txy, equal_nan=True)

    class Replay:
        def __init__(self):
            self.i = 0

        def reset(self):
            self.i = 0

        def locate(self, _frame):
            x, y = txy[self.i, 1:]
            self.i += 1
            return x, y

    mem = make_memory("snn_phasemap", dt_s=WINDOW_US / 1e6, warmup_s=5.0)
    mrows = count_memory(mem)
    traces = evaluate_clip(mem, clip, WINDOW_US, HORIZONS, localiser=Replay())
    pred = np.stack([tr.pred for tr in traces])
    same_pred = np.array_equal(pred, cached_pred, equal_nan=True)
    max_diff = float(np.nanmax(np.abs(pred - cached_pred))) if pred.shape == cached_pred.shape else np.inf
    return loc, mem, txy[:, 0], loc.rows, mrows, same_pos, same_pred, max_diff


def main() -> None:
    torch.manual_seed(0)
    entries = {e["name"]: e for e in load_set("development")}
    out, checks = [], []
    for name in RUNS:
        stem = name.replace("/", "__")
        cached_txy = np.load(DEV / "positions" / f"{stem}__snn.npz")["txy"]
        cached_pred = np.load(DEV / "traces" / f"{stem}__snn_phasemap__snn.npz")["pred"]
        loc, mem, t, lrows, mrows, same_pos, same_pred, diff = run_one(entries[name], cached_txy, cached_pred)
        checks.append((name, same_pos, same_pred, diff))
        print(f"{name}: positions identical {same_pos}, predictions identical {same_pred} (max diff {diff:.2e})",
              flush=True)
        net = loc.loc.net
        (c1, h1, w1), (c2, h2, w2) = net.hw
        steady = t >= STEADY_S
        col = lambda rows, k: np.array([r[k] for r in rows])  # noqa: E731
        neurons = {"input": net.conv1.in_channels * np.prod(loc.loc.in_hw), "conv1": c1 * h1 * w1,
                   "conv2": c2 * h2 * w2, "fc": net.fc.out_features}
        macs = localiser_dense_macs(net)
        layer_rows = [
            ("localiser", "input (events)", neurons["input"], col(lrows, "in_events"), col(lrows, "in_synops_event"), macs["conv1"]),
            ("localiser", "input (active pixels)", neurons["input"], col(lrows, "in_pixels"), col(lrows, "in_synops_pixel"), macs["conv1"]),
            ("localiser", "conv1 LIF -> conv2", neurons["conv1"], col(lrows, "s1"), col(lrows, "syn1"), macs["conv2"]),
            ("localiser", "conv2 LIF -> fc", neurons["conv2"], col(lrows, "s2"), col(lrows, "syn2"), macs["fc"]),
            ("localiser", "fc LIF -> readout", neurons["fc"], col(lrows, "s3"), col(lrows, "syn3"), macs["read"])]
        clock, ring, snap = col(mrows, "clock"), col(mrows, "ring"), col(mrows, "snapshot")
        n_clock, n_ring, B = mem.n_clock * mem.B, mem.n_ring * mem.B, mem.B
        sub = mem.clock.substeps
        d_clock, d_ring = mem.clock.w.decoders.shape[0], mem.ring.w.scaled_encoders.shape[1]
        layer_rows += [
            ("memory", "clock LIF (6 clocks): factored recurrence + readout", n_clock, clock, clock * 2 * d_clock,
             n_clock * d_clock * sub),
            ("memory", "clock LIF (6 clocks): full recurrent matrix", n_clock, clock, clock * mem.n_clock, np.nan),
            ("memory", "ring LIF (6 rings) -> map (and slow copy)", n_ring, ring, ring * 2 * (1 + snap),
             n_ring * d_ring * sub)]
        for part, item, n, spikes, synops, dense in layer_rows:
            out.append({"run": name, "part": part, "item": item, "neurons": int(n),
                        "spikes_per_step": spikes.mean(), "spikes_per_step_steady": spikes[steady].mean(),
                        "firing_fraction_per_step": spikes.mean() / n,
                        "synops_per_step": synops.mean(), "synops_per_step_steady": synops[steady].mean(),
                        "dense_ops_per_step": dense})
        for item, ops in memory_conventional_ops(mem, len(HORIZONS)).items():
            out.append({"run": name, "part": "memory, non-spiking", "item": item, "dense_ops_per_step": ops})
        out.append({"run": name, "part": "neuron updates", "item": "localiser LIF (1 per step)",
                    "neurons": int(neurons["conv1"] + neurons["conv2"] + neurons["fc"]),
                    "dense_ops_per_step": neurons["conv1"] + neurons["conv2"] + neurons["fc"]})
        out.append({"run": name, "part": "neuron updates", "item": f"memory LIF ({sub} substeps per step)",
                    "neurons": int(n_clock + n_ring), "dense_ops_per_step": (n_clock + n_ring) * sub})
        del loc, mem
    fields = ("run", "part", "item", "neurons", "spikes_per_step", "spikes_per_step_steady",
              "firing_fraction_per_step", "synops_per_step", "synops_per_step_steady", "dense_ops_per_step")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    print(f"wrote {OUT}")
    if not all(c[1] and c[2] for c in checks):
        sys.exit(f"outputs differ from the cached run: {checks}")


if __name__ == "__main__":
    main()
