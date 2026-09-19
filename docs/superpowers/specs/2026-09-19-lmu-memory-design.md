# LMU memory core — design

The trajectory memory's slow part becomes a spiking Legendre Memory Unit (LMU): a population
of LIF neurons whose fixed recurrent wiring holds the last θ seconds of the target's position.
Everything that uses it is a learned linear readout of its spikes. The learned two-timescale
network (`snn.TwoLayerNet`) stays as the ablation.

## Why

Under three different training signals (coefficient MSE, geometric path loss, long horizons
routed through the slow layer) the learned slow layer's path readout stalls at ~72 px on
simulated validation clips (mean-path baseline 120, Kalman 1.5–9). Its only seconds-long
variable is an adaptive threshold — a leaky spike count that records occupancy, not a
sequence. Nothing in the network represents the recent trajectory, so nothing can read a
cycle off it. Decided 2026-09-19: a hand-set memory (the mechanism's wiring is designed, its
use is learned); LMU chosen over a tapped delay line (its cartoon) and an oscillator bank.

## What the memory must do

1. **Run without input**: with the target unseen for a stretch, predictions stay on the path
   (the network feeds its own prediction back in).
2. **Describe the path**: period, phase and shape readable at any moment (the path-shape metric
   in `metrics.path_shape_error`).

## Structure

```
position (x, y) ─► place + velocity cells + seen flag ─► fast layer (recurrent LIF, 10–25 ms) ─► 25/50 ms heads
      │                                                       ▲             │
      │                                            learned w_sf (LMU spikes → fast)
      └──────────────────────────────────────────► LMU population (fixed) ─┬─► path head (period, phase, shape)
                                                                           └─► 100/200 ms heads (with the fast layer)
```

- **LMU population** (`trajmem/lmu.py`): input u = position − 0.5 (two channels), window
  θ = 4 s (the longest cycle), order q per channel (chosen by test, expected 24–32).
  Continuous dynamics θ·ṁ = A·m + B·u with the Legendre matrices; represented by a Nengo
  `EnsembleArray` of one-dimensional LIF ensembles (n neurons each, chosen by test), the NEF
  recurrent transform τ_syn·A/θ + I and input transform τ_syn·B/θ, synapse τ_syn = 20 ms.
  Nengo builds it (encoders, gain, bias, decoders → connection weights); torch simulates the
  same neurons (a port of `nengo.LIF.step`, 1 ms sub-steps, 5 per network step) so the
  population runs in the training loop. Nothing in it is learned. Its output is the
  synaptic-filtered spike train (N values, Hz) and, for tests, the decoded state m.
- **Fast layer**: unchanged (`snn.TwoLayerNet`'s fast layer: LIF, τ 10–25 ms, recurrent),
  with an extra learned input `w_sf` from the LMU's filtered spikes and one extra input cell,
  `seen` (1 observed, 0 imagined).
- **Readouts** (all learned, linear, from 15 ms low-passed spikes): 25/50 ms displacement
  heads from the fast layer; 100/200 ms heads from fast ⊕ LMU; path head (17 numbers, as
  `snn.path_targets`) from the LMU population.
- **Self-feeding**: when the observation is NaN, the position fed to the place cells, the
  velocity anchor and the LMU is the network's own estimate of the current position: the
  25 ms head's prediction made 25 ms earlier (a ring buffer). The fed-back value is detached
  from the gradient. The `seen` cell tells the network which regime it is in.
- **Interface**: `model.TrajectoryMemory` unchanged (`observe`, `predict`, `deviation_score`,
  `period`, `path_points`, `reset`). `experiment.make_memory("snn", checkpoint=…)` picks the
  class from the checkpoint's `arch` key (`two_layer` | `lmu`).

## Training

- Same corpus, split, augmentation and loss as today: place-cell cross-entropy on the horizon
  heads, geometric path loss (weight 1.0), firing-rate regulariser on the fast layer only
  (the LMU's rates are fixed by construction). Checkpoint by total validation loss.
- **Blanks**: per chunk, with probability 0.5, one stretch of 0.25–1.0 periods starting after
  the first period is blanked (observation → NaN) in the *input* only; the targets and masks
  are unchanged, so the heads must predict through the blank from the self-fed loop.
- One step function serves training and streaming (encoding, anchor, velocity, self-feeding,
  LMU sub-steps, fast layer, readouts) so the two cannot drift apart.

## Evaluation

- `run_experiment.py --memory snn --checkpoint …` unchanged; the path columns measure job 2.
- Job 1: `evaluate_clip(…, blank=(t0, t1))` sets the observations in [t0, t1) to NaN;
  `score_trace` reports `blank_px` (median prediction error inside the blank). Script flag
  `--blank T0 T1`. Reported on simulated clips (exact truth) and on the fan clip.
- Ablations: two-layer net (today's checkpoints); LMU exact (non-spiking state, the ceiling);
  no self-feeding (blanks fed as zeros).

## Sizes, by measurement before any training

- q: exact LMU on the simulated paths, reconstruct the position one period back from m,
  px error for q ∈ {16, 24, 32, 48}; smallest under ~2 px.
- n per 1-D ensemble: spiking population on the same paths, decoded m vs exact; smallest n
  whose reconstruction is within ~2 px of the exact LMU's.
- Radius of each 1-D ensemble: from the 99th percentile of |m_i| over the corpus, per
  Legendre order, so every dimension uses its neurons' range.

## Files

| File | Role |
|---|---|
| `trajmem/lmu.py` | Legendre matrices, exact discrete LMU (reference), Nengo build → torch tensors, torch LIF population step. |
| `trajmem/snn_lmu.py` | `LmuNet` (fast layer + LMU population + readouts, one step function) and `LmuMemory` (streaming, training, checkpoint). |
| `trajmem/snn.py` | unchanged network; `path_targets`, `path_position`, `_shape_points`, `PlaceCells`, `HORIZONS_S` shared. |
| `trajmem/experiment.py` | `make_memory` dispatch on `arch`; `evaluate_clip(blank=…)`; `blank_px`. |
| `scripts/train_memory.py` | `--arch lmu`, `--blank-prob`, `--lmu-q`, `--lmu-n`. |
| `scripts/run_experiment.py` | `--blank T0 T1`. |
| `BIBLIOGRAPHY.md`, `docs/implementation-plan.md`, `PLAN.md` | Voelker & Eliasmith 2018, Voelker et al. 2019; the new module; status. |

## Out of scope

Replacing the fast layer; Stage 2 raw events; re-learning after a break; e-prop.
