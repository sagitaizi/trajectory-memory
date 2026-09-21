# SNN memory — experiment log (detailed)

Every training run of the Stage-1 spiking trajectory memory (`trajmem/snn.py`), in order,
with configuration, numbers and what was concluded. The short version is
`snn-experiments-summary.md`. Design decisions themselves are in `PLAN.md` §C.

## Setup common to all runs

- **Data**: `corpus/sim/tracks.npz` — 100 simulated clips × 15 s, one measured (centroid)
  and one true position per 5 ms window. Only the **57 clips whose centroid sits within
  15 px (median) of the truth** are used; on the other 43 the centroid is locked on the
  string, and a memory fed positions cannot fix that (it is the localiser's job).
- **Split**: `np.random.default_rng(seed).permutation` over the 57 — with seed 0, the first
  6 are validation, 51 train. Fixed across runs unless noted.
- **Loss masking**: no loss during each clip's first cycle (the network cannot know the
  path yet), none after a scripted break (it must not learn broken motion).
- **Training**: surrogate-gradient BPTT (snnTorch `fast_sigmoid`, slope 25), own PyTorch
  loop, 5 s chunks (1000 steps) with the state carried across chunks, Adam, gradient norm
  clipped at 1.0, batch 16, best epoch by validation error. CPU, 4 threads (measured: the
  GPU is slower at these sizes, and 10 threads are 3× slower than 4), **pinned to the
  performance cores** — found 2026-09-18 07:15: Windows had been scheduling every
  background training run onto the i7-12650H's four efficiency cores (2–3× slower), which
  is why every run in this log took 3–5× its estimate. Runtimes quoted before that point
  are inflated accordingly; `SpikingMemory` now sets the affinity itself.
- **Metrics**: median pixel error of each horizon head on the validation clips, on
  on-pattern steps after the first cycle; the development-set scorecard from
  `run_experiment.py --set development` (8 real-clip entries, pooled median at 100 ms).
- **Reference numbers**: Kalman baseline 7.5 px at 100 ms on sim clips where the centroid
  tracks, 24.8 px pooled on the development set (8.3 on the fan). *Persistence* (repeat the
  input as the prediction) on the validation sim clips: 11 / 18 / 29 / 51 px at
  25 / 50 / 100 / 200 ms — any useful memory must beat this.

## Run 1 — first design (2026-09-17, night)

Design as decided: place cells in (32/axis), fast layer 192 LIF (τ 20–50 ms) ↔ slow layer
128 LIF (τ 300–700 ms), linear (x, y) readout per horizon from a 50 ms low-pass of the fast
spikes, path head (17 numbers) from the slow layer, path weight 0.1, lr 1e-3, 30 epochs.

| | 25 ms | 50 ms | 100 ms | 200 ms |
|---|---|---|---|---|
| validation | 29.9 | 41.2 | 46.5 | 64.0 |

Best epoch 20 (46.5 px), plateau from epoch ~10. Development set pooled 63.2 px (fan 39.2,
`small_01` 45.9, `loop_01` 69.0). Slow layer firing rate fell from 10 % (epoch 1) to 0–1 %
and stayed there; path loss flat (0.72 → 0.61).

**Diagnostics on the trained network** (validation clips, ridge readout fitted post hoc):

| decode from → | current position (0 ms) | 100 ms ahead |
|---|---|---|
| filtered spikes (what the heads see) | 30.1 px | 62.7 px |
| membrane potentials | 14.6 px | 59.9 px |
| untrained network, filtered spikes | 107.8 px | — |

Conclusions: (1) a rate-coding precision floor — the current position cannot be read from
the spikes better than ~30 px, though the neurons hold it to ~15 px; (2) no anticipation —
100 ms ahead is worse than persistence from any decode; (3) the slow layer dies.

**Short variants** (24 tracks, 8 epochs, val 100 ms px per epoch; base = the design above):

| variant | epochs 1→8 | final rates fast/slow | note |
|---|---|---|---|
| base | 151 109 104 84 67 68 64 75 | 0.15 / 0.01 | |
| surrogate slope 5 | NaN from epoch 1 | 0 / 0 | diverged |
| no gradient clipping | 160 126 112 122 105 97 89 94 | 0.18 / 0.02 | worse |
| slow-layer input gain ×3 | 152 111 106 83 68 70 64 74 | 0.15 / 0.01 | no effect; optimiser still silences it |
| lr 3e-3 | 112 85 85 91 62 68 70 69 | 0.09 / 0.00 | faster start, same plateau |
| fast τ 10–25 ms, readout 15 ms | 159 114 98 78 62 66 56 63 | 0.13 / 0.01 | ~10 % better |
| + n_fast 384 | 97 81 61 61 61 62 59 60 | 0.12 / 0.03 | ~10 % better |
| + readout 10 ms, τ 10–20 ms | 106 89 65 59 62 63 58 57 | 0.10 / 0.01 | same |
| rate regulariser target 0.1, w 10 (on n_fast 384, fast τ) | 95 86 66 56 63 63 60 61 | 0.10 / 0.09 | slow layer alive, path loss moves (1.05 → 0.94), error unchanged |
| rate regulariser target 0.2 | 105 108 69 63 69 65 61 62 | 0.13 / 0.20 | same |

## Run 2 — place cells out + rate regulariser (2026-09-17, night)

Change: each horizon head outputs 32 cells per axis (softmax, centre of mass = position),
trained by cross-entropy against a bump on the true position; fast layer 384; firing-rate
regulariser `10 × Σ_layers (rate − 0.1)²`. Two full runs, 30 epochs.

| lr | best epoch | 25 ms | 50 ms | 100 ms | 200 ms | rates |
|---|---|---|---|---|---|---|
| 1e-3 | 30 (still falling) | — | — | 45.7 | — | 0.16 / 0.11 |
| 3e-3 | 28 | 20.7 | 26.6 | 39.0 | 58.9 | 0.13 / 0.10 |

The precision floor moved (30 → 21 px at 25 ms) and the slow layer stays alive. Still
worse than persistence at every horizon: the network reproduces the present and has not
learnt motion. Short-variant check (24 tracks, 8 epochs): 74 / 50–59 (lr 3e-3) / 60 (fast
τ) px — still descending at epoch 8, unlike Run 1's plateau.

## Run 3 — velocity in + displacement out (2026-09-17, night, in progress)

Change: input gains 16 cells per axis over the displacement of the last 25 ms (±0.08 of
the image, ≈ ±50 px); each horizon head predicts the *offset* from the last seen position,
over cells spanning ±max(0.1, 1.75 × horizon) of the image, added back to the position.
Persistence is now the network's zero output. Everything else as Run 2 (lr 3e-3).

Sweep (two lanes in parallel; `runs/memory/sweep.csv`, checkpoints `runs/memory/sweep/`):

| name | change vs base | best ep | 25 | 50 | 100 | 200 | dev pooled | fan | rates | path |
|---|---|---|---|---|---|---|---|---|---|---|
| m1_lr1e-3 | lr=0.001 | 30 | 6.5 | 9.6 | **16.4** | 32.3 | 31.8 | 9.5 | 0.128/0.109 | 0.759 |
| m1_base | base | 30 | 5.8 | 8.7 | **15.6** | 33.8 | 25.3 | 10.7 | 0.108/0.093 | 0.592 |
| m1_seed1 | seed=1 | 14 | 6.2 | 9.3 | **16.6** | 33.7 | 31.0 | 13.6 | 0.109/0.099 | 0.848 |
| m1_vel50 | vel_window_s=0.05 | 26 | 6.2 | 9.5 | **16.4** | 31.8 | 31.7 | 10.7 | 0.104/0.108 | 0.683 |
| m1_vel32 | n_vel_cells=32, vel_range=0.12 | 13 | 6.2 | 9.0 | **15.5** | 30.5 | 31.1 | 11.9 | 0.112/0.11 | 0.667 |
| m1_fasttau | tau_fast=[0.01, 0.025], readout_tau_s=0.015 | 21 | 6.0 | 8.7 | **14.5** | 30.3 | 27.8 | 9.7 | 0.108/0.104 | 0.78 |
| m1_rate02 | rate_target=0.2 | 25 | 5.7 | 8.4 | **15.9** | 33.5 | 29.8 | 14.4 | 0.195/0.199 | 0.6 |
| m1_big768 | n_fast=768 | 19 | 5.8 | 8.7 | **16.1** | 35.3 | 27.4 | 12.1 | 0.1/0.114 | 0.859 |
| m1_pw0 | path_weight=0.0 | 26 | 5.9 | 9.0 | **16.2** | 35.2 | 28.2 | 10.7 | 0.103/0.096 | 0.77 |
| m2_cells48 | n_per_axis=48 | 26 | 5.6 | 8.5 | **16.1** | 33.9 | 28.7 | 9.0 | 0.105/0.101 | 0.652 |
| m1_e60 | epochs=60 | 30 | 5.8 | 8.7 | **15.6** | 33.8 | 25.3 | 10.7 | 0.108/0.093 | 0.592 |
| m2_outnarrow | out_range_per_s=1.2, out_range_min=0.06 | 25 | 6.0 | 9.3 | **16.1** | 31.2 | 28.1 | 10.7 | 0.102/0.099 | 0.669 |
| m2_rw3 | rate_weight=3.0 | 15 | 6.4 | 8.9 | **14.4** | 30.5 | 28.2 | 12.6 | 0.131/0.106 | 0.7 |
| m2_e90cos | epochs=90, schedule=cosine | 22 | 6.0 | 8.6 | **15.4** | 33.2 | 29.9 | 11.9 | 0.107/0.094 | 0.62 |
| m2_e60cos | epochs=60, schedule=cosine | 13 | 6.3 | 8.9 | **15.0** | 33.2 | 33.7 | 14.5 | 0.111/0.099 | 0.79 |
| m2_e60cos_lr5 | epochs=60, lr=0.005, schedule=cosine | 19 | 5.5 | 8.0 | **14.6** | 30.6 | 29.5 | 11.5 | 0.106/0.099 | 0.707 |
| m2_both | heads_from=both | 13 | 6.0 | 8.7 | **15.8** | 30.5 | 29.0 | 15.0 | 0.108/0.092 | 0.685 |
| m2_flips | data=flips | 6 | 5.6 | 7.8 | **13.6** | 27.3 | 26.4 | 14.9 | 0.103/0.095 | 0.55 |
| m2_allgt | data=all_gt | 19 | 4.9 | 7.5 | **12.8** | 24.4 | 24.4 | 12.7 | 0.105/0.106 | 0.541 |
| m2_wd | weight_decay=0.01 | 26 | 5.8 | 8.8 | **15.4** | 30.8 | 27.8 | 9.1 | 0.105/0.102 | 0.757 |
| m2_both_allgt_e60cos | heads_from=both, epochs=60, schedule=cosine, data=all_gt | 31 | 5.2 | 7.6 | **13.5** | 26.2 | 32.4 | 9.8 | 0.099/0.104 | 0.615 |
| m2_allgt_flips_e40cos | epochs=40, schedule=cosine, data=all_gt_flips | 14 | 5.5 | 7.6 | **11.7** | 21.9 | 25.7 | 12.3 | 0.098/0.097 | 0.533 |
| m3_lsnn | slow_kind=adaptive, epochs=40, schedule=cosine, data=all_gt_flips | 14 | 5.4 | 7.5 | **12.2** | 25.2 | 28.9 | 13.2 | 0.119/0.094 | 0.575 |
| m4_200_leaky | slow_kind=leaky, epochs=40, schedule=cosine, data={'include_locked': True, 'augment': 'flips'}, corpus=200, dev_rescored_offset=True | 33 | 5.6 | 7.3 | **10.7** | 19.2 | 16.6 | 13.0 | 0.104/0.096 | 0.462 |
| m4_200_lsnn | slow_kind=adaptive, epochs=40, schedule=cosine, data={'include_locked': True, 'augment': 'flips'}, corpus=200 | 17 | 5.1 | 7.4 | **11.7** | 21.0 | 16.6 | 11.9 | 0.107/0.097 | 0.53 |
| m4_200_leaky_anchor20_partial | slow_kind=leaky, anchor_tau_s=0.02, epochs=40, schedule=cosine, data={'include_locked': True, 'augment': 'flips'}, corpus=200, stopped_at=34 | ~34 (run stopped) | 5.7 | 7.2 | **10.2** | 17.7 | 17.4 | 12.5 | / |  |

**Reading the sweep** (noise: same config, another seed → 15.6 vs 16.6 px at 100 ms; the
development-set pooled number moves ±5 px between seeds because it is a median over 8
entries, so judge on the sim validation columns and the fan clip):

- The velocity-in / displacement-out change is the decisive one: 39 → 15.6 px at 100 ms,
  and below persistence at every horizon (5.8 / 8.7 / 15.6 / 33.8 vs 11 / 18 / 29 / 51).
- Fast time constants (τ 10–25 ms, readout 15 ms) are worth ~1 px; more velocity cells, a
  wider velocity window, 768 fast neurons, 48 output cells, narrower output ranges, rate
  target 0.2, rate weight 3, no path loss, 256 slow neurons: within noise.
- Longer training does not help: with constant lr 3e-3 the validation error sits at 16–18
  px from epoch ~20 on (`m1_e60` never beat its epoch 30); with a 90-epoch cosine schedule
  the validation *loss* bottoms at epoch 15–20 and then rises while the training loss keeps
  falling — **overfitting to 51 training clips**. Every optimiser-side variant lands at
  14.5–16 px. The remaining gap to the Kalman (7.5 px) is a data problem: hence the
  augmentation runs (mirror flips ×4, the 43 string-locked clips with true positions
  standing in, weight decay).
- Mirror flips (204 training tracks): 13.6 / 27.3 px at 100 / 200 ms, the best of the
  night, and the validation loss bottoms lower (2.30 vs 2.37–2.38) — but it still bottoms
  early (epochs 6–13) and drifts up after. The validation set is 6 clips, so the per-epoch
  number wanders ±1 px and "best epoch" flatters by about that much; ~14 px is the honest
  figure. Installed as `runs/memory/snn.pt`.
- The 43 string-locked clips with the true position (+3 px noise) standing in for the
  centroid (`m2_allgt`, 94 training clips): 12.8 / 24.4 px. Combined with flips and a
  40-epoch cosine schedule (`m2_allgt_flips_e40cos`, 376 training tracks): **11.7 / 21.9 px
  at 100 / 200 ms**, 5.5 / 7.6 at 25 / 50 — the best of the sweep, installed as
  `runs/memory/snn.pt`. Weight decay 0.01: no change on sim (15.4). More data is the lever
  that keeps paying; everything else is flat.
- Horizon heads reading from both layers (`m2_both`): no change (15.8; 200 ms 30.5). The
  slow layer is not yet carrying anything the fast layer lacks — consistent with the path
  head's loss barely moving. The cycle memory is the open problem, not the readout.

**Reproducing the best run from the repo** (the sweep driver itself is not in the repo):
`python scripts/train_memory.py --augment flips --include-locked --epochs 40 --schedule cosine`.
`trajmem/augment.py` also has `shift:n`, `scale:n`, `stretch:n` (random offset, resize about
the path centre, replay at 0.8–1.25× speed), untested as of this writing; the corpus is
being extended to 200 clips (`make_sim_dataset.py --n 200`, then `make_tracks.py`).

## Run 4 — adaptive slow layer (LSNN) (2026-09-17, afternoon, in progress)

Decision: the slow layer's neurons become adaptive LIF (`AdaptiveLeaky`): membrane τ
10–25 ms like the fast layer, threshold = 1 + 1.8 × a, with a decaying over τ_a spread
log-uniformly over 0.5–4 s and rising by 1 per spike. Rationale: a long membrane leak is
reset by every spike and blurs the input; the adaptation variable is a separate slow state
that accumulates across spikes — the proven way (Bellec 2018) to give a spiking network
seconds-long memory. Plain LIF kept as `slow_kind="leaky"` for the ablation.

Lane L1 (all on flips + string-locked clips, 40 epochs cosine): `m3_lsnn`;
`m3_lsnn_noreg` (rate regulariser off — does the adaptive layer stay alive on its own?);
`m3_leaky_noreg` (the old layer without the regulariser, for the record).

`m3_lsnn` (332 min, sharing the CPU three ways): 5.4 / 7.5 / 12.2 / 25.2 px vs the plain-LIF
twin's 5.5 / 7.6 / 11.7 / 21.9; path loss 0.58 vs 0.53; slow rate 0.09 throughout. **No
gain** — the adaptive slow layer holds the cycle no better than the leaky one, and the
200 ms head (where cycle knowledge would show) is slightly worse. Reading: the memory
*mechanism* is not the limit; the slow layer has no training signal that demands cycle
memory (path head at weight 0.1, otherwise only indirect via feedback). The options set
aside earlier — a phase head with real weight; long horizons routed through the slow
layer — are what would create that demand. The no-regulariser twins (`m3_lsnn_noreg`,
`m3_leaky_noreg`) were killed 2.2 h in (no progress output, competing with the 200-clip
run) and re-queued on the 200-clip corpus as `m4_200_lsnn_noreg` / `m4_200_leaky_noreg`.

**Centroid jitter goes straight into the SNN's prediction** (found 2026-09-17 afternoon).
The SNN and the baselines receive the identical centroid stream (`frontend.to_position`,
one per 5 ms window); the Kalman and the harmonic fit filter it through their state, but
the SNN's prediction is `raw last centroid + predicted offset`, and its velocity cells are
fed a raw difference of two centroids. Step-to-step jitter of the centroid (its change
minus the truth's change, px, median): sim_024 5.5, sim_038 2.7, **sim_072 12.6**,
small_01 3.4, fan 3.6, **loop_01 13.7** — on the worst clips more than the target moves
per step. Fix under test: `anchor_tau_s` — a causal running average of the centroid
(20–40 ms) as the anchor for the offsets and the velocity, with the training targets
defined relative to the same smoothed anchor (runs `m4_200_*_anchor*`). The proper fix is
the Stage 1 spiking localiser.

**Scoring change (2026-09-17 evening): `run_experiment.py --subtract-offset`.** The
development-set numbers quoted above include each clip's constant label-vs-centroid
offset (pendulum 25–42 px, wall target 12–21 px). With the offset removed, the same
checkpoints score: Kalman 14.8 pooled (fan 7.0, `loop_01` 61.9), Harmonic 24.0,
`runs/memory/snn.pt` 18.3 (fan 11.5, `loop_01` 29.0, `small_01` 14.1). `sweep.csv` rows
keep the old scoring; from lane M1 on, `dev_*` columns are offset-subtracted.

## Run 5 — 200 simulated clips (2026-09-17, evening, queued)

`make_sim_dataset.py --n 200` extends the corpus with clips sim_100–199 (same generator,
seeds [0, i]); `tracks.npz` rebuilt for all 200 (the 100-clip version kept as
`tracks_100.npz`). Validation stays the same six clips (sim_072, 024, 038, 023, 058, 091)
so every row in `sweep.csv` compares directly. Lanes M1 / L2: the best configuration on
200 clips, with plain-LIF and adaptive slow layers; anchor smoothing 20 / 40 ms;
shift/scale/stretch augmentation (`trajmem/augment.py`).

`m4_200_leaky` (776 training tracks, 697 min — most of it on the E-cores): **5.6 / 7.3 /
10.7 / 19.2 px**, best epoch 33 and still descending, path loss 0.46 (from 0.53). The
first run that does not overfit; data is still the lever. Installed as `runs/memory/snn.pt`.
Development (offset-subtracted from here on): pooled 23.4, fan 14.2, `loop_01` 31.0 —
note the sim gain does not show on the fan clip, whose error is now dominated by the
centroid's own 5.7 px and jitter (see the anchor runs). (Its `dev_*` row was first written
without offset subtraction — 23.4 — and rescored: pooled 16.6, fan 13.0, `small_01` 16.3,
`loop_01` 26.6.)

`m4_200_lsnn` (adaptive slow layer, same data, 432 min): 5.1 / 7.4 / 11.7 / 21.0 — again
~1–2 px behind the plain-LIF twin on sim, identical on the development set (16.6 pooled,
fan 11.9). Second null result for the LSNN; the conclusion of Run 4 stands.

`m4_200_leaky_anchor20` (20 ms running-average anchor for the offsets and velocity),
**stopped at epoch ~34 of 40** when training was wrapped up (2026-09-18 15:10); its
partial checkpoint scores 5.7 / 7.2 / **10.2 / 17.7** on sim — the best 100 and 200 ms
numbers so far — and 17.4 pooled on the development set (fan 12.5, `loop_01` 26.2). Kept
as `runs/memory/snn_anchor20.pt`; `runs/memory/snn.pt` stays the completed
`m4_200_leaky`. Not run: anchor 40 ms, shift/scale/stretch augmentation, the two
no-regulariser twins, LSNN + anchor, the `train_memory.py` reproduction — all queued in
the lane files and cheap to restart now that the E-core problem is fixed.

**Planned next (needs a decision): giving the slow layer a job.** Two changes, to be
tested singly and together, all else as the current best:
(a) *phase head*: the slow layer's head predicts only period + (cos φ, sin φ), weight 1.0
instead of the 17-number path head at 0.1 — the cycle position is what the fast layer
lacks at 100–200 ms; (b) *long horizons through the slow layer*: the 100 / 200 ms heads
read the slow layer only (25 / 50 stay on the fast), so the loss itself demands cycle
memory. Both in `TwoLayerNet` as options; ablations: each alone, plain vs adaptive slow
layer under each. Success = the 200 ms column moving (now 22–25 px, Kalman ~15).

**Where the remaining error comes from.** At 25 ms the network is at 5.6 px — the spike
readout's precision floor. From there the error grows to 13.6 px at 100 ms. Extrapolating
at constant velocity would give ~20 px at 100 ms on a 1 s, 100 px-radius circle
(acceleration ω²r ≈ 4000 px/s²), so the network does use some curvature — but the Kalman,
which knows the period and the harmonic shape, gets 7.5. Closing that gap means the slow
layer actually holding the cycle. Candidates for the next session, in order of cost:
longer effective memory via adaptive thresholds (LSNN, Bellec 2018) in the slow layer;
training the path head first (curriculum); more and more varied simulated clips (the
generator is seed-reproducible; overfitting is the binding constraint).

## Run 6 — the path-shape metric, the geometric loss, and routing (2026-09-19)

**Metric first.** `metrics.path_shape_error` scores the memory's own picture of one cycle
against the true cycle (px, best phase shift, sampled over one *true* period so the shape
is judged apart from the period, which is reported as `P / T`). Development set,
offset-subtracted, path median px / period ratio: Kalman 14.0 / 1.00, Harmonic 10.3 /
1.00, `snn_anchor20` **64.4 / 1.83**. On clean sim clips: Kalman 1.5–9 px, SNN 30–65 px.
The learned slow layer's path head held nothing usable; the prediction numbers came from
short-term extrapolation. (Found and fixed on the way: `search_period`'s divisor tie was
absolute and lost to centroid noise, so the baselines ran at 2T or 3T on some clips; it is
now relative, 2 % of the best residual.)

**Geometric path loss** (px of the drawn cycle at 16 phases / 50 + phase pair squared error
+ relative period squared error, weight 1.0; `--path-loss mse` keeps the old form): 12
epochs, 200-clip corpus, locked clips in, no flips, anchor 20 ms. Validation path px
120 → 70 (mean-path baseline 120), 100 ms 19.5 → 22 px (the heads compete for the fast
layer). The checkpoint selection then used the 100 ms error alone and kept epoch 2;
`fit` now selects by total validation loss.

**Routing** (`heads_from="split"`: 100/200 ms heads read the slow layer only), same
settings, side by side with the fast-only twin:

| | val 100 ms | val path px | dev pred | dev path | dev period |
|---|---|---|---|---|---|
| fast | 23.6 | 71.6 | 28.1 | 65.9 | 1.32 |
| split | 26.5 | 71.8 | 30.3 | 66.3 | 1.47 |

Same wall at ~72 px whichever way the cycle is asked for. Diagnosis: the slow layer's only
seconds-long variable is the adaptive threshold — a leaky spike count that records
occupancy, not a sequence; nothing in the network represents the recent trajectory.

**Decided (with Sagi, 2026-09-19): a hand-set window memory** — a spiking Legendre Memory
Unit built by Nengo and simulated in torch — replaces the learned slow layer
(`trajmem/lmu.py`, `trajmem/snn_lmu.py`; design in
`docs/superpowers/specs/2026-09-19-lmu-memory-design.md`). Sizing by measurement: q = 24
per axis over θ = 4 s reconstructs the position one period back to 0.5 px median / 3.8 px
p90 on the sim paths (q = 16: p90 35 px); 200 LIF neurons per state dimension with 1 ms
sub-steps track the exact LMU to 5 % of the signal (100 neurons, or 2.5 ms sub-steps,
drift). The population costs ~10 ms per network step whatever its size — kernel launch
overhead — so training runs at batch 32.

## Run 7 — first LMU memory (2026-09-19, night)

`train_memory.py --arch lmu --include-locked --epochs 12 --batch 32 --schedule cosine
--anchor-tau-s 0.02` → `runs/memory/lmu_quick.pt` (q 24, θ 4 s, 200 neurons per state
dimension, readouts through a fixed 512-feature random projection, blanks in half the
chunks, geometric path loss, path head linear from the LMU features). ~14 min per epoch.

| | val 100 ms | val path px | dev pred | dev path | dev period | fan path | `loop_01` path |
|---|---|---|---|---|---|---|---|
| two-layer, geometric loss (run 6) | 23.6 | 71.6 | 28.1 | 65.9 | 1.32 | 49 | 46 |
| **LMU quick** | 25.9 | **55.7** | 32.0 | **58.3** | 1.30 | **30** | **29** |
| Kalman | 7.5 | 1.5–9 | 14.7 | 14.0 | 1.00 | 6.8 | 49 |

Validation path 147 → 87 (epoch 1) → 56 (epoch 12), still falling at the end. Prediction
is worse than the two-layer net's (26 vs 24 on sim, 32 vs 28 on dev). Through a 1 s blank
(`--blank 8 9`, self-fed loop) the error inside the blank is 52–65 px on sim clips
(Kalman 13–57).

**Why the path stops near 56 px — measured, not guessed.** A readout fitted offline from
the *exact* LMU state (48 numbers, a perfect window) to the path targets: linear (ridge)
61 px shape / period ratio error 0.33; a 2×256 MLP 34 px / 0.15; mean path 111 px. So
the window holds what is needed and the network's linear head already extracts what a
linear map can. The limit is structural: every LMU neuron sees one state dimension, so a
linear readout of the population is additive over dimensions and cannot form the cross
terms (autocorrelation-like products) that a period estimate needs. Options added:
`--path-from both` (the path head also reads the fast layer, a nonlinear spiking layer
that sees the LMU features) and `--path-readout mlp` (hidden layer; the non-spiking upper
bound). Run 8 (`lmu_mlp`, both options) started 09:55.

**Open after this run.** (1) Prediction got worse with the LMU in the loop: ablate the
blanks (`--blank-prob 0`), the LMU→fast input, and the `seen` cell; 12 epochs without
flips is also short of the two-layer's best recipe. (2) Self-feeding drifts through
blanks: feed the LMU's own delay readout at lag = period ("where it was one cycle ago")
instead of the 25 ms head — the window already contains it. (3) The spiking version of
whatever nonlinearity the MLP proves necessary.

**Run 8 (`lmu_mlp`, `--path-from both --path-readout mlp`)**: val 100 ms 26.4 px, path
54.6 px — within a pixel of the linear head (55.7). A hidden layer over the 512 random
features of the spike rates (plus the fast layer) does not find what the offline MLP
found on the exact state (34 px). Run 9 (`lmu_state`, `--path-from state`: the path head
reads the window as Nengo's decoders read it, 48 numbers, scaled by the radii) started
11:45 to separate the substrate from the head.

**Run 9 (`lmu_state`)** stopped at epoch 4: path 57 px, the same plateau as runs 7 and
8 — reading the clean decoded window changes nothing. Remaining difference from the
offline probe: the training blanks (half the chunks carry up to a period of self-fed,
currently wrong, positions while the path targets stay on). **Run 10 (`lmu_noblank`)**:
run 9's configuration with `--blank-prob 0`, started 12:10.

**Run 10 (`lmu_noblank`, run 9 without training blanks)**: epochs 1–4 trace runs 7–9
exactly (path 97 → 58, 100 ms 32 → 28.5). The blanks are not what limits either number.

**Offline readout experiments (2026-09-19 midday) — what the window can give.**
Records the low-passed decoded window (48 numbers) for every masked step of the 188
training and 12 validation tracks (336 k / 21 k samples), then fits path heads offline
with thousands of Adam steps instead of the ~200 BPTT updates a 12-epoch run gives:

| readout of the window | val path px | period error |
|---|---|---|
| linear, exact (non-spiking) state | 61 | 0.33 |
| 2×256 MLP, exact state | 34 | 0.15 |
| 2×256 MLP, spiking state, 1 000 steps | **46** (then overfits: 51 at 6 000) | 0.19 |
| classical period search + harmonic fit on the window reconstructed from the exact state | **5.9** (p90 77) | 0.01 |
| same on the window reconstructed from the spiking state | 74–86 | 0.8–1.6 |

So: (i) a learned static readout of a 4 s window is a poor period estimator even with a
perfect window (34 px vs the baselines' 10–14); (ii) the classical decoder is excellent on
a perfect window — the harmonic baseline *is* a designed readout of this memory — but
collapses on the spiking window, whose state carries a 2–4 %-of-radius systematic error
per dimension (200 neurons per 1-D ensemble; 800 barely better; the discrete-time NEF
mapping of Voelker & Eliasmith 2018 changes nothing; a 200–500 ms low-pass makes it
worse, so it is not fast noise). Reconstructing a 4 s window weights all 24 Legendre
orders equally, so that error becomes ~90 px at any lag. The fidelity of the spiking
window, not the readout, is now the binding constraint on the path.

`LmuMemory.fit(path_pretrain_steps=N)` fits the path head offline on the recorded window
and freezes it (`--path-pretrain-steps`); not yet used in a full run.

**Run 10 final (`lmu_noblank.pt`)**: val 100 ms 26.0 px, path 54.0 px (the lowest of the
LMU runs); development set 33.2 px prediction / 58.9 px path / period 1.45 (fan 40,
`loop_01` 25). No blanks in training ≈ blanks (run 7: 32.0 / 58.3): neither number is
limited by the blanks. The four LMU runs agree within noise on both numbers.

**Standing at the end of the session (2026-09-19, 13:30).** Path on real clips 55–59 px
(learned slow layer 66, Kalman 14); prediction 32–33 px (two-layer best 17.4, Kalman
14.7; all LMU runs are 12 epochs without flips, the two-layer's best was 40 epochs with
flips). Checkpoints: `lmu_quick.pt` (linear head, blanks), `lmu_mlp.pt`, `lmu_noblank.pt`
(decoded-window MLP head, no blanks). Next: (1) the fidelity of the spiking window is the
lever for the path — more neurons per dimension scale slowly, so the question is the
representation (e.g. lower q with a shorter θ, or a 2-D ensemble per Legendre order pair,
or reading period from a *bank of shorter windows*); (2) the prediction side needs the
two-layer recipe (flips, 40 epochs) before it can be compared; (3) through blanks, feed
the window's own one-period-back position.

## Run 11 — the clock-and-map architecture, non-spiking prototype (2026-09-20, night)

Decided with Sagi (2026-09-19 evening) after the LMU: the memory is a **phase advancing
at a rate** (a clock) plus a **map from phase to position** learned online in the clip —
the Kalman's structure, buildable in neurons — instead of a recording of the past.
`trajmem/phasemap.py` (`PhaseMap`, registered as `--memory phasemap`):

- five clocks start at periods 0.8–4 s; each is an adaptive-frequency oscillator
  (Righetti, Buchli & Ijspeert 2006) pulled by the motion's main-axis signal (found online
  from a running covariance), phase and rate nudged by F·sin φ, in the clock's own units;
- each clock owns a 64-bin map, updated by a delta rule (running mean at first, then a
  steady step) and a global scale that follows the radial mismatch (the pendulum's decay);
- the clock is elected by the drive's zero-crossing period when that is steady, else by
  the smallest mismatch; prediction = map at φ + ωh plus the smoothed current residual,
  decaying over 2 s; `period`, `path_points` straight from the clock and map;
- deviation = mismatch against a **snapshot** of the map taken two laps after election
  (the remembered path; the working map keeps adapting for prediction).

Locking rule as first written (phase error from the map's own tangent) could not lock —
the map cannot referee the clock that draws it. Righetti's input-driven rule locks within
a few cycles on clean paths (100 ms error 3–8 px). Two bugs fixed on the way: additive
pulls slammed slow clocks to the rate floor; the early drive is unnormalised (clipped).

Development set, offset-subtracted, prototype vs Kalman:

| | pred 100 ms | path px | period | AUC | latency s | fp/min |
|---|---|---|---|---|---|---|
| Kalman | 14.7 | 14.0 | 1.00 | 0.80 | 0.95 | 13.2 |
| **PhaseMap** | **13.9** | **13.5** | 1.00 | **0.99** | **0.10** | **9.8** |

Per clip: fan 5.0 / 4.8 (Kalman 7.0 / 6.8), `small_01` 11.0 / 11.6 (12.1 / 12.1),
`loop_01` 29 / 26 (62 / 49), `wide_break` AUC 1.00 (0.85), `loop_break_01` AUC 0.98
(0.75). Loses: `loop_break_01` prediction 13.2 vs 12.2; `wide_02` 18.5 vs 18.2; the
diagonal sweep locks at half the period. Sim: `sim_000` 10.1 vs 9.1, through a 1 s blank
`sim_007` 10.5 vs 59.7; the 3:2 Lissajous (`sim_003`) is the known weak case — neither
coordinate carries the fundamental, so nothing at the true period drives the clock.
Snapshot for deviation: without it the memory re-learns the new path after a break within
~3 laps and the AUC on `loop_break_01` is 0.58.

## Run 12 — the clock-and-map memory in spiking neurons (2026-09-20)

`trajmem/snn_phasemap.py` (`--memory snn_phasemap`). Built with Nengo, run in torch, as
the LMU was: the five clocks are one batched 4-D LIF population (6 000 neurons) whose
recurrent wiring computes the adaptive-frequency-oscillator dynamics — the pull F·sin φ
and the amplitude term are products the neurons compute — with the phase as (x, y), the
rate and the drive as inputs; the rate is a slow modulatory scalar updated from the decoded
F·y, because as an integrator dimension inside the population it drifted the slow clocks
to the clamp within seconds (a known NEF limitation). Each clock drives a ring of 400
phase cells (2-D LIF population, unit-circle encoders, 10 ms synapse); the map is the
ring → position weights learned by the PES rule (normalised LMS: the unnormalised step was
~0.6 per observation and diverged). Prediction ahead evaluates the ring's rate curves at
the rotated phase. The output scale adapts about a running centre of the map's own output
(scaling the weights themselves about the image centre collapsed the map on the fan clip).
Election, residual carry and the snapshot are the prototype's; election commits at the
snapshot, and the snapshot waits until the mismatch has stopped falling (≥ 3 laps, ≤ 6
laps or 12 s). Nothing is pretrained. Cost ~3 ms per step for all five clocks.

Development set, offset-subtracted:

| | pred 100 ms | path px | period | AUC | latency s | fp/min |
|---|---|---|---|---|---|---|
| Kalman | 14.7 | 14.0 | 1.00 | 0.80 | 0.95 | 13.2 |
| PhaseMap (arithmetic) | 14.0 | 14.0 | 1.00 | 1.00 | 0.10 | — |
| **SpikingPhaseMap** | **14.3** | **12.4** | 1.00 | **0.99** | **0.11** | **8.1** |

Per clip vs the Kalman: prediction within 0.3–2 px behind on six clips, far ahead on two
(`loop_01` 36 vs 62, horizontal sweep 13.7 vs 17.3) — a tie in substance; path better on
five of eight; AUC 1.00 / 0.99 vs 0.85 / 0.75 and latency 0.14 / 0.08 s vs 1.49 / 0.40 on
the two break clips. Known weak cases: the diagonal sweep locks at half the period (both
versions; the zero-crossing estimate gives T/2 there); a 3:2 Lissajous has no energy at the
fundamental to lock to. Sweep: ring synapse 10 ms (20 → prediction 18.9; 5 → 16.1); ring
400 cells over 200: path 13.2 → 12.4.

**Run 13 — the input stage (2026-09-21).** Sagi's live look at the unlabelled fast fan
(`fan_brush_fast_02`): the centroid flips between brush and string (>40 px jumps on 8.8 %
of steps, vs 1.9 % on the slow fan) and the path looked wrong. Two causes. (1) The clip's
period is 0.71 s and the clocks' range stopped at 0.7 s — pinned, no lock; range now
0.4–6 s (prototype) / 0.4–4.5 s (spiking). (2) The flips. Tried: a period-scaled input
low-pass (tau = 5 % of the period, lag compensated in the map read) — on the labelled
development set it *costs* (pred 14.0 → 16.9, path 14.0 → 17.4): a low-pass over 5 % of the
period cuts the swing's amplitude ~5 % at the turnarounds; off by default (option kept).
An extra 0.5 s starting clock also cost (elected wrongly on some clips); dropped. **A map
gate** — once a clock is elected, an observation farther from the map's expectation than
2 × the smoothed all-sample gap + 10 px is not believed (the tolerance is measured over
all samples, otherwise rejection feeds on itself) — helps: prototype pred 14.0 → 13.9,
path 14.0 → 12.7; fast fan mismatch 14.7 → 15.6 (no smoothing; rejects 8 %, the flip
rate). Spiking with the gate and the wider range: pred 14.5, path 13.4, AUC 1.00.
The proper answer to the string remains the Stage 1 learned localiser.
Follow-ups the tests forced: missing and rejected samples must reach the memory as
*unseen* (holding the last position made it learn a frozen point through blanks); the
deviation score is computed from the raw observation before the gate (a break is exactly
what the gate refuses to believe — with the gate in front the score froze); and the
snapshot is now a **slow map** that consolidates toward the working map, quickly while
young (rate 1/age) and with a 30 s constant once old — a frozen snapshot taken during a
re-locking transient stayed 25–35 px off on a clean path. Development set after all of
this: prototype pred 13.5 / path 13.5 / AUC 1.00; spiking 14.5 / 13.4 / 0.98.

**Run 14 — code review, slow clips, and the spiking regression (2026-09-21).** The review
(eight findings, all fixed) included one behavioural bug: the drive's zero-crossing clock
advanced only on seen samples, so unseen or gate-rejected windows shortened the measured
period. Sagi's live look at slow rig scans found the centroid pure noise there: stuck
pixels carry 20–31 % of all events on the rig recordings (`frontend.without_hot_pixels`,
> 100 events/s over the clip, clips longer than 5 s), and the slow pans' periods (6–8 s)
exceeded the clocks' 6 s ceiling (range now 0.4–12 s, a 7 s starting clock, and no
commitment until the zero-crossing rhythm agrees with the elected clock or 20 s pass —
otherwise a fast clock commits on a slow, locally linear motion). `scan_pan_slow_02`
(6.0 s) now locks at 6.41 s with 6 px mismatch; the tilt scan (31 px of motion) and the
early-stopped break scan do not lock. Residual-fade hypothesis for the 200 ms horizon
disproven (faster fade: 19.3 → 22.6 px); the residual is informative there.

The spiking version had slipped to 22.7 px prediction (deterministic across runs), the
prototype not: the gate. Its map read is noisier than the prototype's table, so the same
tolerance rejected legitimate samples and starved the clock's drive. Development set,
spiking: gate 2.0 → 16.6 / 14.1 / 0.96; off → 13.8 / 13.0 / 0.95; **3.0 → 13.5 / 12.9 /
0.98** (default now). Prototype 14.5 / 13.1 / 1.00. Kalman 14.7 / 14.0 / 0.80.

## Run 15 — the spiking localiser (2026-09-21, night)

`trajmem/snn_localise.py` (`--localiser snn`, spec
`docs/superpowers/specs/2026-09-21-spiking-localiser-design.md`): conv 5×5/2 LIF → conv
5×5/2 LIF → dense LIF → 32 + 32 place cells and a "present" unit, read from a 15 ms
low-pass; runs continuously over the 5 ms count frames (2 × 60 × 80, hot-pixel filtered),
one step per frame. Trained on the 200 simulated clips only (180 / 20), augmented per
clip with 50–400 stuck pixels at 50–1500 events/s, a Poisson noise floor up to 0.5 per
cell, mirror flips, and up to three blank stretches with the target's events removed
(`present` False — without these the network never learns to say "not seen").

What it took to train at all: the frames are sparse (0.003 counts per cell on average),
so the default initialisation leaves the deeper layers silent and the surrogate gradient
has nothing to work with — `calibrate` scales each layer's weights so its currents reach
threshold on a sample of frames; the rate regulariser applies to the dense layer only
(sparse conv activity is what a localiser should have); and BPTT over 0.25 s chunks, not
2 s (the leak spans a few frames; long chunks dilute the gradient: 114 → 18 px on the
toy set). The first full run (8/16 channels, 128 hidden, lr 1e-2) stalled at 15–19 px
on sim validation with a flat training loss from epoch 1; the wide run (16/32, 256,
lr 3e-3) reached 10.2 px after one epoch and **6.7 px** at epoch 19 (the classical
centroid: ~11 px on sim against the exact truth). ~9 min per epoch.

Development clips, centroid vs localiser, px against the labels with the constant offset
removed (median / p90 / share of >40 px jumps):

| clip | centroid | spiking localiser | unseen |
|---|---|---|---|
| `small_01` | 7.6 / 15.1 / 1.5 % | 23.1 / 35.9 / 0 | 0 |
| `wide_02/steady` | 11.9 / 25.8 / 0.9 % | 24.9 / 48.3 / 0 | 0.6 % |
| `wide_break` | 21.9 / 65.0 / 2.5 % | 25.5 / 48.6 / 0.8 % | 8.5 % |
| `fan_brush_slow_02` | 5.2 / 15.3 / 1.5 % | 7.9 / 12.7 / 0 | 5.8 % |
| `loop_01` | 16.2 / 52.2 / 5.7 % | 45.0 / 77.7 / 0.6 % | 2.7 % |
| `loop_break_01` | 14.0 / 26.4 / 1.1 % | 46.7 / 108 / 0.4 % | 77 % |

It transfers to the fan — the setup the simulator was matched to: the string flips are
gone and the tail is tighter, the median a little worse. On the pendulum it removes the
jumps but marks a different point of the brush less precisely. On the wall target it
fails: the target is a small rectangle printed on a large hand-held sheet whose outline
also makes events, nothing like the simulator's filled blobs, and the network mostly
calls it absent. A simulator-content gap, not a network one: a "sheet" target family in
the simulator is the fix.

End to end (memory = `snn_phasemap`, development set, offset-subtracted), centroid input
vs localiser input, prediction / path: fan **7.3 / 6.6 → 7.8 / 5.2** (path better,
prediction a little worse); `small_01` 12.5 / 10.7 → 13.4 / 13.5; the wide pendulum
clips 17–20 → 19–23; `loop_01` 18 / 16 → 41 / 22; `loop_break_01` 13 / 24 → 83 / 63 (the
localiser calls the sheet absent most of the time, so the memory coasts). Pooled 13.5 /
12.9 / AUC 0.98 → 32.4 / 21.8 / 0.79. The localiser is the memory's better input on the
fan only; the centroid stays the default.
