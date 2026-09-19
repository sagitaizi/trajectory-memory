# LMU Memory Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the learned slow layer with a spiking Legendre Memory Unit built by Nengo and run in torch, so the path readout and through-blank prediction work.

**Architecture:** `trajmem/lmu.py` holds the LMU maths, the Nengo build and the torch LIF population; `trajmem/snn_lmu.py` holds `LmuNet` (fast layer ⊕ LMU population ⊕ readouts, one step function for training and streaming with self-feeding through blanks) and `LmuMemory` (the `TrajectoryMemory`). `experiment.make_memory` dispatches on the checkpoint's `arch`.

**Tech Stack:** Python 3.14 (`D:\Programs\Anaconda\envs\thesis\python.exe`), torch 2.14 CPU, snntorch 1.0 (fast layer), nengo 4.1 (build only), numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-19-lmu-memory-design.md`

## Global Constraints

- Run everything with `D:/Programs/Anaconda/envs/thesis/python.exe`; never `conda run`.
- Comments minimal: a non-obvious *why* in a line or two. No narrative in source.
- Nothing inside the LMU population is learned; Nengo is used at construction time only.
- The LMU population steps 5 sub-steps of 1 ms per 5 ms network step (Nengo LIF: τ_rc 20 ms, τ_ref 2 ms).
- `TrajectoryMemory` interface unchanged; existing checkpoints (`arch` absent → `two_layer`) keep loading.

---

### Task 1: Exact LMU and the q test

**Files:** Create `trajmem/lmu.py`; Test `tests/test_lmu.py`.

**Produces:** `legendre_matrices(q, theta) -> (A, B)` continuous (q,q), (q,1); `ExactLmu(q, theta, dt, n_channels=2)` with `.step(u)` → state (…, n_channels*q) via zero-order-hold discretisation (`scipy.linalg.expm` on the augmented matrix); `reconstruct(state, lag_s, q, theta)` → value at `lag_s` back, using shifted Legendre polynomials P_i(2·lag/θ − 1) (Voelker 2019, eq. for the delay readout).

- [ ] Test: a 1 Hz sinusoid through `ExactLmu(q=24, theta=2.0, dt=0.005)`; after 3 s, `reconstruct(state, 0.5)` ≈ sin at t−0.5 within 0.02.
- [ ] Test: `legendre_matrices` shapes and A's known first entries (A[0,0] = −1/θ, A[1,0] = −3/θ, B[0] = 1/θ, B[1] = −3/θ).
- [ ] Implement; run tests; commit "LMU: exact reference".
- [ ] q test (script in scratchpad, result recorded in the log): all sim tracks, u = gt − 0.5, θ = 4 s, reconstruct one period back; report px error for q ∈ {16, 24, 32, 48}. Pick smallest with median < 2 px.

### Task 2: Nengo-built spiking LMU population, torch-simulated

**Files:** Modify `trajmem/lmu.py`; Test `tests/test_lmu.py`.

**Produces:** `build_population(q, theta, n_per_dim, radii, tau_syn=0.02, seed=0) -> LmuWeights` (dataclass of torch tensors: `encoders` (N, D), `gain` (N,), `bias` (N,), `w_rec` (D, N) = (τA/θ+I)·decoders, `w_in` (D, C) = τB/θ per channel, D = C·q). Built from `nengo.networks.EnsembleArray(n_per_dim, D)` with `radius` per sub-ensemble, `Connection(ea.output, ea.input, transform=…, synapse=tau_syn)`, `Connection(inp, ea.input, transform=…, synapse=tau_syn)`; weights read from `sim.data[conn].weights` and `sim.data[ens]` per sub-ensemble, stacked block-diagonally. `SpikingLmu(weights, dt=0.005, substeps=5)` with `.init_state(batch)` and `.step(u, state) -> (act, state)`: per sub-step, `x = w_rec @ act + w_in @ u`; `J = gain ⊙ (encoders @ x) + bias`; Nengo LIF step ported (voltage, refractory, spikes·1/dt_sub); `act = lowpass(spikes, tau_syn)` in Hz. `.decode(act) -> m` via the decoders (kept in `LmuWeights.decoders` (D, N)).

- [ ] Test: same sinusoid as Task 1 through `SpikingLmu` with q=8, n_per_dim=50; decoded state tracks `ExactLmu` (RMS < 0.1 of signal amplitude after 1 s), reconstruction at lag 0.5 s within 0.15.
- [ ] Test: with u frozen (fed as the last value) the decoded state keeps evolving as the exact LMU does under the same frozen input (window keeps sliding; both agree).
- [ ] Test: torch `SpikingLmu` vs Nengo's own `Simulator` on the same built network, same input, dt 1 ms: probe of `ea.output` (synapse 0.01) vs torch decode — RMS difference < 0.05 (seeds differ in spike timing; compare filtered).
- [ ] Implement; tests pass; commit "LMU: Nengo-built spiking population in torch".
- [ ] n test (scratchpad script, logged): sim tracks, chosen q, n_per_dim ∈ {30, 50, 100}; reconstruction-one-period-back px error vs exact. Pick smallest within ~2 px of exact.

### Task 3: `LmuNet` — one step function

**Files:** Create `trajmem/snn_lmu.py`; Test `tests/test_snn_lmu.py`.

**Produces:** `LmuNet(n_pos_in, n_fast, lmu: SpikingLmu, n_horizons, n_cells, n_path, n_short, tau_fast, readout_tau_s, dt_s, seed)`; `.init_state(B, device)`; `.step(pos_cells, vel_cells, seen, u, state) -> (heads (B, n_horizons, 2, n_cells), path (B, n_path), state)` where the fast layer input is `cat[pos_cells, vel_cells, seen] @ w_in + w_ff(spk_f) + w_sf(act_lmu_scaled)`, heads for the first `n_short` horizons read `r_f`, the rest read `cat[r_f, r_lmu]`, path reads `r_lmu`; `r_lmu` = 15 ms low-pass of the LMU's act divided by 100 (Hz → order 1). `.rates` = the fast layer's mean rate over the calls since `reset_rates()`.

- [ ] Test: shapes; state carries across calls; the fast rate is tracked.
- [ ] Implement; commit "LmuNet: fast layer + LMU population + readouts".

### Task 4: `LmuMemory` — streaming with self-feeding, training with blanks

**Files:** Modify `trajmem/snn_lmu.py`; Test `tests/test_snn_lmu.py`.

**Produces:** `LmuMemory(dt_s, q=…, n_per_dim=…, theta_s=4.0, n_per_axis=32, n_vel_cells=16, vel_range=0.08, vel_window_s=0.025, anchor_tau_s=0.02, n_fast=384, tau_fast=(0.01,0.025), readout_tau_s=0.015, rate_target=0.1, rate_weight=10.0, score_tau_s=0.1, resolution=(640,480), horizons_s=HORIZONS_S, seed=0)` implementing `TrajectoryMemory` + `period` + `path_points` + `fit(tracks, …, blank_prob=0.5, path_loss="geometric", path_weight=1.0, …)` + `save/load` (checkpoint dict with `"arch": "lmu"`, config, state_dict of the learned parts, LMU weights regenerated from config+seed at load).

Core: `_run(obs (T,B,2) with NaN, state) -> (cells (T,B,H,2,n_cells), path (T,B,n_path), ref (T,B,2), state)` — the shared loop: per step, `seen = isfinite(obs)`; `pos = obs if seen else imagined` where `imagined` = the 25 ms head's absolute prediction from `k25` steps ago (ring buffer, detached) or the last anchor if none; anchor EMA on `pos`; velocity = anchor − anchor k_vel steps ago; encode; `u = pos − 0.5`; `LmuNet.step`. `observe` calls `_run` on one step; training calls it on chunks. Targets `yh` (offsets from the *anchor*) and `yp` as in `SpikingMemory._arrays`, but `x` is no longer precomputed — the loop encodes.

- [ ] Test: protocol; `predict` NaN before observing; `period()`/`path_points` NaN/None before any path, arrays after.
- [ ] Test: fit on 6 analytic tracks (tiny sizes: q=8, n_per_dim=20, n_fast=48) for 15 epochs lowers val 100 ms px and the path px is finite; then streaming predicts the first track at 100 ms within 25 px (as `test_fit_lowers_the_error…` for the two-layer net).
- [ ] Test: streaming with NaN observations for 0.3 s keeps returning finite predictions and the LMU input was the imagined position (assert `memory.last_u` finite during the blank).
- [ ] Test: checkpoint round trip keeps predictions bit-for-bit.
- [ ] Implement; tests pass; commit "LmuMemory: streaming, self-feeding through blanks, training".

### Task 5: Wiring — `make_memory`, `train_memory.py --arch lmu`, blank evaluation

**Files:** Modify `trajmem/experiment.py`, `scripts/train_memory.py`, `scripts/run_experiment.py`; Test `tests/test_experiment.py`.

- [ ] `make_memory("snn", checkpoint)`: `torch.load` the dict, dispatch on `ck.get("arch", "two_layer")` → `SpikingMemory.load` / `LmuMemory.load`. Test with a saved tiny `LmuMemory`.
- [ ] `evaluate_clip(memory, clip, window_us, horizon_s, blank=None)`: with `blank=(t0, t1)` observations in [t0, t1) become NaN before `observe`; `Trace.blank` keeps the tuple; `score_trace` adds `blank_px` = median error at steps inside the blank (NaN if no blank). Test on the analytic clip with `HarmonicFit` (it skips NaN) — blank_px finite and < 10.
- [ ] `run_experiment.py --blank T0 T1` (single-clip mode), column `blank_px`.
- [ ] `train_memory.py --arch {two_layer,lmu}`, `--blank-prob`, `--lmu-q`, `--lmu-n`; builds the chosen memory; prints the same epoch line.
- [ ] Tests pass; commit "LMU memory wired into training and evaluation".

### Task 6: Runs and bookkeeping

- [ ] Quick run: `train_memory.py --arch lmu --include-locked --epochs 12 --schedule cosine --anchor-tau-s 0.02 --out runs/memory/lmu_quick.pt` (log to `runs/memory/train_lmu_quick.log`); score on development set with `--subtract-offset`; a sim clip with `--blank`.
- [ ] If the path px moves well below 70 on validation: full run with `--augment flips --epochs 40` (background, hours); score at the end.
- [ ] Update `docs/snn-experiments-log.md` (run entries), `docs/snn-experiments-summary.md` (plain-language), `PLAN.md` (Phase C design + numbers), `docs/implementation-plan.md` (module table), `BIBLIOGRAPHY.md` (Voelker & Eliasmith 2018; Voelker, Kajić & Eliasmith 2019), `scripts/README.md`-equivalent in `README.md` if a command changed. Commit.
