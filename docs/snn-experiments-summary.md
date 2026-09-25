# SNN memory — what was tried, in short

The plain-language companion to `snn-experiments-log.md` (all the numbers). Written for
picking the work up after a break. The sections are dated snapshots; the current state is
in `PLAN.md`.

## The job

The memory network gets the target's position every 5 ms (from the centroid) and has to
say where the target will be 25, 50, 100 and 200 ms later, plus describe the repeating path
(period, phase, shape). It is trained on simulated clips, then frozen and scored on real
ones. The bar: the classical Kalman baseline is 7.5 px at 100 ms on clean sim positions and
24.8 px on the real development clips. A sanity floor: just repeating the current position
("persistence") gives 29 px at 100 ms on sim — a memory that is worse than that has not
learnt motion at all.

## What happened, step by step

1. **First design** (place cells in, two spiking layers, plain x-y readout): 46.5 px at
   100 ms — worse than persistence. Two causes found by probing the trained network:
   - *Position gets blurred on the way out.* The best any readout could recover from the
     spikes was 30 px, although the neurons' internal voltages held it to 15 px. Reading a
     position as a weighted sum of noisy firing rates is imprecise.
   - *No motion.* At 100 ms ahead the network was no better than persistence: it output
     "where the target is now" for every horizon. And the slow layer switched itself off
     during training.
2. **Place cells at the output + a firing-rate regulariser** (a loss term that keeps a
   layer from going silent): 39 px at 100 ms. The precision problem eased (21 px at 25 ms)
   and the slow layer stays alive, but the network still just copies the present.
3. **Velocity at the input + displacement at the output** (the current step, running
   overnight): the network is told how the target moved in the last 25 ms, and predicts the
   *offset* from the current position rather than the position itself — so "do nothing"
   already equals persistence and all learning is about motion. Results below as they come.

## Things learnt that hold regardless

- Ten CPU threads are 3× slower than four for this network; the GPU is slower still
  (the layers are too small to fill it). Training runs on the CPU with 4 threads.
- Every long run took 3–5× longer than it should have: Windows was putting the
  background trainings on the laptop's four *efficiency* cores. Found on the morning of
  the 18th; the memory now pins itself to the performance cores.
- The slow layer dies without the rate regulariser, whatever the initial gain.
- A wider surrogate gradient (slope 5) diverges; no gradient clipping is worse; learning
  rate 3e-3 beats 1e-3 at 30 epochs.
- Faster time constants (fast layer 10–25 ms, readout 15 ms) and a bigger fast layer each
  buy ~10 %.

## Where it stands (morning of 2026-09-17)

Step 3 worked. With velocity in and displacement out, the network predicts 100 ms ahead
to **~15 px on simulated clips** (persistence 29, Kalman 7.5) and is **at the Kalman bar on
the real development clips** (25–30 px pooled vs 24.8; fan clip 9–11 vs 8.3). At the short
horizons it is better than the input it receives (6 px at 25 ms vs 11 px persistence).

What the sweep then showed, in one line each:

- Faster fast-layer and readout time constants: ~1 px better. Kept.
- Bigger layers, more or finer cells, other velocity windows, rate targets, path-head
  weight, chunk length, learning-rate schedules: no change beyond run-to-run noise (±1 px).
- Training longer does not help: the network starts to memorise the 51 training clips after
  ~20 epochs (validation stops improving while training keeps improving). **The remaining
  gap to the Kalman is a data problem, not a network problem.**
- So the last runs of the night add data: mirror-flipped copies of every clip (×4), the 43
  clips the centroid can't track with their true positions standing in, and weight decay.

- More data is the lever that keeps paying. Mirror flips: 13.6 px at 100 ms. Adding the
  43 clips the centroid can't track (their true positions standing in): 12.8. Both
  together: **11.7 px at 100 ms, 22 px at 200 ms** — now the default checkpoint
  (`runs/memory/snn.pt`). Honest figure ±1 px: the validation set is only 6 clips.
  Reproduce it with `scripts/train_memory.py --augment flips --include-locked --epochs 40
  --schedule cosine`.
- Letting the horizon heads also read the slow layer changed nothing — the slow layer is
  not yet holding the cycle. That, not the readout, is where the rest of the gap to the
  Kalman (7.5 px) sits: the Kalman knows the period and shape; the network still mostly
  extrapolates recent motion.

Every run's checkpoint and per-epoch curve is in `runs/memory/sweep/`, the table in
`runs/memory/sweep.csv`; the sweep driver and lane files are in Claude's scratchpad (not
in the repo) — `scripts/train_memory.py` reproduces any single run from its config.

## Where it stopped (2026-09-18, 15:10)

Training was wrapped up with two runs cut short. Best numbers on simulated clips, 100 /
200 ms ahead: **10.2 / 17.7 px** (20 ms anchor smoothing, run stopped at epoch 34 —
`runs/memory/snn_anchor20.pt`); the completed default is 10.7 / 19.2
(`runs/memory/snn.pt`). Against the baselines, offset-subtracted, on the real
development clips: SNN 16.6–17.4 px pooled, Kalman 14.8, Harmonic 24.0. The SNN beats
both on the pendulum and on the hand-moved loop; the Kalman wins the fan clip.

Not yet run (queued, cheap to restart): anchor 40 ms, shift/scale/stretch augmentation,
the no-regulariser twins, LSNN + anchor, and the `train_memory.py` reproduction of the
best run.

## Open for the next session

- Whether the augmentation runs close the gap; if they do, how much more simulated data
  is worth generating (the corpus is 100 clips; the generator is seed-reproducible).
- The path head is still weak (its loss barely moves) and the 200 ms horizon is 30 px:
  the slow layer's memory is not yet doing much. `heads_from="both"` (horizon heads read
  from both layers) is in the sweep as a variation on the decided design — to accept or not.
- ~~The development-set number is dominated by the pendulum label offset.~~ Settled:
  `run_experiment.py --subtract-offset` removes each clip's constant label-vs-centroid gap
  (pendulum 25–42 px, wall target 12–21 px) and reports it. Honest real-clip numbers at
  100 ms, pooled: **Kalman 14.8, SNN 18.3, Harmonic 24.0**; the SNN wins the hand-moved
  `loop_01` by 2× (29 vs 62) and loses the fan (11.5 vs 7.0).
- The SNN is more exposed to centroid jitter than the baselines (its prediction is the raw
  last centroid plus an offset; the baselines filter). A smoothed anchor (`anchor_tau_s`)
  is under test; the real fix is the Stage 1 spiking localiser.
- Doubling the simulated corpus to 200 clips: **10.7 px at 100 ms, 19 px at 200 ms** — the
  new default checkpoint, and the first run that doesn't overfit. More data keeps paying.
- The slow layer became adaptive LIF (LSNN) by decision; on the 100-clip corpus it made
  no difference (12.2 vs 11.7 px) — the slow layer has no training signal that demands
  cycle memory, whichever neuron it uses. A plan to give it one (phase head; long
  horizons routed through it) is written in the log, awaiting a decision. The corpus was extended to 200 sim clips (119 usable vs 57); runs on it are
  queued. A code review found and fixed three bugs (NaN deviation score after an unseen
  first window, CUDA crash in the encoder, epochs sampling with replacement).


## Where it stands (2026-09-19, morning)

**The metric first.** We now score the *path itself*: how far the memory's own picture of
one cycle sits from the true cycle (px), with the period reported separately. It showed
that the learned two-layer network never held the path — its cycle was 64 px off on real
clips (the size of the path), its period twice the truth — and that its decent prediction
numbers came from short-term extrapolation. A better loss (px of the drawn cycle) and
routing the long horizons through the slow layer each moved the path to ~72 px and no
further: the slow layer's only long memory is a spike count, which records where the
target has been but not in what order.

**The new core.** Decided with Sagi: a hand-set window memory — a spiking Legendre Memory
Unit, built by Nengo (encoders, decoders, the fixed recurrent wiring) and simulated in
torch beside the fast layer. It holds the last 4 s of the position by construction
(verified: it reconstructs the position one period back to 0.5 px median on the exact
system, and the spiking population tracks that to 5 %). The readouts and the fast layer
are learned; when the target is unseen the network feeds its own prediction back in.

**First numbers (12 epochs, no augmentation).** Path on real development clips **58 px**
(two-layer 66, Kalman 14); fan 30 (was 77), hand-moved loop 29 (Kalman 49). Prediction
at 100 ms **32 px** — worse than the two-layer's 17 and the Kalman's 14.7. Through a 1 s
blank the network drifts (52–65 px). So: the memory now clearly holds more of the path
than the learned slow layer ever did, and it is not yet near the baselines on either
number. The night's goal — beating the baselines across the board — was not reached.

**What limits the path, measured (midday).** A head fitted offline on a *perfect*
window: linear 61 px, small MLP 34 px. The same MLP on the *spiking* window: 46 px at
best. The classical period search + harmonic fit on the window reconstructed from a
perfect state: **5.9 px** — that is the harmonic baseline living inside the memory — but
on the spiking state it collapses (74–86 px): each state dimension carries a 2–4 %
systematic error, and rebuilding a 4 s window weights all 24 Legendre orders equally, so
that error becomes ~90 px at every lag. Time-averaging does not help (not fast noise);
neither do 4× more neurons nor the discrete-time NEF mapping. **The fidelity of the
spiking window is now the binding constraint**, and the readouts (linear, MLP, with or
without training blanks, in-loop or offline) all land at 54–59 px because of it. Four LMU
runs (`lmu_quick`, `lmu_mlp`, `lmu_noblank`, and run 9 stopped early) agree.

**Next, in order.** (1) A representation whose reconstruction tolerates the neurons'
error: shorter window with fewer orders (θ 2 s, q 12), or a bank of short windows read
for period, or 2-D ensembles per pair of orders — a design question for Sagi. (2) Get
prediction back with the two-layer recipe (flips, 40 epochs): every LMU run so far is 12
epochs without augmentation, so 32 px vs 17 is not yet a like-for-like comparison. (3)
Through blanks, feed the window's own "one cycle ago" position instead of the 25 ms head.


## Where it stands (2026-09-20, afternoon) — the architecture changed

**Why.** Every memory so far tried to store a *recording* of the last few seconds and read
"period, shape, phase" off it. The Kalman never does that: it keeps a clock hand and a
drawing, and corrects both a little every step. We built that in neurons.

**The clock-and-map memory.** Five clocks (spiking populations whose wiring makes them
oscillate and lets the input pull their phase and rate — Righetti's adaptive-frequency
oscillator) race from different starting rates; the one whose rate matches the motion's
rhythm is elected. Each clock drives a ring of phase cells; the connections from the ring
to a position readout are the map — the memory of the path — learned during the clip by a
local rule. Prediction reads the map a little ahead on the ring; the period is the clock's
rate; the shape is the map; a deviation is a mismatch against a snapshot of the map taken
once the memory has locked. Nothing is pretrained. First checked in plain arithmetic
(`phasemap.py`), then built in spiking neurons (`snn_phasemap.py`): Nengo solves the
wiring, torch runs it, as with the LMU.

**Numbers** (development set, offset-subtracted, against the Kalman): prediction 14.3 vs
14.7 px (a tie per clip: slightly behind on six clips, far ahead on two), **path 12.4 vs
14.0** (better on five of eight), **break AUC 0.99 vs 0.80** with latency 0.11 s vs 0.95
and fewer false alarms. The prototype in arithmetic is about the same. This is the first
memory of the project that holds the cycle; the previous best spiking path was 55 px.

**Known weak cases.** The diagonal sweep locks at half the period (both versions); a 3:2
Lissajous has no energy at the fundamental, so nothing drives the clock at the true period.
The rate lives outside the population as a slow scalar because a neural integrator drifted.

**Next.** (1) The half-period lock: a second, slower pull toward map consistency once a map
exists, or a harmonic check in the election. (2) A gradient-trained fast layer for the
25–50 ms detail on top of the map (snnTorch; the two-layer network's fast layer reached
5.6 px at 25 ms). (3) The held-out set, scored once. (4) The framework statement for the
paper: NEF-built populations (Nengo at construction) simulated in torch, a local PES rule
for the memory, snnTorch for any gradient-trained stage.


## Where it stands (2026-09-22, morning) — the localiser

**What it is.** The other half of Stage 1: a spiking network that looks at each 5 ms event
frame and says where the target is — or that it isn't there — replacing the classical
"densest cells" centroid that flipped to the brush's string and to noise clusters. A small
spiking convolutional net (snnTorch), running continuously over the frames so a few frames
add up; trained on the simulated clips only, with the real cameras' quirks imitated (stuck
pixels, a noise floor) and stretches where the target is removed so it learns to say "not
seen". Nothing real was trained on.

**What it took.** Sparse frames leave a default-initialised spiking net silent, and a silent
spiking net cannot learn; the fix was to calibrate each layer's gain on a sample of frames
before training, regularise only the dense layer's rate, and backprop over short chunks
(0.25 s). A wider net at a lower learning rate then learned steadily: **6.7 px** on
simulated validation, against ~11 for the classical centroid.

**On the real clips** (against the labels, constant offset removed): on the **fan** — the
setup the simulator was matched to — it does the job it was built for: the string flips
are gone (1.5 % → 0), the worst errors shrink (p90 15 → 13 px), the median is a little
worse (5.2 → 7.9). Feeding it to the memory on the fan gives prediction 6.6 px and path
4.7 px, versus 7.3 / 6.6 from the centroid and 7.0 / 6.8 for the Kalman — the first place
the all-spiking pipeline beats the classical one on both numbers. On the **pendulum** it
removes the jumps but marks a different point of the brush, less precisely (median 23 vs
8–22). On the **wall target** it fails: that target is a small rectangle on a large
hand-held sheet whose outline also makes events — nothing like the simulator's blobs — and
the network mostly says "not there".

**What that means.** The localiser works where the simulator resembles reality and not
where it doesn't; the gap is simulator content, not the network. Next: a "sheet" target
family in the simulator (a rectangle outline with a small rectangle inside) and a retrain;
until then the classical centroid stays the memory's default input, with the spiking
localiser as the fan's better alternative.


## Where it stands (2026-09-23, morning) — the localiser after the wall-target work

**What was wrong.** The spiking localiser trailed the wall target by ~200 ms. Not the
neurons' time constants, not the target's faintness (each tested alone): the real
camera's ON/OFF polarity is the reverse of the simulator's, and on an outline-only
target the network had learned "the leading edge is ON" — so it sat on the trailing edge.
Swapping the channels of a real clip flipped the lag from +240 to −160 ms.

**What changed.** Training now swaps ON and OFF on half the clips, turns a quarter of
them by 90° (the real target is sometimes held tall; every simulated sheet was wide),
and sees 100 sheet targets among 300 clips. The lag is gone (+35 ms everywhere — the
neurons' own delay), `loop_01` is on par with the classical centroid (15.8 vs 16.2 px),
and `loop_break_01` halved (40 → 22 px). A polarity-blind input (ON + OFF summed) was
tried and is worse: the two edges are a cue to the centre.

**Where it is not enough.** On the pendulum the network's estimate drifts slowly along
the brush (20–26 px against the centroid's 8–12), and end to end the classical centroid
still feeds the memory better (pooled 13.5 / 12.9 px vs 17.7 / 24.4). The drift is not
jitter (smoothing does nothing); it is where the network puts the "centre" of an
elongated shape. Next candidate: an elongated head-and-tail target family in the
simulator. Until then the centroid stays the default input and the spiking localiser is
the ablation, with its specific wins (no string flips on the fan; the wall target
tracked without lag). A visual bench (`scripts/bench.py`) renders every iteration on a
fixed clip list, three pipelines side by side.
