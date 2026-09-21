# Spiking localiser — design

Stage 1's first half: a spiking network that turns each accumulated event frame into the
target's position (or "not seen"), replacing the classical dense-cell centroid as the
memory's input. Decided 2026-09-21.

## Why

The classical centroid fails in three ways no rule fixes together: it flips to the brush's
string when the brush is dim; it returns a noise cluster when the target's own events vanish
(turnarounds, slow scans, stuck pixels); and cluster continuity, which fixes the fan, doubles
the error on the multi-part wall target. Finding the target means recognising it — a learning
problem, and the part of the pipeline where a spiking network can do what arithmetic cannot.

## Decisions

- **Input**: the current 5 ms ON/OFF count frame only, 2 × 60 × 80 (8× downsample of the
  640 × 480 sensor), from `frontend.to_frames` — hot-pixel-filtered like `to_position`. No
  memory expectation, no explicit history.
- **Time**: the network runs continuously over the frame stream, one step per frame, state
  carried; leaky membranes (10–20 ms) integrate the last few frames, so sparse frames add up
  and stuck pixels average into background. Output every frame.
- **Output**: 32 place cells over x and 32 over y (position = centre of mass, as the memory's
  heads) plus one "present" unit; below a threshold the localiser returns unseen (NaN).
  Readouts from a 15 ms low-pass of the output spikes.
- **Network** (`SpikingLocaliserNet`, snnTorch, surrogate gradients): conv 2→8 (5×5, stride 2)
  LIF → conv 8→16 (5×5, stride 2) LIF → 15 × 20 × 16 → dense LIF 128 → linear readouts
  (64 + 1). Learnable per-neuron time constants spread over 10–20 ms. Under 100 k weights,
  a few ms per frame on the CPU.
- **Training data**: simulation only (`corpus/sim`, 200 clips, exact truth, string-bright
  clips kept). Augmentation at training time, on the frames: stuck pixels (a random set of
  50–400 pixels firing at 50–1500 events/s, i.e. 0.25–7.5 events per 5 ms window, per
  clip), a noise-floor scale (0.5–3×), left/right and up/down flips (with the truth), and a
  random brightness scale. "Present" label: the target's own event count in the window
  above a floor (from the simulator's per-event provenance where available, else the truth
  on-sensor and the frame's count near it above a floor).
- **Loss**: cross-entropy of each axis's cells against the true position's bump, binary
  cross-entropy on "present" (masked to steps with a defined label), a firing-rate
  regulariser as in the memory. BPTT over 2 s chunks with state carried; checkpoint by total
  validation loss; 10 % of clips validate.
- **Evaluation**: `localise.evaluate_localiser` + `score_track` against the labels of the
  development clips — median and p90 px, offset-subtracted like the memory scorecard — with
  `FrameCentroid` and `frontend.to_position` as the bars (fan 5.4, pendulum 8–12, wall
  target 13–18 px median today, tails up to 79). Then end to end: the memory fed by the
  localiser, `run_experiment.py --localiser snn`, same scorecard as before.
- **Interface**: `SpikingLocaliser` implements `model.Localiser` (`locate(frame)`,
  `reset()`); `experiment.evaluate_clip(..., localiser=...)` streams frames when one is
  given; `run_experiment.py --localiser {centroid,snn}` with `--localiser-checkpoint`;
  `replay_gt.py` draws the localiser's position in place of the centroid.

## Success

Better tails than the centroid on the development clips (fewer flips: p90 down, jumps
> 40 px rarer) at a median no worse; the memory's development numbers with the localiser
no worse than with the centroid; "unseen" reported where the target is not there.

## Out of scope

Raw-event input (Stage 2); training on any real clip; a localiser conditioned on the memory.
