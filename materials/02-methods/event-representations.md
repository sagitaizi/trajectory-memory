# Event representations

Raw event: `(x, y, t, p)` — pixel, microsecond timestamp, polarity `p ∈ {−1, +1}`. To feed
frame-based or step-based models you convert a time window of events into a tensor. The
Thesis repo already implements the first two (`pipeline/accumulator.py` B.1,
`pipeline/time_surface.py` B.2).

## Accumulated frame / event image
Over window `[t0, t0+Δt)`, per pixel:
- **count**: number of events (polarity-agnostic).
- **polarity sum**: `Σ p` (can cancel).
- **two-channel**: separate ON and OFF counts.
Output `H×W` (or `H×W×2`). `Δt` trades latency (= window length) against SNR. Typical 5–40 ms.

## Time surface (Surface of Active Events)
Per pixel keep the last event time `t_last(x,y)`; the surface at query time `t` is
```
S(x,y) = exp( −( t − t_last(x,y) ) / τ )        # τ ≈ 5–50 ms
```
Encodes recency → implicitly local motion direction/speed. Often computed per polarity.
Variants: linear decay, or averaged over a neighbourhood (HATS, HOTS).

## Voxel grid
Split the window into `B` temporal bins; **bilinearly split** each event's contribution
between the two nearest bins by its timestamp (and optionally between the 4 nearest pixels).
Output `H×W×B`. Preserves sub-window timing — the usual input to surrogate-gradient SNNs /
CNNs on event data.

## Spike tensor (Stage 2, raw-ish)
Bin at a fine `Δt` (e.g. 1 ms) → sequence of sparse `H×W` (or downsampled) binary/again-count
frames, fed step-by-step to the SNN. This is as "asynchronous" as CPU/GPU frameworks get.

## For this project
- Stage 1 input: accumulated frame and/or time surface.
- Stage 2 input: spike tensor (fine bins), possibly on a spatially downsampled grid.
- Whatever the localiser eats, keep the **window length** a logged hyperparameter — it's the
  latency knob the paper's title cares about.
