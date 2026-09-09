# Sequence memory & dynamical prediction in spiking networks

A trajectory is a low-dimensional curve evolving in time. All of the below can represent /
predict such a curve; they differ in how learning happens and how "online" it is.

## Legendre Memory Unit (LMU) — Voelker, Kajić & Eliasmith 2019 (NeurIPS)
- Built from the **Legendre Delay Network (LDN)**: a linear system `ṁ = A m + B u` whose
  `d`-dim state `m` holds the last `θ` seconds of the scalar input `u`, projected onto shifted
  Legendre polynomials. `A`, `B` come from a Padé approximant of a pure delay.
- Any function of the recent history (including "value `h` seconds from now" for a periodic
  signal) is a **linear readout** of `m`.
- Beats LSTM on chaotic time-series prediction; handles windows of 10^5 steps; few state
  variables. Runs on Loihi.
- For us: hand `m` a position stream, learn a linear (or small MLP) map `m → x(t+h)`.
  Pure-NumPy LDN is ~20 lines — see `02-methods/legendre-memory-unit.md`.

## FORCE — Sussillo & Abbott 2009 (Neuron)
- A **chaotic recurrent network** (rate or spiking) is tamed by training an output weight
  vector (and feeding the output back) with **recursive least squares (RLS)**, online, until
  the feedback suppresses the chaos and the network autonomously produces the target signal.
- Learns arbitrary targets including **periodic** patterns; the period lives on a low-D
  manifold of network states.
- Online by construction — the RLS update runs while the network runs.

## Spiking FORCE — Nicola & Clopath 2017 (Nature Communications)
FORCE carried into networks of spiking neurons; reproduces and predicts oscillatory /
periodic signals and can encode multiple patterns. The concrete "spiking reservoir that
learns a trajectory online" reference.

## e-prop — Bellec et al. 2020 (Nature Communications)
- Online, **local** approximation to backprop-through-time for recurrent SNNs, via
  per-synapse **eligibility traces** × a learning signal. No storing the full unrolled
  network.
- Runs on SpiNNaker 2. The tool if you want genuine online weight updates in a trained
  recurrent SNN rather than pretrain-then-freeze.

## FOLLOW — Gilra & Gerstner 2017 (eLife)
Local plasticity rule that makes a recurrent spiking network learn an arbitrary
**dynamical system** `ẋ = f(x, u)` from observing it. Directly relevant to "represent the
trajectory as network dynamics".

## Learning recurrent dynamics — Kim & Chow 2018 (eLife)
RLS-trained recurrent spiking networks that reproduce target dynamics stably; method
reference for the reservoir readout / internal training.

## Echo State Network (ESN) — the plain baseline
Fixed random recurrent reservoir, train only a linear readout by ridge regression (offline)
or RLS (online). Cheapest thing that works for temporal prediction. Equations and
hyperparameters in `02-methods/reservoir-computing.md`.

## Which to reach for
| Need | Fit |
|---|---|
| Fastest working result, online readout | ESN + RLS, or spiking FORCE |
| Strong provable memory, Loihi story, stays in Nengo | LMU |
| Trained recurrent SNN with real online weight updates | e-prop |
| Trajectory as an explicit learned dynamical system | FOLLOW / Kim&Chow |
