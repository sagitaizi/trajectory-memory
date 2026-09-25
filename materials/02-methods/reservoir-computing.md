# Reservoir computing (ESN / LSM / FORCE)

Background only: an early memory-core candidate, not used (the memory is the clock and map
in `PLAN.md`). A fixed recurrent network provides a rich temporal state; only a
readout is trained. Spiking variant = Liquid State Machine.

## Echo State Network — update equations
Reservoir state `x ∈ R^N`, input `u ∈ R^K`, output `y ∈ R^L`.

```
x_tilde(t) = tanh( W_in u(t) + W x(t-1) + W_fb y(t-1) )
x(t)       = (1 - α) x(t-1) + α x_tilde(t)          # leaky integration, α ∈ (0, 1]
y(t)       = W_out [1; u(t); x(t)]
```

## Building the reservoir
- `W`: sparse random (e.g. 1–10% connectivity), then **rescale so spectral radius ρ(W) ≈
  0.8–1.2**. ρ near 1 = long memory; > 1 risks instability (unless FORCE tames it).
- `W_in`: dense random, scale controls how strongly input drives the reservoir (input
  scaling, e.g. 0.1–1.0). Larger = more nonlinear, shorter memory.
- `W_fb`: only if using output feedback (needed for autonomous generation / long-horizon
  prediction). Scale small (0.01–0.5).
- `N`: 200–2000 to start.
- `α` (leak rate): match to the timescale of the trajectory — smaller α = slower reservoir.

## Training the readout
**Offline, ridge regression** (collect states over the pretraining clips after a washout of
~100–1000 steps):
```
W_out = Y_target Xᵀ ( X Xᵀ + β I )⁻¹        # X columns = [1; u(t); x(t)], β ≈ 1e-6…1e-2
```
**Online, RLS** (updates every step, matches "adapts to the object in front of it"):
```
k     = P x / (λ + xᵀ P x)
W_out += (y_target - y) kᵀ
P      = (P - k xᵀ P) / λ                     # λ ≈ 0.999–1.0 forgetting factor
```
Init `P = I / δ`, `δ` small.

## FORCE
Same as online RLS **but with output feedback active during training** — the feedback of the
(initially wrong, then rapidly correct) output suppresses the reservoir's chaos. Train until
`W_out` stops changing, then it runs autonomously. Works for rate and spiking reservoirs.

## Prediction modes
- **Teacher-forced**: feed the true `u(t)`, predict `x(t+h)` — evaluate short-horizon
  accuracy.
- **Generative**: feed the network's own output back — needed for lock-on / long-horizon and
  for a clean deviation signal (error accumulates when the world stops matching).

## Deviation score
Run generative; `e(t) = ||y(t) − observed(t)||`; z-score `e` against its on-pattern running
mean/std, or CUSUM. Threshold → detection.

## Hyperparameter search (week-1 bake-off)
Grid/random over: `N`, ρ(W), input scaling, `α`, `β` (or RLS `λ`), washout. Score on
simulated held-out trajectories with `02-methods/metrics.md`.
