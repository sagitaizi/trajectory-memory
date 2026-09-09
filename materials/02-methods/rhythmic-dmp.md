# Rhythmic (periodic) Dynamic Movement Primitive

Strongest classical baseline for "learn a repetitive path from one demonstration, reproduce
it, notice when reality diverges". One DMP per coordinate.

Refs: Ijspeert, Nakanishi & Schaal 2002; Saveriano et al., *IJRR* 2023 (tutorial survey).

## Phase
A limit-cycle oscillator supplies phase `φ` and amplitude `r`:
```
φ̇ = Ω                      # Ω = 2π / T, the rhythm frequency
```
`φ` wraps on `[0, 2π)`.

## Transformation system (the shaped oscillator)
```
τ ż = α_z ( β_z (g − y) − z ) + f(φ)
τ ẏ = z
```
`y` = generated position, `g` = baseline (mean of the cycle), `α_z`, `β_z` constants with
`β_z = α_z / 4` (critically damped), `τ = 1/Ω`.

## Forcing term (what's learned)
```
f(φ) = ( Σ_i ψ_i(φ) w_i / Σ_i ψ_i(φ) ) · r
ψ_i(φ) = exp( h_i ( cos(φ − c_i) − 1 ) )        # von Mises basis, centres c_i on [0, 2π)
```

## Learning the weights
From one demonstrated period `y_demo(t)` (and its `ẏ`, `ÿ`):
```
f_target(φ) = τ² ÿ_demo − α_z ( β_z (g − y_demo) − τ ẏ_demo )
```
Solve each `w_i` by **locally weighted regression**:
```
w_i = Σ_t ψ_i(φ_t) f_target(t) / Σ_t ψ_i(φ_t)     # (with the r, φ weighting; see refs)
```

## Use
- **Reproduce / predict**: integrate the system forward; `y(t+h)` is the prediction.
- **Modulate**: change `Ω` (tempo) or `r` (amplitude) online without relearning.
- **Deviation**: residual between observed position and the DMP rollout; threshold like the
  Kalman NIS.
