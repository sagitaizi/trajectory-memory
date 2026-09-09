# Kalman filter with a periodic model

Classical baseline for predicting and monitoring a repetitive trajectory. Run one filter per
spatial coordinate, or a joint 2-D state.

## State options (per coordinate)
**Constant-velocity** (simplest): `s = [p, v]ᵀ`.
```
F = [[1, Δt],
     [0,  1 ]]          Q = process noise (tune)
H = [1, 0]              R = measurement noise
```

**Harmonic** (period known or estimated, `ω = 2π / T`): `s = [p, v, c1, d1, …, cM, dM]ᵀ` with
each harmonic pair rotating:
```
[[c_k],  =  [[ cos(kωΔt),  sin(kωΔt)],   [[c_k],
 [d_k]]      [-sin(kωΔt),  cos(kωΔt)]] ·  [d_k]]
p_model = Σ_k c_k
```
Blend with a slow trend term if the path drifts.

## Standard recursion
```
Predict:  s⁻ = F s ;              P⁻ = F P Fᵀ + Q
Update:   y  = z − H s⁻ ;         S  = H P⁻ Hᵀ + R
          K  = P⁻ Hᵀ S⁻¹
          s  = s⁻ + K y ;         P  = (I − K H) P⁻
```

## Prediction horizon `h`
Iterate the predict step `⌈h/Δt⌉` times from the current estimate (no updates). Read `p`.

## Unknown period
- **EKF**: add `ω` to the state, linearise the harmonic transition w.r.t. `ω`.
- **IMM**: run a bank of filters at candidate periods, mix by posterior model probability.
- Or estimate `T` up front from the autocorrelation / dominant FFT peak of a warm-up segment
  and keep it fixed.

## Deviation score
**Normalised innovation squared**: `ε(t) = y(t)ᵀ S(t)⁻¹ y(t)`. Under the model, `ε ~ χ²` with
`dim(z)` dof. Flag when a smoothed `ε` exceeds the χ² quantile (e.g. 0.99) for a sustained
window. Detection latency = first sustained exceedance after the break.
