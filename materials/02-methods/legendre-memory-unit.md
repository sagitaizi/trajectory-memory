# Legendre Memory Unit / Legendre Delay Network

## The idea
A small linear ODE system whose `d`-dimensional state `m(t)` is an orthogonal (Legendre-basis)
compression of the last `θ` seconds of a scalar signal `u(t)`. Any function of that window —
including "the value `h` seconds ahead" for a periodic signal — is a linear (or small
nonlinear) readout of `m`.

## Continuous system
```
θ ṁ(t) = A m(t) + B u(t)
```
with, for `i, j = 0 … d−1`:
```
A_ij = (2i + 1) * ( -1            if i <  j
                     (-1)^(i-j+1) if i >= j )
B_i  = (2i + 1) * (-1)^i
```
(This is the Padé-approximant realisation of a pure delay `θ`.)

## Discretise (step Δt)
Euler: `m(t+Δt) = m(t) + (Δt/θ) (A m(t) + B u(t))`.
Better: zero-order hold — `Ā = expm(A Δt/θ)`, `B̄ = A⁻¹ (Ā − I) B`; then
`m(t+Δt) = Ā m(t) + B̄ u(t)`.

## Reading it back
The window `u(t − θ')`, `θ' ∈ [0, θ]`, is recovered as `Σ_i m_i(t) P_i(2θ'/θ − 1)` with `P_i`
the Legendre polynomials. In practice: **learn** a map `g(m(t)) → x(t+h)` by ridge regression
(linear) or a tiny MLP, from the pretraining trajectories. For 2-D position, run one LDN per
coordinate (or a vector LDN).

## LMU cell (if you want the full RNN, not just the LDN memory)
```
u_t   = e_x·x_t + e_h·h_{t-1} + e_m·m_{t-1}        # scalar input to the memory
m_t   = Ā m_{t-1} + B̄ u_t
h_t   = f( W_x x_t + W_h h_{t-1} + W_m m_t )       # nonlinear hidden state
```
Train `e_*`, `W_*` by BPTT (offline). The `A`, `B` are fixed.

## Why it's a candidate here
Provably optimal window memory, strong on chaotic-series prediction, small state, runs on
Loihi, and it's the natural NEF-family choice. Pure-NumPy LDN is the snippet above — no
dependency. Nengo builds it too (see `03-tooling/nengo-lmu-cheatsheet.md`).

Ref: Voelker, Kajić & Eliasmith, "Legendre Memory Units", NeurIPS 2019.
