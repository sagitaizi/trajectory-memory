# Nengo / LMU cheatsheet

Candidate for the memory core if G-F picks the NEF family. Already in the `thesis` env.
Verify against installed version.

## Nengo core objects
```python
import nengo, numpy as np

with nengo.Network() as model:
    stim = nengo.Node(lambda t: ...)                     # input signal
    ens  = nengo.Ensemble(n_neurons=200, dimensions=2)   # represents a 2-D vector
    out  = nengo.Node(size_in=2)

    nengo.Connection(stim, ens)
    nengo.Connection(ens, out, function=lambda x: x)     # decoded transformation

    # recurrent dynamics: ẋ = f(x) implemented as a synapse-filtered feedback
    nengo.Connection(ens, ens, function=feedback, synapse=0.05)

    p = nengo.Probe(out, synapse=0.01)

with nengo.Simulator(model, dt=0.001) as sim:
    sim.run(2.0)
data = sim.data[p]
```

## Online learning rules (no framework change needed)
- `nengo.PES(learning_rate=...)` — supervised, adjusts decoders from an error signal. Wire an
  error `Node`/Connection into `learning_rule_type=`.
- `nengo.Voja(learning_rate=...)` — unsupervised, moves encoders toward the current input.
- `nengo.BCM`, `nengo.Oja` — Hebbian on weights.
Standard pattern: **Voja (learn what the categories/patterns are) + PES (learn what to do)**.

## LMU / LDN in Nengo
- The `A`, `B` matrices from `02-methods/legendre-memory-unit.md` become a recurrent
  `Connection` on an ensemble representing `m` (dimension `d`), with `synapse = θ` and
  `transform` derived from `A`, `B` (continuous → the connection's synaptic filter does the
  integration). `nengolib`'s `LinearNetwork` / `DiscreteDelay` build this directly if
  installed.
- Simplest robust route: run the **pure-NumPy discretised LDN** as a `nengo.Node`, feed its
  state to an `Ensemble` only if you need the spiking representation for the paper's claim.

## Not available (Thesis repo doc §4, Python 3.14)
`nengo-dl` (gradient training in Nengo) — build fails. `nengo-loihi` — stale, untested on
Nengo 4.x. So: Nengo for the low-D dynamical memory only; no trained conv-SNN here, no Loihi
deployment.
