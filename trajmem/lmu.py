"""Legendre Memory Unit: a fixed linear system whose state holds the last `theta` seconds
of its input (Voelker, Kajic & Eliasmith 2019). `ExactLmu` is the reference (non-spiking);
`build_population` has Nengo wire a population of LIF neurons that implements the same
dynamics (Voelker & Eliasmith 2018), and `SpikingLmu` runs those neurons in torch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.linalg import expm


def legendre_matrices(q: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Continuous-time (A, B) of theta * dm/dt = A m + B u, already divided by theta."""
    i = np.arange(q)[:, None]
    j = np.arange(q)[None, :]
    a = np.where(i < j, -1.0, (-1.0) ** (i - j + 1)) * (2 * i + 1)
    b = ((-1.0) ** i * (2 * i + 1)).astype(float)[:, 0]
    return a / theta, b / theta


def delay_readout(q: int, theta: float, lag_s) -> np.ndarray:
    """Weights that read u(t - lag) off the state: shifted Legendre polynomials at lag/theta.
    lag_s scalar -> (q,); array (L,) -> (L, q)."""
    x = 2.0 * np.atleast_1d(np.asarray(lag_s, dtype=float)) / theta - 1.0
    out = np.empty((len(x), q))
    out[:, 0] = 1.0
    if q > 1:
        out[:, 1] = x
    for n in range(1, q - 1):                       # Bonnet recurrence: stable at high order
        out[:, n + 1] = ((2 * n + 1) * x * out[:, n] - n * out[:, n - 1]) / (n + 1)
    return out if np.ndim(lag_s) else out[0]


class ExactLmu:
    """The LMU as a discrete linear recurrence (zero-order hold), one state per channel."""

    def __init__(self, q: int, theta: float, dt: float, n_channels: int = 2):
        a, b = legendre_matrices(q, theta)
        m = np.zeros((q + 1, q + 1))
        m[:q, :q], m[:q, q] = a * dt, b * dt
        e = expm(m)
        self.ad, self.bd = e[:q, :q], e[:q, q]
        self.q, self.theta, self.n_channels = q, theta, n_channels

    def init_state(self, batch: int = 1) -> np.ndarray:
        return np.zeros((batch, self.n_channels, self.q))

    def step(self, u, state) -> np.ndarray:
        """u: (batch, n_channels) -> new state (batch, n_channels, q)."""
        return state @ self.ad.T + np.asarray(u, dtype=float)[..., None] * self.bd

    def reconstruct(self, state, lag_s) -> np.ndarray:
        """Input `lag_s` back, per channel: (batch, n_channels)."""
        return state @ delay_readout(self.q, self.theta, lag_s).T


# --- spiking population, built by Nengo, run in torch ----------------------------------

@dataclass
class LmuWeights:
    """Everything Nengo decided for the population; tensors, nothing learned."""
    scaled_encoders: torch.Tensor      # (N, D)  gain * encoder / radius
    bias: torch.Tensor                 # (N,)
    decoders: torch.Tensor             # (D, N)
    w_rec: torch.Tensor                # (D, N)  (tau A + I) @ decoders
    w_in: torch.Tensor                 # (D, C)  tau B, block per channel
    q: int
    theta: float
    tau_syn: float
    n_channels: int


def build_population(q: int, theta: float, n_per_dim: int, radii, tau_syn: float = 0.02,
                     n_channels: int = 2, seed: int = 0) -> LmuWeights:
    """One 1-D LIF ensemble per state dimension (Nengo solves encoders, gains, biases,
    decoders); the LMU dynamics become the recurrent transform (NEF principle 3 with a
    first-order synapse). `radii` (D,) is what each dimension must represent."""
    import nengo

    a, b = legendre_matrices(q, theta)
    d = n_channels * q
    a_big = np.kron(np.eye(n_channels), a)
    b_big = np.kron(np.eye(n_channels), b[:, None])
    radii = np.broadcast_to(np.asarray(radii, dtype=float), (d,))
    with nengo.Network(seed=seed) as net:
        gather = nengo.Node(size_in=d)
        ensembles, conns = [], []
        for i in range(d):
            ens = nengo.Ensemble(n_per_dim, 1, radius=float(radii[i]), neuron_type=nengo.LIF(), seed=seed * 1000 + i)
            conns.append(nengo.Connection(ens, gather[i], synapse=None))
            ensembles.append(ens)
    with nengo.Simulator(net, progress_bar=False) as sim:
        enc = np.zeros((d * n_per_dim, d))
        bias, dec = [], np.zeros((d, d * n_per_dim))
        for i, (ens, conn) in enumerate(zip(ensembles, conns)):
            rows = slice(i * n_per_dim, (i + 1) * n_per_dim)
            enc[rows, i] = sim.data[ens].scaled_encoders[:, 0]
            bias.append(sim.data[ens].bias)
            dec[i, rows] = sim.data[conn].weights[0]
    t = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)  # noqa: E731
    return LmuWeights(scaled_encoders=t(enc), bias=t(np.concatenate(bias)), decoders=t(dec),
                      w_rec=t((tau_syn * a_big + np.eye(d)) @ dec), w_in=t(tau_syn * b_big),
                      q=q, theta=theta, tau_syn=tau_syn, n_channels=n_channels)


class SpikingLmu:
    """The population in time. `step(u, state)` advances one network step of `dt` in
    `substeps` LIF steps (Nengo's LIF, tau_rc 20 ms, tau_ref 2 ms) and returns the
    synapse-filtered spike rates (B, N) in Hz -- what readouts see -- and the state."""

    TAU_RC, TAU_REF = 0.02, 0.002

    def __init__(self, weights: LmuWeights, dt: float = 0.005, substeps: int = 5, device=None):
        self.w = weights
        self.dt_sub = dt / substeps
        self.substeps = substeps
        self.n = weights.bias.shape[0]
        self.decay = math.exp(-self.dt_sub / weights.tau_syn)
        self.to(device or "cpu")

    def to(self, device):
        self.device = torch.device(device)
        for k, v in vars(self.w).items():
            if isinstance(v, torch.Tensor):
                setattr(self.w, k, v.to(self.device))
        return self

    def init_state(self, batch: int):
        z = lambda *s: torch.zeros(batch, *s, device=self.device)  # noqa: E731
        return (z(self.n), z(self.n), z(self.n), z(self.w.w_in.shape[0]))   # voltage, refractory, act, filtered input

    def step(self, u: torch.Tensor, state):
        """u: (B, C) input value, held over the substeps."""
        v, r, act, x_in = state
        w = self.w
        drive = u.to(self.device) @ w.w_in.T                                    # (B, D), tau B u
        for _ in range(self.substeps):
            x_in = self.decay * x_in + (1.0 - self.decay) * drive
            x = act @ w.w_rec.T + x_in                                          # represented-space input
            j = x @ w.scaled_encoders.T + w.bias
            spk, v, r = self._lif(j, v, r)
            act = self.decay * act + (1.0 - self.decay) * spk
        return act, (v, r, act, x_in)

    def _lif(self, j, v, r):
        dt, tau = self.dt_sub, self.TAU_RC
        r = r - dt
        delta = (dt - r).clamp(0.0, dt)
        v = v - (j - v) * torch.expm1(-delta / tau)
        spiked = v > 1.0
        t_spike = dt + tau * torch.log1p(-((v - 1.0) / (j - 1.0)).clamp(max=0.999))
        v = torch.where(spiked, torch.zeros_like(v), v.clamp(min=0.0))
        r = torch.where(spiked, self.TAU_REF + t_spike, r)
        return spiked.to(v.dtype) / dt, v, r

    def decode(self, act: torch.Tensor) -> torch.Tensor:
        """Filtered rates (B, N) -> represented state (B, C, q)."""
        return (act @ self.w.decoders.T).reshape(act.shape[0], self.w.n_channels, self.w.q)


def state_radii(positions, q: int, theta: float, dt: float, margin: float = 1.25) -> np.ndarray:
    """What each state dimension must represent: the 99th percentile of |m| over the exact
    LMU run on `positions` (list of (T, 2) arrays, already centred), times a margin. (D,)"""
    lmu = ExactLmu(q, theta, dt, n_channels=2)
    peaks = []
    for u in positions:
        st = lmu.init_state(1)
        for i in range(len(u)):
            st = lmu.step(u[None, i], st)
            if i % 10 == 0:
                peaks.append(np.abs(st[0]).reshape(-1))
    return np.maximum(np.percentile(np.array(peaks), 99, axis=0) * margin, 0.05)


def build_dynamics(n: int, dims: int, radius: float, fn, in_dims: int, tau_syn: float = 0.02,
                   seed: int = 0, intercepts=None) -> LmuWeights:
    """One `dims`-D LIF ensemble whose recurrent connection computes `fn(state)` (already in
    NEF form: tau * f(s) + s for the dynamic dimensions), with `in_dims` inputs entering the
    last `in_dims` dimensions directly. Same tensors as `build_population`, so `SpikingLmu`
    runs it; `decoders` read the identity."""
    import nengo

    with nengo.Network(seed=seed) as net:
        ens = nengo.Ensemble(n, dims, radius=radius, neuron_type=nengo.LIF(), seed=seed,
                             **({} if intercepts is None else {"intercepts": intercepts}))
        out = nengo.Node(size_in=dims)
        c_id = nengo.Connection(ens, out, synapse=None)
        c_fn = nengo.Connection(ens, out, function=fn, synapse=None)
    with nengo.Simulator(net, progress_bar=False) as sim:
        enc, bias = sim.data[ens].scaled_encoders, sim.data[ens].bias
        dec, w_rec = sim.data[c_id].weights, sim.data[c_fn].weights
    w_in = np.zeros((dims, in_dims))
    w_in[dims - in_dims:, :] = np.eye(in_dims)
    t = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)  # noqa: E731
    return LmuWeights(scaled_encoders=t(enc), bias=t(bias), decoders=t(dec), w_rec=t(w_rec), w_in=t(w_in),
                      q=dims, theta=0.0, tau_syn=tau_syn, n_channels=1)
