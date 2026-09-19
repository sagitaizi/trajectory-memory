"""Legendre Memory Unit: a fixed linear system whose state holds the last `theta` seconds
of its input (Voelker, Kajic & Eliasmith 2019). `ExactLmu` is the reference (non-spiking);
`build_population` has Nengo wire a population of LIF neurons that implements the same
dynamics (Voelker & Eliasmith 2018), and `SpikingLmu` runs those neurons in torch.
"""
from __future__ import annotations

from math import comb

import numpy as np
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
    r = np.atleast_1d(np.asarray(lag_s, dtype=float)) / theta
    out = np.empty((len(r), q))
    for i in range(q):
        out[:, i] = (-1) ** i * sum(comb(i, k) * comb(i + k, k) * (-r) ** k for k in range(i + 1))
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
