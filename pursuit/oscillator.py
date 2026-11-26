"""A bank of spiking oscillators with adaptive rates, each a 3-D LIF population (Nengo-built,
torch-run): a damped oscillator that keeps its swing only while the target drives it.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from trajmem.lmu import LmuWeights, SpikingLmu, build_dynamics
from trajmem.snn_phasemap import F_SCALE, OMEGA_HI, OMEGA_LO, OMEGA_MID, OMEGA_SPAN, clock_dynamics, omega_to_w

PERIODS_S = (0.8, 1.3, 2.0, 3.0, 4.5, 7.0)
TWO_PI = 2.0 * np.pi


@lru_cache(maxsize=8)
def _damped_weights(n: int, damping_cycles: float, tau: float, seed: int, radius: float = 1.4) -> LmuWeights:
    """State (x, y, rate); the drive enters x as an input. Undriven, the swing falls by e
    in `damping_cycles` of the clock's own cycles."""
    import nengo

    def f(s):
        x, y, w = s
        omega = OMEGA_MID + OMEGA_SPAN * w
        decay = omega / (TWO_PI * damping_cycles)
        return [tau * (-decay * x - omega * y) + x, tau * (-decay * y + omega * x) + y, 0.0]

    with nengo.Network(seed=seed) as net:
        ens = nengo.Ensemble(n, 3, radius=radius, neuron_type=nengo.LIF(), seed=seed)
        out = nengo.Node(size_in=3)
        identity = nengo.Connection(ens, out, synapse=None)
        recurrent = nengo.Connection(ens, out, function=f, synapse=None)
    with nengo.Simulator(net, progress_bar=False) as sim:
        t = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)  # noqa: E731
        return LmuWeights(scaled_encoders=t(sim.data[ens].scaled_encoders), bias=t(sim.data[ens].bias),
                          decoders=t(sim.data[identity].weights), w_rec=t(sim.data[recurrent].weights),
                          w_in=t(np.eye(3)), q=3, theta=0.0, tau_syn=tau, n_channels=1)


class OscillatorBank:
    """Clocks that differ in their starting period. `step(drive)` takes the target's position
    along its main direction, in units of its amplitude (NaN = unseen). `damping_cycles`
    None is the MCSoC clock, which keeps its swing by itself."""

    def __init__(self, dt_s: float, periods_s=PERIODS_S, damping_cycles: float | None = 1.0, n: int = 3000,
                 tau_syn_s: float = 0.05, k_rate: float = 0.6, kick_s: float = 0.1, seed: int = 1, device=None):
        self.dt_s, self.periods_s, self.damping_cycles = dt_s, tuple(periods_s), damping_cycles
        self.k_rate, self.kick_s, self.tau_syn_s = k_rate, kick_s, tau_syn_s
        self.device = torch.device(device or "cpu")
        torch.set_num_threads(min(2, torch.get_num_threads()))      # more threads run slower at this size
        if damping_cycles is None:
            weights = build_dynamics(1500, 4, 1.7, clock_dynamics(0.02, 2.0, 0.3), in_dims=4, tau_syn=0.02, seed=seed)
        else:
            weights = _damped_weights(n, damping_cycles, tau_syn_s, seed)
        self.pop = SpikingLmu(weights, dt_s, device=self.device)
        self.reset()

    def reset(self) -> None:
        self.t = 0.0
        self.state = self.pop.init_state(len(self.periods_s))
        self.omega = np.array([TWO_PI / T for T in self.periods_s])
        self.xy, self.driven = np.zeros((len(self.periods_s), 2)), False

    @property
    def swing(self) -> np.ndarray:
        return np.hypot(self.xy[:, 0], self.xy[:, 1])

    @property
    def phase(self) -> np.ndarray:
        return np.arctan2(self.xy[:, 1], self.xy[:, 0])

    def ahead(self, seconds: float) -> np.ndarray:
        """Each clock's (x, y) that much later, by its own dynamics: pushed by a target that
        moves as the clock expects, or by nothing if the target is unseen."""
        x, y = self.xy.T
        w = self.omega
        d = np.zeros_like(w) if self.damping_cycles is None else w / (TWO_PI * self.damping_cycles)
        if self.driven and self.damping_cycles is not None:
            turn = np.sqrt(np.maximum(w * w - d * d, 1e-9))
            c, s = np.cos(turn * seconds), np.sin(turn * seconds) / turn
            return np.column_stack([c * x + s * (d * x - w * y), c * y + s * (w * x - d * y)])
        c, s, fade = np.cos(w * seconds), np.sin(w * seconds), np.exp(-d * seconds)
        return np.column_stack([fade * (c * x - s * y), fade * (s * x + c * y)])

    def step(self, drive: float) -> None:
        self.t += self.dt_s
        self.driven = bool(np.isfinite(drive))
        F = float(np.clip(drive, -1.5, 1.5)) if np.isfinite(drive) else 0.0
        kick = 1.0 if self.t < self.kick_s else 0.0
        rate = [omega_to_w(o) for o in self.omega]
        B = len(self.omega)
        if self.damping_cycles is None:
            u = np.column_stack([np.full(B, kick), np.zeros(B), rate, np.full(B, F_SCALE * F)])
        else:
            push = self.tau_syn_s * self.omega / (np.pi * self.damping_cycles) * F     # unit swing at unit drive
            u = np.column_stack([kick + push, np.zeros(B), rate])
        with torch.no_grad():
            act, self.state = self.pop.step(torch.tensor(u, dtype=torch.float32, device=self.device), self.state)
            self.xy = self.pop.decode(act).reshape(B, -1)[:, :2].cpu().numpy()
        y = np.clip(self.xy[:, 1], -1.2, 1.2)
        self.omega = np.clip(self.omega - self.k_rate * self.omega * F * y * self.dt_s, OMEGA_LO, OMEGA_HI)
