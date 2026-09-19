"""Stage-1 trajectory memory: a two-timescale recurrent spiking network (snnTorch).

Position and its recent change -> place cells -> fast recurrent LIF layer <-> slow
recurrent LIF layer. Linear readouts from low-passed spikes: horizon heads from the fast
layer, as place cells over the *displacement* from the current position (centre of mass,
added to the position), and a path head (period, phase, harmonics) from the slow layer.
Trained offline on simulated tracks by surrogate-gradient BPTT, then frozen. Design and
rationale: PLAN.md section C, materials/01-literature/multi-timescale-memory.md.
"""
from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import numpy as np
import snntorch as snn
import torch
from snntorch import surrogate
from torch import nn

from .data import Track
from .trajectories import _harmonic_fit

N_HARMONICS = 3
N_PATH = 3 + 2 * (1 + 2 * N_HARMONICS)          # period, cos/sin phase, then per axis: mean + 3 harmonics
HORIZONS_S = (0.025, 0.05, 0.1, 0.2)
SHAPE_PHASES = 16                                # where the geometric path loss samples the cycle
SHAPE_PX_SCALE = 50.0                            # px of shape error that count as one unit of loss


# --- input code ------------------------------------------------------------------

class PlaceCells:
    """`n_per_axis` Gaussian tuning curves along x and along y over [lo, hi]; a value drives
    the few cells whose centres are nearest. Unseen (NaN) values drive nothing."""

    def __init__(self, n_per_axis: int = 32, lo: float = 0.0, hi: float = 1.0, width: float | None = None):
        self.n = n_per_axis
        self.centres = torch.linspace(lo, hi, n_per_axis)
        self.width = width if width is not None else (hi - lo) / (n_per_axis - 1)

    @property
    def n_out(self) -> int:
        return 2 * self.n

    def bumps(self, xy) -> torch.Tensor:
        """(..., 2) -> (..., 2, n): each axis's tuning-curve activity; zero where unseen."""
        xy = xy.float() if torch.is_tensor(xy) else torch.as_tensor(np.asarray(xy, dtype=np.float32))
        seen = torch.isfinite(xy).all(-1, keepdim=True)
        d = (torch.nan_to_num(xy)[..., :, None] - self.centres.to(xy.device)) / self.width
        return torch.exp(-0.5 * d * d) * seen[..., None]

    def encode(self, xy) -> torch.Tensor:
        b = self.bumps(xy)
        return b.reshape(*b.shape[:-2], self.n_out)

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """(..., 2, n) cell activity (logits) -> (..., 2) position: softmax centre of mass.
        Biased inward within about a cell of the image edge, as any centre of mass is."""
        return (torch.softmax(logits, dim=-1) * self.centres.to(logits.device)).sum(-1)


# --- path parameters -----------------------------------------------------------------

def path_targets(t, gt, period_s: float, until_s: float = np.nan,
                 n_harmonics: int = N_HARMONICS) -> np.ndarray:
    """Per step: [period, cos phi, sin phi, x: mean + (a_k, b_k) per harmonic, y: same].
    Fitted on the steps before `until_s` (the break). Phase 0 is a point on the path --
    the rightmost point of the first-harmonic ellipse -- so clips of one path agree."""
    t, gt = np.asarray(t, dtype=float), np.asarray(gt, dtype=float)
    keep = np.isfinite(gt).all(axis=1) & ((t < until_s) if np.isfinite(until_s) else True)
    coef, _ = _harmonic_fit(t[keep], gt[keep], period_s, n_harmonics)
    a1, b1 = coef[1], coef[2]
    axis = 0 if np.hypot(a1[0], b1[0]) > 1e-6 else 1
    theta0 = math.atan2(b1[axis], a1[axis])
    w = 2.0 * np.pi / period_s
    phase = np.mod(w * t - theta0, 2.0 * np.pi)
    out = np.empty((len(t), N_PATH))
    out[:, 0], out[:, 1], out[:, 2] = period_s, np.cos(phase), np.sin(phase)
    for axis in range(2):
        base = 3 + axis * (1 + 2 * n_harmonics)
        out[:, base] = coef[0, axis]
        for k in range(1, n_harmonics + 1):
            a, b = coef[2 * k - 1, axis], coef[2 * k, axis]
            c, s = math.cos(k * theta0), math.sin(k * theta0)
            out[:, base + 2 * k - 1] = a * c + b * s       # coefficients re-based on phase 0
            out[:, base + 2 * k] = b * c - a * s
    return out


def path_position(p, horizon_s, n_harmonics: int = N_HARMONICS) -> np.ndarray:
    """Position `horizon_s` (scalar or per row) ahead from path parameters (N, N_PATH) -> (N, 2)."""
    p = np.asarray(p, dtype=float)
    phi = np.arctan2(p[:, 2], p[:, 1]) + 2.0 * np.pi * horizon_s / p[:, 0]
    out = np.empty((len(p), 2))
    for axis in range(2):
        base = 3 + axis * (1 + 2 * n_harmonics)
        acc = p[:, base].copy()
        for k in range(1, n_harmonics + 1):
            acc += p[:, base + 2 * k - 1] * np.cos(k * phi) + p[:, base + 2 * k] * np.sin(k * phi)
        out[:, axis] = acc
    return out


def _shape_points(p: torch.Tensor, n_phases: int) -> torch.Tensor:
    """The cycle the path parameters describe, at `n_phases` even phases: (..., n_phases, 2).
    Torch twin of `path_position` over a whole cycle, for the training loss."""
    theta = torch.arange(n_phases, dtype=p.dtype, device=p.device) * (2.0 * math.pi / n_phases)
    out = []
    for axis in range(2):
        base = 3 + axis * (1 + 2 * N_HARMONICS)
        acc = p[..., base, None].expand(*p.shape[:-1], n_phases)
        for k in range(1, N_HARMONICS + 1):
            acc = acc + p[..., base + 2 * k - 1, None] * torch.cos(k * theta) + p[..., base + 2 * k, None] * torch.sin(k * theta)
        out.append(acc)
    return torch.stack(out, dim=-1)


# --- the network -------------------------------------------------------------------

def _betas(n: int, tau_range, dt_s: float, gen: torch.Generator) -> torch.Tensor:
    """Per-neuron decay per step for time constants spread log-uniformly over `tau_range`."""
    lo, hi = tau_range
    tau = lo * (hi / lo) ** torch.rand(n, generator=gen)
    return torch.exp(-dt_s / tau)


class AdaptiveLeaky(nn.Module):
    """LIF neuron whose threshold rises by `scale` per spike and decays back over seconds
    (Bellec et al. 2018, LSNN). The adaptation is a slow state that spikes do not reset,
    so it holds a trace of the neuron's own recent activity. Same call shape as
    snn.Leaky: (input, (mem, adapt)) -> (spk, (mem, adapt))."""

    def __init__(self, beta: torch.Tensor, rho: torch.Tensor, scale: float = 1.8, spike_grad=None):
        super().__init__()
        self.beta = nn.Parameter(beta)               # membrane decay per step
        self.rho = nn.Parameter(rho)                 # adaptation decay per step
        self.scale = scale
        self.spike_grad = spike_grad or surrogate.fast_sigmoid(slope=25)

    def forward(self, cur: torch.Tensor, state):
        mem, adapt = state
        threshold = 1.0 + self.scale * adapt
        mem = self.beta.clamp(0, 1) * mem + cur
        spk = self.spike_grad(mem - threshold)
        mem = mem - (spk * threshold).detach()             # reset by subtraction
        adapt = self.rho.clamp(0, 1) * adapt + spk
        return spk, (mem, adapt)


class TwoLayerNet(nn.Module):
    """Fast recurrent LIF layer (takes the input, feeds the slow layer, receives its
    feedback) and slow recurrent LIF layer. Readouts from low-passed spikes; the horizon
    heads are `n_cells` place cells per axis, decoded by their centre of mass."""

    def __init__(self, n_in: int, n_fast: int = 384, n_slow: int = 128, dt_s: float = 0.005,
                 tau_fast=(0.01, 0.025), tau_slow=(0.3, 0.7), readout_tau_s: float = 0.015,
                 n_horizons: int = len(HORIZONS_S), n_cells: int = 32, n_path: int = N_PATH,
                 heads_from: str = "fast", slow_kind: str = "adaptive", tau_adapt=(0.5, 4.0),
                 adapt_scale: float = 1.8, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        grad = surrogate.fast_sigmoid(slope=25)
        self.fast = snn.Leaky(beta=_betas(n_fast, tau_fast, dt_s, gen), learn_beta=True, spike_grad=grad)
        self.slow_kind = slow_kind
        if slow_kind == "adaptive":          # fast membrane, seconds-long threshold adaptation
            self.slow = AdaptiveLeaky(_betas(n_slow, tau_fast, dt_s, gen), _betas(n_slow, tau_adapt, dt_s, gen),
                                      adapt_scale, grad)
        else:                                # plain LIF with a long membrane leak (ablation)
            self.slow = snn.Leaky(beta=_betas(n_slow, tau_slow, dt_s, gen), learn_beta=True, spike_grad=grad)
        self.w_in = nn.Linear(n_in, n_fast)
        self.w_ff = nn.Linear(n_fast, n_fast, bias=False)
        self.w_sf = nn.Linear(n_slow, n_fast, bias=False)
        self.w_fs = nn.Linear(n_fast, n_slow)
        self.w_ss = nn.Linear(n_slow, n_slow, bias=False)
        self.heads_from = heads_from                        # "fast" or "both" layers
        self.read_h = nn.Linear(n_fast + (n_slow if heads_from == "both" else 0), n_horizons * 2 * n_cells)
        self.read_p = nn.Linear(n_slow, n_path)
        self.n_horizons, self.n_cells = n_horizons, n_cells
        self.alpha = math.exp(-dt_s / readout_tau_s)
        # A slow neuron sums its input over ~1/(1-beta) steps; scale what reaches it so
        # the two layers start at comparable firing rates.
        slow_gain = float(1.0 - self.slow.beta.detach().mean())
        if slow_kind == "adaptive":
            slow_gain = 0.5                  # membrane is fast; adaptation alone slows it down
        with torch.no_grad():
            for lin, gain in ((self.w_in, 3.0), (self.w_ff, 0.5), (self.w_sf, 0.5),
                              (self.w_fs, 3.0 * slow_gain), (self.w_ss, 0.5 * slow_gain)):
                nn.init.normal_(lin.weight, std=gain / math.sqrt(lin.in_features))
                if lin.bias is not None:
                    lin.bias.zero_()
        self.rates = (torch.zeros(()), torch.zeros(()))    # mean firing rate per layer, last forward

    @property
    def last_rates(self) -> tuple[float, float]:
        return float(self.rates[0]), float(self.rates[1])

    def init_state(self, batch: int, device):
        z = lambda n: torch.zeros(batch, n, device=device)             # noqa: E731
        n_f, n_s = self.w_ff.in_features, self.w_ss.in_features
        return z(n_f), z(n_s), z(n_f), z(n_s), z(n_f), z(n_s), z(n_s)

    def _slow_step(self, cur, mem_s, adapt_s):
        if self.slow_kind == "adaptive":
            spk, (mem_s, adapt_s) = self.slow(cur, (mem_s, adapt_s))
            return spk, mem_s, adapt_s
        spk, mem_s = self.slow(cur, mem_s)
        return spk, mem_s, adapt_s

    def forward(self, x: torch.Tensor, state=None):
        """x: (T, B, n_in) -> head cells (T, B, n_horizons, 2, n_cells), path (T, B, n_path),
        state. The pass's mean firing rates stay in `self.rates`, with gradient."""
        T, B, _ = x.shape
        mem_f, mem_s, spk_f, spk_s, r_f, r_s, adapt_s = state or self.init_state(B, x.device)
        heads, paths, rate_f, rate_s = [], [], 0.0, 0.0
        for i in range(T):
            spk_f, mem_f = self.fast(self.w_in(x[i]) + self.w_ff(spk_f) + self.w_sf(spk_s), mem_f)
            spk_s, mem_s, adapt_s = self._slow_step(self.w_fs(spk_f) + self.w_ss(spk_s), mem_s, adapt_s)
            r_f = self.alpha * r_f + (1.0 - self.alpha) * spk_f
            r_s = self.alpha * r_s + (1.0 - self.alpha) * spk_s
            heads.append(self.read_h(torch.cat([r_f, r_s], dim=-1) if self.heads_from == "both" else r_f))
            paths.append(self.read_p(r_s))
            rate_f, rate_s = rate_f + spk_f.mean(), rate_s + spk_s.mean()
        self.rates = (rate_f / T, rate_s / T)
        heads = torch.stack(heads).reshape(T, B, self.n_horizons, 2, self.n_cells)
        return heads, torch.stack(paths), (mem_f, mem_s, spk_f, spk_s, r_f, r_s, adapt_s)


# --- the memory ------------------------------------------------------------------------

def _prefer_performance_cores() -> None:
    """On a hybrid Intel CPU Windows parks a background process on the efficiency cores
    (measured: 2-3x slower). Pin to the performance cores; harmless elsewhere."""
    import os
    if os.name != "nt" or (os.cpu_count() or 0) < 12:
        return
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32                 # 6 P-cores x 2 threads come first, then the E-cores
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        k32.SetProcessAffinityMask(k32.GetCurrentProcess(), 0xFFF)
    except Exception:
        pass


def _forward_fill(obs: np.ndarray) -> np.ndarray:
    """Each row replaced by the last finite row before it (NaN until the first)."""
    out = obs.copy()
    for i in range(1, len(out)):
        if not np.isfinite(out[i]).all():
            out[i] = out[i - 1]
    return out


def _anchor(obs: np.ndarray, alpha: float) -> np.ndarray:
    """The position the offsets are measured from: the last seen position, smoothed by a
    causal running average (`alpha` = 1 means none) so one window's centroid jitter does
    not pass straight into the prediction. Unseen steps hold the last value."""
    out = _forward_fill(obs)
    if alpha >= 1.0:
        return out
    for i in range(1, len(out)):
        if not np.isfinite(obs[i]).all():
            out[i] = out[i - 1]
        elif np.isfinite(out[i - 1]).all():
            out[i] = out[i - 1] + alpha * (obs[i] - out[i - 1])
    return out


class SpikingMemory:
    """TrajectoryMemory backed by TwoLayerNet. `fit` pretrains on Tracks; `observe` streams
    one position per step through the frozen network."""

    def __init__(self, dt_s: float, device=None, horizons_s=HORIZONS_S,
                 n_per_axis: int = 32, n_vel_cells: int = 16, vel_range: float = 0.08,
                 vel_window_s: float = 0.025, anchor_tau_s: float = 0.0,
                 out_range_per_s: float = 1.75, out_range_min: float = 0.1,
                 n_fast: int = 384, n_slow: int = 128,
                 tau_fast=(0.01, 0.025), tau_slow=(0.3, 0.7), readout_tau_s: float = 0.015,
                 rate_target: float = 0.1, rate_weight: float = 10.0, heads_from: str = "fast",
                 slow_kind: str = "adaptive", tau_adapt=(0.5, 4.0), adapt_scale: float = 1.8,
                 score_tau_s: float = 0.1, resolution=(640, 480), seed: int = 0):
        # CPU by default: at these layer sizes per-step launch overhead makes a GPU slower,
        # and more than a few threads slow the small per-step ops down (measured 3x at 10).
        self.device = torch.device(device or "cpu")
        if self.device.type == "cpu":
            torch.set_num_threads(min(4, torch.get_num_threads()))
            _prefer_performance_cores()
        self.dt_s, self.horizons_s, self.n_per_axis = dt_s, tuple(horizons_s), n_per_axis
        self.n_fast, self.n_slow, self.tau_fast, self.tau_slow = n_fast, n_slow, tuple(tau_fast), tuple(tau_slow)
        self.readout_tau_s, self.score_tau_s, self.resolution, self.seed = readout_tau_s, score_tau_s, tuple(resolution), seed
        self.rate_target, self.rate_weight, self.heads_from = rate_target, rate_weight, heads_from
        self.slow_kind, self.tau_adapt, self.adapt_scale = slow_kind, tuple(tau_adapt), adapt_scale
        self.n_vel_cells, self.vel_range, self.vel_window_s = n_vel_cells, vel_range, vel_window_s
        self.anchor_tau_s = anchor_tau_s
        self.anchor_alpha = 1.0 if anchor_tau_s <= 0 else min(1.0, dt_s / anchor_tau_s)
        self.out_range_per_s, self.out_range_min = out_range_per_s, out_range_min
        self.k_vel = max(1, round(vel_window_s / dt_s))
        self.pos_enc = PlaceCells(n_per_axis)
        self.vel_enc = PlaceCells(n_vel_cells, -vel_range, vel_range)
        # each horizon's displacement cells span what the target can move in that time
        self.out_enc = [PlaceCells(n_per_axis, -r, r) for r in
                        (max(out_range_min, out_range_per_s * h) for h in self.horizons_s)]
        n_in = self.pos_enc.n_out + self.vel_enc.n_out
        self.net = TwoLayerNet(n_in, n_fast, n_slow, dt_s, tau_fast, tau_slow, readout_tau_s,
                               len(self.horizons_s), n_per_axis, N_PATH, heads_from, slow_kind, tau_adapt,
                               adapt_scale, seed).to(self.device)
        self.path_mean, self.path_std = np.zeros(N_PATH), np.ones(N_PATH)
        self.path_loss = "geometric"
        self.scales = (1.0, 1.0)                     # on-pattern level of (immediate, structural), px
        self.reset()

    # -- persistence --

    def config(self) -> dict:
        return {k: getattr(self, k) for k in ("dt_s", "horizons_s", "n_per_axis", "n_vel_cells", "vel_range",
                                              "vel_window_s", "anchor_tau_s", "out_range_per_s", "out_range_min", "n_fast",
                                              "n_slow", "tau_fast", "tau_slow", "readout_tau_s", "rate_target",
                                              "rate_weight", "heads_from", "slow_kind", "tau_adapt", "adapt_scale",
                                              "score_tau_s", "resolution", "seed")}

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": self.config(), "state": self.net.state_dict(), "path_mean": self.path_mean,
                    "path_std": self.path_std, "scales": self.scales}, path)
        return path

    @classmethod
    def load(cls, path, device=None, dt_s: float | None = None) -> "SpikingMemory":
        """Rebuild from a checkpoint. `dt_s`, if given, must match the training step."""
        ck = torch.load(Path(path), map_location="cpu", weights_only=False)
        if dt_s is not None and not np.isclose(ck["config"]["dt_s"], dt_s):
            raise ValueError(f"{path}: trained at dt {ck['config']['dt_s']} s, not {dt_s} s")
        m = cls(device=device, **{"slow_kind": "leaky", **ck["config"]})    # pre-LSNN checkpoints
        m.net.load_state_dict(ck["state"])
        m.path_mean, m.path_std, m.scales = ck["path_mean"], ck["path_std"], tuple(ck["scales"])
        m.reset()
        return m

    # -- streaming --

    def reset(self) -> None:
        self.net.eval()
        self.state = None
        self.last_heads = None
        self.last_path = None
        self.recent = deque(maxlen=self.k_vel + 1)        # anchors, for the velocity cells
        self.ref = np.array([np.nan, np.nan])             # smoothed last seen position: what offsets add to
        self.ring = deque(maxlen=max(1, round(self.horizons_s[self._i100] / self.dt_s)))
        self.err_imm = self.err_str = 0.0
        self.seen_any = False

    @property
    def _i100(self) -> int:
        return int(np.argmin(np.abs(np.array(self.horizons_s) - 0.1)))

    def _encode(self, obs, vel) -> torch.Tensor:
        return torch.cat([self.pos_enc.encode(obs), self.vel_enc.encode(vel)], dim=-1)

    def _decode(self, cells: torch.Tensor) -> torch.Tensor:
        """(..., n_horizons, 2, n_cells) -> (..., n_horizons, 2) displacements."""
        return torch.stack([enc.decode(cells[..., j, :, :]) for j, enc in enumerate(self.out_enc)], dim=-2)

    def observe(self, x: float, y: float) -> None:
        obs = np.array([x, y], dtype=float)
        if np.isfinite(obs).all():
            self.ref = obs if not np.isfinite(self.ref).all() else self.ref + self.anchor_alpha * (obs - self.ref)
        self.recent.append(self.ref.copy())
        vel = self.recent[-1] - self.recent[0] if len(self.recent) == self.recent.maxlen else obs * np.nan
        inp = self._encode(obs[None], vel[None]).reshape(1, 1, -1).to(self.device)
        with torch.no_grad():
            cells, path, self.state = self.net(inp, self.state)
        self.last_heads = self.ref + self._decode(cells[0, 0]).cpu().numpy()
        self.last_path = path[0, 0].cpu().numpy() * self.path_std + self.path_mean
        alpha = min(1.0, self.dt_s / self.score_tau_s)
        seen = np.isfinite(obs).all()
        if seen and len(self.ring) == self.ring.maxlen and np.isfinite(self.ring[0]).all():
            e = np.hypot(*((self.ring[0] - obs) * self.resolution))
            self.err_imm += alpha * (e - self.err_imm)
        if seen and np.isfinite(self.last_path[0]) and self.last_path[0] > 1e-3:
            e = np.hypot(*((path_position(self.last_path[None], 0.0)[0] - obs) * self.resolution))
            self.err_str += alpha * (e - self.err_str)
        self.ring.append(self.last_heads[self._i100].copy())
        self.seen_any = self.seen_any or bool(seen)

    def period(self) -> float:
        if self.last_path is None or not (np.isfinite(self.last_path[0]) and self.last_path[0] > 1e-3):
            return np.nan
        return float(self.last_path[0])

    def path_points(self, fractions):
        if not np.isfinite(self.period()):
            return None
        ahead = np.asarray(fractions, dtype=float) * self.last_path[0]
        return path_position(np.repeat(self.last_path[None], len(ahead), axis=0), ahead)

    def predict(self, horizon_s: float) -> tuple[float, float]:
        if self.last_heads is None:
            return (np.nan, np.nan)
        hits = np.isclose(self.horizons_s, horizon_s, atol=1e-6)
        if not hits.any():
            raise ValueError(f"horizon {horizon_s} s is not one of the trained heads {self.horizons_s}")
        x, y = self.last_heads[int(np.argmax(hits))]
        return (float(x), float(y))

    def deviation_score(self) -> float:
        if not self.seen_any:
            return 0.0
        return float(max(self.err_imm / self.scales[0], self.err_str / self.scales[1]))

    # -- pretraining --

    def _arrays(self, tracks: list[Track]):
        """Per track: encoded input (position + velocity cells), horizon targets as offsets
        from the last seen position + mask, path targets + mask, cropped to the shortest
        track. Loss is off before the first cycle and after a break."""
        n = min(len(tr.t) for tr in tracks)
        shifts = [round(h / self.dt_s) for h in self.horizons_s]
        out = []
        for tr in tracks:
            t, obs, gt = tr.t[:n], tr.obs[:n], tr.gt[:n]
            dev = tr.deviation_t if np.isfinite(tr.deviation_t) else np.inf
            ref = _anchor(obs, self.anchor_alpha)
            vel = np.full_like(obs, np.nan)
            vel[self.k_vel:] = ref[self.k_vel:] - ref[: n - self.k_vel]
            yh = np.full((n, len(shifts), 2), np.nan)
            mh = np.zeros((n, len(shifts)), dtype=bool)
            for j, (k, h) in enumerate(zip(shifts, self.horizons_s)):
                yh[: n - k, j] = gt[k:] - ref[: n - k]
                mh[:, j] = (t >= tr.period_s) & (t + h < dev) & np.isfinite(yh[:, j]).all(-1)
            yp = path_targets(t, gt, tr.period_s, until_s=dev)
            mp = (t >= tr.period_s) & (t < dev)
            out.append({"x": self._encode(obs, vel), "yh": torch.tensor(np.nan_to_num(yh), dtype=torch.float32),
                        "mh": torch.tensor(mh), "yp": yp, "mp": torch.tensor(mp)})
        return out

    def _standardise(self, arrays, fit: bool) -> None:
        if fit:
            rows = np.concatenate([a["yp"][a["mp"].numpy()] for a in arrays])
            self.path_mean, self.path_std = rows.mean(0), np.maximum(rows.std(0), 1e-6)
        for a in arrays:
            a["yp"] = torch.tensor((a["yp"] - self.path_mean) / self.path_std, dtype=torch.float32)

    def _stack(self, arrays, idx, key):
        return torch.stack([arrays[i][key] for i in idx], dim=1).to(self.device)     # (N, B, ...)

    def _loss(self, cells, path, yh, mh, yp, mp, path_weight: float):
        """Cross-entropy of each head's cell activity against the true position's bump,
        the path-head loss, and the firing-rate regulariser that keeps both layers near
        `rate_target` (a silent layer cannot learn). Returns (loss, heads, path, path px).

        Path loss "geometric": the cycle the head describes against the true cycle, as
        px distance at SHAPE_PHASES phases (scaled by SHAPE_PX_SCALE), plus the phase
        pair's squared error and the period's relative squared error -- so each of the
        17 numbers costs what it does to the drawn path. "mse": squared error on the
        standardised numbers (the ablation)."""
        target = torch.stack([enc.bumps(yh[..., j, :]) for j, enc in enumerate(self.out_enc)], dim=-3)
        target = target / target.sum(-1, keepdim=True).clamp_min(1e-6)
        ce = -(target * torch.log_softmax(cells, dim=-1)).sum(-1).mean(-1)[mh]      # mean over x, y
        if self.path_loss == "geometric":
            std = torch.as_tensor(self.path_std, dtype=path.dtype, device=path.device)
            mean = torch.as_tensor(self.path_mean, dtype=path.dtype, device=path.device)
            p, q = path * std + mean, yp * std + mean
            res = torch.as_tensor(self.resolution, dtype=path.dtype, device=path.device)
            diff = (_shape_points(p, SHAPE_PHASES) - _shape_points(q, SHAPE_PHASES)) * res
            px = torch.sqrt((diff ** 2).sum(-1) + 1e-6).mean(-1)      # smooth at zero, unlike hypot
            phase = ((p[..., 1:3] - q[..., 1:3]) ** 2).sum(-1)
            period = ((p[..., 0] - q[..., 0]) / q[..., 0]) ** 2
            ep, px = (px / SHAPE_PX_SCALE + phase + period)[mp], px[mp]
        else:
            ep, px = ((path - yp) ** 2).mean(-1)[mp], None
        lh = ce.mean() if ce.numel() else cells.sum() * 0
        lp = ep.mean() if ep.numel() else path.sum() * 0
        reg = sum((r - self.rate_target) ** 2 for r in self.net.rates)
        path_px = px.mean().item() if px is not None and px.numel() else np.nan
        return lh + path_weight * lp + self.rate_weight * reg, lh.item(), lp.item(), path_px

    def _run_chunks(self, arrays, idx, chunk: int, path_weight: float, optimiser=None):
        """Forward (and update, if an optimiser is given) over one batch of tracks, chunk by
        chunk with the state carried across. Returns mean losses and the 100 ms head's px
        errors on the masked steps."""
        x, yh, mh = self._stack(arrays, idx, "x"), self._stack(arrays, idx, "yh"), self._stack(arrays, idx, "mh")
        yp, mp = self._stack(arrays, idx, "yp"), self._stack(arrays, idx, "mp")
        state, losses, errs = None, [], []
        for s in range(0, x.shape[0], chunk):
            sl = slice(s, s + chunk)
            with torch.set_grad_enabled(optimiser is not None):
                cells, path, state = self.net(x[sl], state)
                loss, lh, lp, path_px = self._loss(cells, path, yh[sl], mh[sl], yp[sl], mp[sl], path_weight)
                heads = self._decode(cells)
            if optimiser is not None:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimiser.step()
            state = tuple(v.detach() for v in state)
            losses.append((loss.item(), lh, lp, path_px))
            h100, m100 = heads[:, :, self._i100].detach(), mh[sl][:, :, self._i100]
            px = torch.hypot(*(((h100 - yh[sl][:, :, self._i100]) * torch.tensor(self.resolution, device=h100.device))).unbind(-1))
            errs.append(px[m100].cpu().numpy())
        return np.nanmean(losses, axis=0), np.concatenate(errs)

    def fit(self, tracks: list[Track], epochs: int = 30, chunk_s: float = 5.0, batch: int = 16,
            val_fraction: float = 0.1, lr: float = 1e-3, path_weight: float = 1.0, seed: int = 0,
            schedule: str = "none", weight_decay: float = 0.0, val_tracks: list[Track] | None = None,
            path_loss: str = "geometric", log_fn=None) -> list[dict]:
        """Train on `tracks` (a held-back fraction validates and picks the best epoch, or
        `val_tracks` if given), then set the deviation scales on the validation tracks.
        `schedule` "cosine" decays the learning rate to zero over the epochs; `path_loss`
        as in `_loss`. Returns the per-epoch log."""
        if path_loss not in ("geometric", "mse"):
            raise ValueError(f"path_loss {path_loss!r}: geometric or mse")
        self.path_loss = path_loss
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if val_tracks is not None:
            train_idx, val_idx = list(range(len(tracks))), list(range(len(tracks), len(tracks) + len(val_tracks)))
            tracks = list(tracks) + list(val_tracks)
        else:
            order = rng.permutation(len(tracks))
            n_val = max(1, round(len(tracks) * val_fraction)) if len(tracks) > 1 else 0
            val_idx, train_idx = list(order[:n_val]), list(order[n_val:])
        arrays = self._arrays(tracks)
        self._standardise([arrays[i] for i in train_idx], fit=True)
        self._standardise([arrays[i] for i in val_idx], fit=False)
        chunk = max(1, round(chunk_s / self.dt_s))
        optimiser = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)
                     if schedule == "cosine" else None)
        log, best = [], (np.inf, None)

        def validate(epoch, train_loss):
            self.net.eval()
            if val_idx:
                loss, errs = self._run_chunks(arrays, val_idx, chunk, path_weight)
                val_px = float(np.median(errs)) if errs.size else np.nan
            else:
                loss, val_px = (np.nan,) * 4, np.nan
            entry = {"epoch": epoch, "train_loss": train_loss, "val_loss": loss[0], "val_heads": loss[1],
                     "val_path": loss[2], "val_path_px": loss[3], "val_px": val_px,
                     "rate_fast": self.net.last_rates[0], "rate_slow": self.net.last_rates[1]}
            log.append(entry)
            if log_fn:
                log_fn(entry)
            return val_px

        validate(0, np.nan)
        for epoch in range(1, epochs + 1):
            self.net.train()
            losses = []
            order_epoch = rng.permutation(len(train_idx))              # every track once per epoch
            for b in range(0, len(train_idx), batch):
                idx = [train_idx[i] for i in order_epoch[b:b + batch]]
                if idx:
                    losses.append(self._run_chunks(arrays, idx, chunk, path_weight, optimiser)[0][0])
            if scheduler is not None:
                scheduler.step()
            val_px = validate(epoch, float(np.mean(losses)))
            score = val_px if np.isfinite(val_px) else -epoch                 # no validation: keep the last
            if score < best[0]:
                best = (score, {k: v.detach().clone() for k, v in self.net.state_dict().items()})
        if best[1] is not None:
            self.net.load_state_dict(best[1])
        self._calibrate([tracks[i] for i in (val_idx or train_idx)])
        self.reset()
        return log

    def _calibrate(self, tracks: list[Track]) -> None:
        """On-pattern level of each deviation signal, streamed exactly as at evaluation."""
        imm, strc = [], []
        for tr in tracks:
            self.reset()
            dev = tr.deviation_t if np.isfinite(tr.deviation_t) else np.inf
            for i, (x, y) in enumerate(tr.obs):
                self.observe(x, y)
                if tr.period_s + 0.5 <= tr.t[i] < dev:
                    imm.append(self.err_imm)
                    strc.append(self.err_str)
        self.scales = (max(float(np.median(imm)), 1e-3) if imm else 1.0,
                       max(float(np.median(strc)), 1e-3) if strc else 1.0)
