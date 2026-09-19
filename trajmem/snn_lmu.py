"""Trajectory memory with a hand-set window memory: the fast recurrent LIF layer of `snn`
plus a spiking Legendre Memory Unit (`lmu`) holding the last `theta_s` of the position.
Horizon heads read the fast layer (short) or fast + LMU (long); the path head reads the
LMU. When the target is unseen the network feeds its own 25 ms prediction back in, so
the window keeps rolling and the heads keep predicting (design:
docs/superpowers/specs/2026-09-19-lmu-memory-design.md).
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
from .lmu import SpikingLmu, build_population, state_radii
from .snn import HORIZONS_S, N_PATH, SHAPE_PHASES, SHAPE_PX_SCALE, SpikingMemory, _betas, _shape_points, path_targets

LMU_ACT_SCALE = 100.0                        # Hz -> order one, for the readouts and the fast layer
LMU_FEATURES = 512                           # fixed random projection of the population the readouts see


class LmuNet(nn.Module):
    """Fast layer + readouts around a fixed LMU population (stepped outside, its filtered
    rates come in as `act`). `step` is one network step; the caller carries the state."""

    def __init__(self, n_in: int, n_fast: int, n_lmu: int, dt_s: float, tau_fast, readout_tau_s: float,
                 n_horizons: int, n_cells: int, n_path: int, n_short: int, seed: int = 0,
                 n_features: int = LMU_FEATURES, path_from: str = "lmu", path_readout: str = "linear",
                 n_state: int = 0):
        super().__init__()
        if path_from not in ("lmu", "both", "state") or path_readout not in ("linear", "mlp"):
            raise ValueError(f"path_from {path_from!r} / path_readout {path_readout!r}")
        gen = torch.Generator().manual_seed(seed)
        grad = surrogate.fast_sigmoid(slope=25)
        self.fast = snn.Leaky(beta=_betas(n_fast, tau_fast, dt_s, gen), learn_beta=True, spike_grad=grad)
        # The population is read through a fixed random projection: a random subspace of its
        # tuning curves keeps the nonlinear-readout capacity at a fraction of the cost.
        self.register_buffer("proj", torch.randn(n_lmu, n_features, generator=gen) / math.sqrt(n_lmu))
        self.w_in = nn.Linear(n_in, n_fast)
        self.w_ff = nn.Linear(n_fast, n_fast, bias=False)
        self.w_sf = nn.Linear(n_features, n_fast, bias=False)
        self.read_short = nn.Linear(n_fast, n_short * 2 * n_cells)
        self.read_long = nn.Linear(n_fast + n_features, (n_horizons - n_short) * 2 * n_cells)
        # A linear readout of one-dimensional ensembles is additive over the state dimensions
        # and cannot form the cross terms a period estimate needs; the fast layer ("both")
        # or a hidden layer ("mlp") supplies them.
        # "state": the window as Nengo's decoders read it (n_state numbers), also hand-set
        n_p = n_state if path_from == "state" else n_features + (n_fast if path_from == "both" else 0)
        self.read_p = nn.Linear(n_p, n_path) if path_readout == "linear" else             nn.Sequential(nn.Linear(n_p, 256), nn.ReLU(), nn.Linear(256, n_path))
        self.path_from = path_from
        self.n_fast, self.n_features, self.n_horizons, self.n_cells, self.n_short = n_fast, n_features, n_horizons, n_cells, n_short
        self.alpha = math.exp(-dt_s / readout_tau_s)

    def init_state(self, batch: int, device):
        z = lambda n: torch.zeros(batch, n, device=device)  # noqa: E731
        return (z(self.n_fast), z(self.n_fast), z(self.n_fast), z(self.n_features), None)   # mem, spikes, r_fast, r_lmu, r_state

    def step(self, x: torch.Tensor, act: torch.Tensor, state, decoded: torch.Tensor | None = None):
        """x: (B, n_in) input cells; act: (B, n_lmu) LMU rates in Hz; decoded: (B, n_state) the
        LMU's decoded window -> heads (B, H, 2, n_cells), path (B, n_path), state, this
        step's fast-layer spike fraction (with gradient)."""
        mem, spk, r_f, r_l, r_s = state
        a = (act / LMU_ACT_SCALE) @ self.proj
        if decoded is not None:
            r_s = decoded if r_s is None else self.alpha * r_s + (1.0 - self.alpha) * decoded
        spk, mem = self.fast(self.w_in(x) + self.w_ff(spk) + self.w_sf(a), mem)
        r_f = self.alpha * r_f + (1.0 - self.alpha) * spk
        r_l = self.alpha * r_l + (1.0 - self.alpha) * a
        heads = torch.cat([self.read_short(r_f), self.read_long(torch.cat([r_f, r_l], dim=-1))], dim=-1)
        heads = heads.reshape(x.shape[0], self.n_horizons, 2, self.n_cells)
        src = {"lmu": r_l, "both": torch.cat([r_l, r_f], dim=-1) if r_f is not None else r_l, "state": r_s}[self.path_from]
        path = self.read_p(src)
        return heads, path, (mem, spk, r_f, r_l, r_s), spk.mean()


class LmuMemory(SpikingMemory):
    """SpikingMemory whose slow part is the LMU population. Streaming and training share
    `_run`, which does the encoding, the anchor, the self-feeding and the LMU stepping."""

    def __init__(self, dt_s: float, device=None, q: int = 24, n_per_dim: int = 200, theta_s: float = 4.0,
                 tau_syn_s: float = 0.02, radii=None, lmu_seed: int = 0, n_fast: int = 384,
                 path_from: str = "lmu", path_readout: str = "linear", **kw):
        kw.setdefault("anchor_tau_s", 0.02)
        super().__init__(dt_s, device=device, n_fast=n_fast, n_slow=4, **kw)
        self.q, self.n_per_dim, self.theta_s, self.tau_syn_s, self.lmu_seed = q, n_per_dim, theta_s, tau_syn_s, lmu_seed
        self.path_from, self.path_readout = path_from, path_readout
        self.radii = np.full(2 * q, 0.5) if radii is None else np.asarray(radii, dtype=float)
        self.lmu = SpikingLmu(build_population(q, theta_s, n_per_dim, self.radii, tau_syn_s, 2, lmu_seed),
                              dt_s, device=self.device)
        n_in = self.pos_enc.n_out + self.vel_enc.n_out + 1                       # + the `seen` cell
        self.net = LmuNet(n_in, n_fast, self.lmu.n, dt_s, self.tau_fast, self.readout_tau_s,
                          len(self.horizons_s), self.n_per_axis, N_PATH,
                          sum(h <= 0.05 for h in self.horizons_s), self.seed,
                          path_from=path_from, path_readout=path_readout, n_state=2 * q).to(self.device)
        self.k_feed = max(1, round(self.horizons_s[0] / dt_s))                   # the 25 ms head feeds back
        self.radii_t = torch.tensor(self.radii, dtype=torch.float32, device=self.device)   # decoded state -> order one
        self.blank_prob = 0.0
        self.reset()

    def set_radii_from(self, tracks: list[Track]) -> None:
        """Size the LMU's representation to the corpus and rebuild the population."""
        self.radii = state_radii([tr.gt - 0.5 for tr in tracks], self.q, self.theta_s, self.dt_s)
        self.radii_t = torch.tensor(self.radii, dtype=torch.float32, device=self.device)
        self.lmu = SpikingLmu(build_population(self.q, self.theta_s, self.n_per_dim, self.radii, self.tau_syn_s,
                                               2, self.lmu_seed), self.dt_s, device=self.device)

    # -- persistence --

    def config(self) -> dict:
        c = super().config()
        c.pop("n_slow"); c.pop("tau_slow"); c.pop("heads_from"); c.pop("slow_kind"); c.pop("tau_adapt"); c.pop("adapt_scale")
        c.update(q=self.q, n_per_dim=self.n_per_dim, theta_s=self.theta_s, tau_syn_s=self.tau_syn_s,
                 radii=self.radii.tolist(), lmu_seed=self.lmu_seed, path_from=self.path_from,
                 path_readout=self.path_readout)
        return c

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"arch": "lmu", "config": self.config(), "state": self.net.state_dict(), "path_mean": self.path_mean,
                    "path_std": self.path_std, "scales": self.scales}, path)
        return path

    @classmethod
    def load(cls, path, device=None, dt_s: float | None = None) -> "LmuMemory":
        ck = torch.load(Path(path), map_location="cpu", weights_only=False)
        if dt_s is not None and not math.isclose(ck["config"]["dt_s"], dt_s):
            raise ValueError(f"{path}: trained at dt {ck['config']['dt_s']} s, not {dt_s} s")
        m = cls(device=device, **ck["config"])
        m.net.load_state_dict(ck["state"])
        m.path_mean, m.path_std, m.scales = ck["path_mean"], ck["path_std"], tuple(ck["scales"])
        m.reset()
        return m

    # -- the one loop --

    def _run(self, obs: torch.Tensor, state, record: list | None = None):
        """obs: (T, B, 2) positions, NaN = unseen. Returns head cells (T, B, H, 2, n_cells),
        path (T, B, n_path), the anchor per step (T, B, 2), the state, and the mean fast
        rate over the steps (with gradient). Unseen steps are fed the network's own
        position estimate: the 25 ms head's prediction from 25 ms ago."""
        T, B, _ = obs.shape
        if state is None:
            state = {"net": self.net.init_state(B, self.device), "lmu": self.lmu.init_state(B),
                     "ref": torch.full((B, 2), float("nan"), device=self.device),
                     "anchors": deque(maxlen=self.k_vel + 1), "fed": deque(maxlen=self.k_feed), "act": None}
        if state["act"] is None:
            state["act"] = torch.zeros(B, self.lmu.n, device=self.device)
        cells, paths, refs, rate = [], [], [], 0.0
        for i in range(T):
            o = obs[i]
            seen = torch.isfinite(o).all(-1)                                             # (B,)
            imagined = state["fed"][0] if len(state["fed"]) == state["fed"].maxlen else state["ref"]
            pos = torch.where(seen[:, None], o, imagined)
            have = torch.isfinite(pos).all(-1)
            ref = state["ref"]
            new_ref = torch.where(torch.isfinite(ref).all(-1)[:, None], ref + self.anchor_alpha * (pos - ref), pos)
            ref = torch.where(have[:, None], new_ref, ref)
            state["ref"] = ref
            state["anchors"].append(ref)
            vel = (state["anchors"][-1] - state["anchors"][0]) if len(state["anchors"]) == state["anchors"].maxlen \
                else torch.full_like(ref, float("nan"))
            x = torch.cat([self.pos_enc.encode(pos), self.vel_enc.encode(vel), seen[:, None].float()], dim=-1)
            with torch.no_grad():
                u = torch.nan_to_num(pos - 0.5)
                act, state["lmu"] = self.lmu.step(u, state["lmu"])
            state["act"] = act
            decoded = self.lmu.decode(act).reshape(B, -1) / self.radii_t if self.path_from == "state" else None
            heads, path, state["net"], r = self.net.step(x, act, state["net"], decoded)
            if record is not None:
                record.append(state["net"][4].detach())
            disp = self.out_enc[0].decode(heads[:, 0].detach())                          # 25 ms head, (B, 2)
            state["fed"].append(ref.detach() + disp)
            cells.append(heads); paths.append(path); refs.append(ref); rate = rate + r
        return torch.stack(cells), torch.stack(paths), torch.stack(refs), state, rate / T

    # -- streaming --

    def reset(self) -> None:
        super().reset()
        self.state = None
        self.last_u = None

    def observe(self, x: float, y: float) -> None:
        self.net.eval()
        obs = torch.tensor([[[x, y]]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            cells, path, ref, self.state, _ = self._run(obs, self.state)
        seen = bool(np.isfinite([x, y]).all())
        self.last_u = self.state["ref"][0].cpu().numpy()
        self.ref = ref[0, 0].cpu().numpy()
        self.last_heads = self.ref + self._decode(cells[0, 0]).cpu().numpy()
        self.last_path = path[0, 0].cpu().numpy() * self.path_std + self.path_mean
        self._score_step(np.array([x, y], dtype=float), seen)

    def _score_step(self, obs: np.ndarray, seen: bool) -> None:
        from .snn import path_position

        alpha = min(1.0, self.dt_s / self.score_tau_s)
        if seen and len(self.ring) == self.ring.maxlen and np.isfinite(self.ring[0]).all():
            e = np.hypot(*((self.ring[0] - obs) * self.resolution))
            self.err_imm += alpha * (e - self.err_imm)
        if seen and np.isfinite(self.last_path[0]) and self.last_path[0] > 1e-3:
            e = np.hypot(*((path_position(self.last_path[None], 0.0)[0] - obs) * self.resolution))
            self.err_str += alpha * (e - self.err_str)
        self.ring.append(self.last_heads[self._i100].copy())
        self.seen_any = self.seen_any or seen

    # -- pretraining --

    def _arrays(self, tracks: list[Track]):
        """Per track: observations, truth, path targets and the masks, cropped to the
        shortest track. Horizon targets are formed in the loop from the anchor it used."""
        n = min(len(tr.t) for tr in tracks)
        out = []
        for tr in tracks:
            t, obs, gt = tr.t[:n], tr.obs[:n], tr.gt[:n]
            dev = tr.deviation_t if np.isfinite(tr.deviation_t) else np.inf
            mh = np.zeros((n, len(self.horizons_s)), dtype=bool)
            for j, h in enumerate(self.horizons_s):
                k = round(h / self.dt_s)
                mh[: n - k, j] = (t[: n - k] >= tr.period_s) & (t[: n - k] + h < dev)
            yp = path_targets(t, gt, tr.period_s, until_s=dev)
            mp = (t >= tr.period_s) & (t < dev)
            out.append({"obs": torch.tensor(obs, dtype=torch.float32), "gt": torch.tensor(gt, dtype=torch.float32),
                        "t": torch.tensor(t, dtype=torch.float32), "period": float(tr.period_s),
                        "mh": torch.tensor(mh), "yp": yp, "mp": torch.tensor(mp)})
        return out

    def _blank(self, obs: torch.Tensor, t: torch.Tensor, periods, rng: np.random.Generator) -> torch.Tensor:
        """With probability `blank_prob` per track, hide one stretch of 0.25-1 periods after
        the first period: the network must carry itself through it."""
        obs = obs.clone()
        T = obs.shape[0]
        for b, period in enumerate(periods):
            if rng.random() >= self.blank_prob:
                continue
            length = int(rng.uniform(0.25, 1.0) * period / self.dt_s)
            length = min(length, T // 2)
            first = int(np.searchsorted(t.numpy(), period))
            if first + length >= T:
                continue
            start = int(rng.integers(first, T - length))
            obs[start:start + length, b] = float("nan")
        return obs

    def _run_chunks(self, arrays, idx, chunk: int, path_weight: float, optimiser=None):
        obs, gt = self._stack(arrays, idx, "obs"), self._stack(arrays, idx, "gt")
        mh, yp, mp = self._stack(arrays, idx, "mh"), self._stack(arrays, idx, "yp"), self._stack(arrays, idx, "mp")
        t = arrays[idx[0]]["t"]
        if optimiser is not None and self.blank_prob > 0:
            obs = self._blank(obs.cpu(), t, [arrays[i]["period"] for i in idx], self._rng).to(self.device)
        shifts = [round(h / self.dt_s) for h in self.horizons_s]
        state, losses, errs = None, [], []
        for s in range(0, obs.shape[0], chunk):
            sl = slice(s, s + chunk)
            with torch.set_grad_enabled(optimiser is not None):
                cells, path, ref, state, rate = self._run(obs[sl], state)
                yh = torch.stack([torch.roll(gt, -k, 0)[sl] - ref for k in shifts], dim=2)      # (T, B, H, 2)
                self.net.rates, self.net.last_rates = (rate,), (float(rate.detach()), 0.0)
                loss, lh, lp, path_px = self._loss(cells, path, torch.nan_to_num(yh), mh[sl], yp[sl], mp[sl], path_weight)
                heads = self._decode(cells)
            if optimiser is not None:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimiser.step()
            state = self._detach(state)
            losses.append((loss.item(), lh, lp, path_px))
            h100, m100 = heads[:, :, self._i100].detach(), mh[sl][:, :, self._i100]
            px = torch.hypot(*(((h100 - yh[:, :, self._i100].detach()) * torch.tensor(self.resolution, device=h100.device))).unbind(-1))
            errs.append(px[m100].cpu().numpy())
        return np.nanmean(losses, axis=0), np.concatenate(errs)

    @staticmethod
    def _detach(state):
        state["net"] = tuple(None if v is None else v.detach() for v in state["net"])
        state["ref"] = state["ref"].detach()
        state["anchors"] = deque((a.detach() for a in state["anchors"]), maxlen=state["anchors"].maxlen)
        state["fed"] = deque((a.detach() for a in state["fed"]), maxlen=state["fed"].maxlen)
        return state

    def fit(self, tracks: list[Track], blank_prob: float = 0.5, path_pretrain_steps: int = 0,
            val_tracks: list[Track] | None = None, **kw) -> list[dict]:
        """As SpikingMemory.fit. With `path_pretrain_steps` the path head is first fitted
        offline on the recorded window (needs `path_from="state"`: the window does not
        depend on the learned parts) and then frozen -- a readout of a fixed memory is a
        decoder, solved as such, not a few hundred BPTT updates."""
        self.blank_prob = blank_prob
        self._rng = np.random.default_rng(kw.get("seed", 0))
        if path_pretrain_steps > 0:
            if self.path_from != "state":
                raise ValueError("path_pretrain_steps needs path_from='state'")
            self._pretrain_path_head(tracks, val_tracks, path_pretrain_steps, kw.get("path_loss", "geometric"),
                                     kw.get("seed", 0))
            for prm in self.net.read_p.parameters():
                prm.requires_grad_(False)
            kw["path_weight"] = 0.0
        return super().fit(tracks, val_tracks=val_tracks, **kw)

    def _recorded_windows(self, tracks: list[Track]):
        """The path head's input (the low-passed decoded window) and its targets over the
        masked steps of `tracks`, streamed without blanks or gradient."""
        arrays = self._arrays(tracks)
        self._standardise(arrays, fit=False)
        xs, ys = [], []
        blank, self.blank_prob = self.blank_prob, 0.0
        with torch.no_grad():
            for b in range(0, len(tracks), 32):
                idx = list(range(b, min(b + 32, len(tracks))))
                obs, yp, mp = self._stack(arrays, idx, "obs"), self._stack(arrays, idx, "yp"), self._stack(arrays, idx, "mp")
                state, feats = None, []
                for s in range(0, obs.shape[0], 1000):
                    _, _, _, state, _ = self._run(obs[s:s + 1000], state, record=feats)
                x = torch.stack(feats)                                             # (T, B, n_state)
                xs.append(x[mp]); ys.append(yp[mp])
        self.blank_prob = blank
        return torch.cat(xs), torch.cat(ys)

    def _pretrain_path_head(self, tracks, val_tracks, steps: int, path_loss: str, seed: int) -> None:
        arrays = self._arrays(tracks)
        self._standardise(arrays, fit=True)
        x, y = self._recorded_windows(tracks)
        xv, yv = self._recorded_windows(val_tracks) if val_tracks else (None, None)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        opt = torch.optim.Adam(self.net.read_p.parameters(), lr=1e-3)
        mask = torch.ones(1, 512, dtype=torch.bool, device=self.device)
        for step in range(steps):
            i = torch.randint(0, len(x), (512,), generator=gen).to(self.device)
            loss = self._path_loss_only(self.net.read_p(x[i])[None], y[i][None], mask, path_loss)
            opt.zero_grad(); loss.backward(); opt.step()
        if xv is not None:
            with torch.no_grad():
                self.path_pretrain_val_px = float(self._path_px(self.net.read_p(xv)[None], yv[None]))

    def _path_loss_only(self, path, yp, mp, path_loss: str):
        if path_loss == "mse":
            return ((path - yp) ** 2).mean(-1)[mp].mean()
        p, q = self._destandardise(path), self._destandardise(yp)
        px = self._shape_px(p, q)
        phase = ((p[..., 1:3] - q[..., 1:3]) ** 2).sum(-1)
        period = ((p[..., 0] - q[..., 0]) / q[..., 0]) ** 2
        return (px / SHAPE_PX_SCALE + phase + period)[mp].mean()

    def _path_px(self, path, yp):
        return self._shape_px(self._destandardise(path), self._destandardise(yp)).median()

    def _destandardise(self, path):
        std = torch.as_tensor(self.path_std, dtype=path.dtype, device=path.device)
        mean = torch.as_tensor(self.path_mean, dtype=path.dtype, device=path.device)
        return path * std + mean

    def _shape_px(self, p, q):
        res = torch.as_tensor(self.resolution, dtype=p.dtype, device=p.device)
        diff = (_shape_points(p, SHAPE_PHASES) - _shape_points(q, SHAPE_PHASES)) * res
        return torch.sqrt((diff ** 2).sum(-1) + 1e-6).mean(-1)
