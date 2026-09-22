"""Stage-1 spiking localiser: an accumulated event frame in, the target's position (or "not
seen") out. A small spiking convolutional network (snnTorch, surrogate gradients) that
runs continuously over the frame stream, one step per frame, so leaky membranes integrate
the last few frames. Trained on simulated frames with stuck pixels, extra noise and flips
added -- never on a real clip. Design: docs/superpowers/specs/2026-09-21-spiking-localiser-design.md
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import snntorch as snn
import torch
from snntorch import surrogate
from torch import nn

from .localise import FrameSet, load_frame_set
from .snn import PlaceCells, _betas, _prefer_performance_cores

INPUT_SCALE = 1.0                       # counts per cell -> input current


# --- augmentation ---------------------------------------------------------------------

def augment_frames(frames: np.ndarray, gt: np.ndarray, present: np.ndarray, rng: np.random.Generator,
                   stuck=(50, 400), rate=(50.0, 1500.0), background=(0.0, 0.5), flips: bool = True,
                   blanks=(0, 3), blank_s=(0.1, 0.6), blank_radius: int = 6, brightness=(0.08, 2.0),
                   polarity_swap: bool = True, dt_s: float = 0.005):
    """What the real cameras add that the simulator does not: a set of stuck pixels firing
    at random high rates for the whole clip, a uniform noise floor, the two mirror flips
    (with the truth), a brightness scale (a faint real target drives the neurons weakly,
    and a neuron below threshold answers late -- the network must not depend on the
    event rate), and an ON/OFF swap (a dark target on a light ground reverses which edge
    leads; keyed on one polarity the network puts a moving target at its back, which
    looks like a lag) -- and what the target does on its own: a few stretches where its
    events are gone (the target removed within `blank_radius` cells of the truth,
    `present` False), so the network learns to say "not seen". Frames come back float32."""
    out = frames.astype(np.float32)
    n, _, h, w = out.shape
    gt, present = gt.copy(), present.copy()
    if brightness is not None:
        out *= float(np.exp(rng.uniform(np.log(brightness[0]), np.log(brightness[1]))))
    if polarity_swap and rng.random() < 0.5:
        out = out[:, ::-1].copy()
    for _ in range(int(rng.integers(blanks[0], blanks[1] + 1))):
        length = int(rng.uniform(blank_s[0], blank_s[1]) / dt_s)
        start = int(rng.integers(0, max(1, n - length)))
        for i in range(start, min(n, start + length)):
            if np.isfinite(gt[i]).all():
                cx, cy = int(gt[i, 0] * w), int(gt[i, 1] * h)
                out[i, :, max(0, cy - blank_radius):cy + blank_radius + 1, max(0, cx - blank_radius):cx + blank_radius + 1] = 0
            present[i] = False
    k = int(rng.integers(stuck[0], stuck[1] + 1))
    if k > 0:
        ys, xs = rng.integers(0, h, k), rng.integers(0, w, k)
        pol = rng.integers(0, 2, k)
        lam = rng.uniform(rate[0], rate[1], k) * dt_s
        counts = rng.poisson(lam[None, :], size=(n, k)).astype(np.float32)
        np.add.at(out, (slice(None), pol, ys, xs), counts)
    bg = float(rng.uniform(background[0], background[1]))
    if bg > 0:
        out += rng.poisson(bg, size=out.shape).astype(np.float32)
    if flips and rng.random() < 0.5:
        out, gt[:, 0] = out[..., ::-1].copy(), 1.0 - gt[:, 0]
    if flips and rng.random() < 0.5:
        out, gt[:, 1] = out[..., ::-1, :].copy(), 1.0 - gt[:, 1]
    return out, gt, present


# --- the network ----------------------------------------------------------------------

class SpikingLocaliserNet(nn.Module):
    """conv -> LIF -> conv -> LIF -> dense LIF -> linear readouts (place cells x, y; present)."""

    def __init__(self, n_cells: int = 32, ch=(8, 16), hidden: int = 128, dt_s: float = 0.005,
                 tau=(0.01, 0.02), readout_tau_s: float = 0.015, in_hw=(60, 80), seed: int = 0,
                 learn_tau: bool = False, polarity: str = "both"):
        super().__init__()
        if polarity not in ("both", "sum"):
            raise ValueError(f"polarity must be 'both' or 'sum', not {polarity!r}")
        self.polarity = polarity                  # "sum" folds ON and OFF into one channel: polarity-blind by construction
        gen = torch.Generator().manual_seed(seed)
        grad = surrogate.fast_sigmoid(slope=25)
        c1, c2 = ch
        # Time constants are fixed by default: learnable ones ran to beta = 1 (integrators
        # that never leak), and the network's answer then trails its input by ~100 ms.
        self.conv1 = nn.Conv2d(1 if polarity == "sum" else 2, c1, 5, stride=2, padding=2)
        self.lif1 = snn.Leaky(beta=_betas(c1, tau, dt_s, gen)[:, None, None], learn_beta=learn_tau, spike_grad=grad)
        self.conv2 = nn.Conv2d(c1, c2, 5, stride=2, padding=2)
        self.lif2 = snn.Leaky(beta=_betas(c2, tau, dt_s, gen)[:, None, None], learn_beta=learn_tau, spike_grad=grad)
        h, w = in_hw
        self.hw = (c1, math.ceil(h / 2), math.ceil(w / 2)), (c2, math.ceil(h / 4), math.ceil(w / 4))
        self.fc = nn.Linear(c2 * self.hw[1][1] * self.hw[1][2], hidden)
        self.lif3 = snn.Leaky(beta=_betas(hidden, tau, dt_s, gen), learn_beta=learn_tau, spike_grad=grad)
        self.read = nn.Linear(hidden, 2 * n_cells + 1)
        self.n_cells, self.hidden = n_cells, hidden
        self.alpha = math.exp(-dt_s / readout_tau_s)

    def init_state(self, batch: int, device):
        z = lambda *s: torch.zeros(batch, *s, device=device)  # noqa: E731
        return (z(*self.hw[0]), z(*self.hw[1]), z(self.hidden), z(self.hidden))     # mem1, mem2, mem3, readout trace

    def _input(self, frame: torch.Tensor) -> torch.Tensor:
        frame = frame * INPUT_SCALE
        return frame.sum(1, keepdim=True) if self.polarity == "sum" else frame

    @torch.no_grad()
    def calibrate(self, frames: torch.Tensor, target: float = 1.5) -> None:
        """Scale each layer's weights so its input current reaches `target` at the 99.9th
        percentile over `frames` (T, 2, H, W): the default initialisation assumes dense
        inputs, and event frames are sparse enough to leave the deeper layers silent."""
        state = self.init_state(1, frames.device)
        for layer, lif, name in ((self.conv1, self.lif1, "conv1"), (self.conv2, self.lif2, "conv2"), (self.fc, self.lif3, "fc")):
            currents = []
            st = self.init_state(1, frames.device)
            for f in frames:
                m1, m2, m3, r = st
                c1 = self.conv1(self._input(f[None])); s1, m1 = self.lif1(c1, m1)
                c2 = self.conv2(s1); s2, m2 = self.lif2(c2, m2)
                c3 = self.fc(s2.flatten(1)); s3, m3 = self.lif3(c3, m3)
                st = (m1, m2, m3, r)
                currents.append({"conv1": c1, "conv2": c2, "fc": c3}[name].flatten())
            level = float(torch.quantile(torch.cat(currents), 0.999))
            if level > 1e-6:
                layer.weight.mul_(target / level)
                layer.bias.mul_(target / level)

    def step(self, frame: torch.Tensor, state):
        """frame (B, 2, H, W) counts -> cells (B, 2, n_cells) logits, present (B,) logit,
        state, the layers' mean spike fraction (with gradient)."""
        m1, m2, m3, r = state
        s1, m1 = self.lif1(self.conv1(self._input(frame)), m1)
        s2, m2 = self.lif2(self.conv2(s1), m2)
        s3, m3 = self.lif3(self.fc(s2.flatten(1)), m3)
        r = self.alpha * r + (1.0 - self.alpha) * s3
        out = self.read(r)
        cells = out[:, :2 * self.n_cells].reshape(-1, 2, self.n_cells)
        return cells, out[:, -1], (m1, m2, m3, r), s3.mean()      # the dense layer's rate: conv activity is sparse by nature


# --- the localiser --------------------------------------------------------------------

class SpikingLocaliser:
    """model.Localiser: `locate(frame)` -> (x, y) normalised, or (nan, nan) when the
    network says the target is not there. `fit` trains on FrameSets."""

    def __init__(self, dt_s: float = 0.005, n_cells: int = 32, ch=(8, 16), hidden: int = 128,
                 present_threshold: float = 0.5, rate_target: float = 0.1, rate_weight: float = 10.0,
                 in_hw=(60, 80), resolution=(640, 480), seed: int = 0, learn_tau: bool = False,
                 polarity: str = "both", device=None):
        self.device = torch.device(device or "cpu")
        if self.device.type == "cpu":
            torch.set_num_threads(min(4, torch.get_num_threads()))
            _prefer_performance_cores()
        self.dt_s, self.n_cells, self.ch, self.hidden = dt_s, n_cells, tuple(ch), hidden
        self.present_threshold, self.rate_target, self.rate_weight = present_threshold, rate_target, rate_weight
        self.in_hw, self.resolution, self.seed, self.learn_tau = tuple(in_hw), tuple(resolution), seed, learn_tau
        self.polarity = polarity
        self.cells = PlaceCells(n_cells)
        self.net = SpikingLocaliserNet(n_cells, ch, hidden, dt_s, in_hw=in_hw, seed=seed, learn_tau=learn_tau,
                                       polarity=polarity).to(self.device)
        self.val_clips: list[str] | None = None          # names of the clips held out while fitting
        self.reset()

    # -- persistence --

    def config(self) -> dict:
        return {k: getattr(self, k) for k in ("dt_s", "n_cells", "ch", "hidden", "present_threshold", "rate_target",
                                              "rate_weight", "in_hw", "resolution", "seed", "learn_tau", "polarity")}

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"arch": "localiser", "config": self.config(), "state": self.net.state_dict(),
                    "val_clips": self.val_clips}, path)
        return path

    @classmethod
    def load(cls, path, device=None) -> "SpikingLocaliser":
        ck = torch.load(Path(path), map_location="cpu", weights_only=False)
        m = cls(device=device, **{"learn_tau": True, **ck["config"]})       # older checkpoints learned them
        m.net.load_state_dict(ck["state"])
        m.val_clips = ck.get("val_clips")
        m.reset()
        return m

    # -- streaming --

    def reset(self) -> None:
        self.net.eval()
        self.state = None
        self.last_present = 0.0

    def locate(self, obs) -> tuple[float, float]:
        frame = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)[None]
        if self.state is None:
            self.state = self.net.init_state(1, self.device)
        with torch.no_grad():
            cells, present, self.state, _ = self.net.step(frame, self.state)
        self.last_present = float(torch.sigmoid(present[0]))
        if self.last_present < self.present_threshold:
            return (np.nan, np.nan)
        x, y = self.cells.decode(cells[0]).cpu().numpy()
        return (float(x), float(y))

    # -- training --

    def _loss(self, cells, present_logit, gt, present, rate):
        """Cross-entropy of each axis's cells against the truth's bump (present steps),
        binary cross-entropy on `present` (steps with a defined truth), rate regulariser."""
        target = self.cells.bumps(gt)                                           # (T, B, 2, n)
        target = target / target.sum(-1, keepdim=True).clamp_min(1e-6)
        ce = -(target * torch.log_softmax(cells, dim=-1)).sum(-1).mean(-1)      # (T, B)
        known = torch.isfinite(gt).all(-1)
        lp = ce[present & known]
        lb = nn.functional.binary_cross_entropy_with_logits(present_logit[known], present[known].float()) \
            if known.any() else present_logit.sum() * 0
        loss_p = lp.mean() if lp.numel() else cells.sum() * 0
        return loss_p + lb + self.rate_weight * (rate - self.rate_target) ** 2, loss_p.item(), lb.item()

    def _run_chunks(self, sets, chunk: int, rng, optimiser=None, augment: bool = True, augment_kw=None):
        """One batch of frame sets (or paths to them, loaded here so a corpus need not sit
        in memory), chunk by chunk with the state carried. Returns the mean losses (total,
        position, present) and per-step (px error on present frames, present hit)."""
        sets = [fs if isinstance(fs, FrameSet) else load_frame_set(fs, round(self.dt_s * 1e6), 8) for fs in sets]
        n = min(len(fs.t) for fs in sets)
        fr, gt, pr = [], [], []
        for fs in sets:
            f, g, p = (augment_frames(fs.frames[:n], fs.gt[:n], fs.present[:n], rng, **(augment_kw or {})) if augment and optimiser is not None
                       else (fs.frames[:n].astype(np.float32), fs.gt[:n], fs.present[:n]))
            fr.append(f); gt.append(g); pr.append(p)
        frames = torch.tensor(np.stack(fr, axis=1))                                          # (T, B, 2, H, W)
        gt_t = torch.tensor(np.stack(gt, axis=1), dtype=torch.float32, device=self.device)   # (T, B, 2)
        pr_t = torch.tensor(np.stack(pr, axis=1), device=self.device)                        # (T, B)
        state, losses, errs, hits = None, [], [], []
        for s in range(0, n, chunk):
            sl = slice(s, s + chunk)
            x = frames[sl].to(self.device)
            with torch.set_grad_enabled(optimiser is not None):
                if state is None:
                    state = self.net.init_state(x.shape[1], self.device)
                cells, pres, rate = [], [], 0.0
                for i in range(x.shape[0]):
                    c, p, state, r = self.net.step(x[i], state)
                    cells.append(c); pres.append(p); rate = rate + r
                cells, pres = torch.stack(cells), torch.stack(pres)
                loss, lp, lb = self._loss(cells, pres, gt_t[sl], pr_t[sl], rate / x.shape[0])
            if optimiser is not None:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimiser.step()
            state = tuple(v.detach() for v in state)
            losses.append((loss.item(), lp, lb))
            with torch.no_grad():
                pos = self.cells.decode(cells.detach())                                     # (T, B, 2)
                px = torch.hypot(*((pos - gt_t[sl]) * torch.tensor(self.resolution, device=self.device)).unbind(-1))
                m = pr_t[sl] & torch.isfinite(gt_t[sl]).all(-1)
                errs.append(px[m].cpu().numpy())
                hits.append(((torch.sigmoid(pres) >= self.present_threshold) == pr_t[sl])[torch.isfinite(gt_t[sl]).all(-1)].cpu().numpy())
        return np.mean(losses, axis=0), np.concatenate(errs), np.concatenate(hits)

    def fit(self, sets: list[FrameSet], val_sets: list[FrameSet] | None = None, epochs: int = 20,
            chunk_s: float = 0.25, batch: int = 8, lr: float = 1e-2, augment: bool = True, augment_kw=None,
            seed: int = 0, schedule: str = "cosine", log_fn=None) -> list[dict]:
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        chunk = max(1, round(chunk_s / self.dt_s))
        first = sets[0] if isinstance(sets[0], FrameSet) else load_frame_set(sets[0], round(self.dt_s * 1e6), 8)
        self.net.calibrate(torch.tensor(first.frames[:400].astype(np.float32), device=self.device))
        optimiser = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs) if schedule == "cosine" else None
        log, best = [], (np.inf, None)

        def validate(epoch, train_loss):
            self.net.eval()
            if val_sets:
                loss, errs, hits = self._run_chunks(val_sets, chunk, rng, augment=False)
                entry = {"epoch": epoch, "train_loss": train_loss, "val_loss": loss[0], "val_pos": loss[1],
                         "val_present": loss[2], "val_px": float(np.median(errs)) if errs.size else np.nan,
                         "val_present_acc": float(hits.mean()) if hits.size else np.nan}
            else:
                entry = {"epoch": epoch, "train_loss": train_loss, "val_loss": np.nan, "val_pos": np.nan,
                         "val_present": np.nan, "val_px": np.nan, "val_present_acc": np.nan}
            log.append(entry)
            if log_fn:
                log_fn(entry)
            return entry["val_loss"]

        validate(0, np.nan)
        for epoch in range(1, epochs + 1):
            self.net.train()
            losses = []
            order = rng.permutation(len(sets))
            for b in range(0, len(sets), batch):
                group = [sets[i] for i in order[b:b + batch]]
                losses.append(self._run_chunks(group, chunk, rng, optimiser, augment, augment_kw)[0][0])
            if scheduler is not None:
                scheduler.step()
            val_loss = validate(epoch, float(np.mean(losses)))
            score = val_loss if np.isfinite(val_loss) else -epoch
            if score < best[0]:
                best = (score, {k: v.detach().clone() for k, v in self.net.state_dict().items()})
        if best[1] is not None:
            self.net.load_state_dict(best[1])
        self.reset()
        return log
