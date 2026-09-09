"""v2e wrapper: TrajectorySpec + camera config -> synthetic event stream + exact ground truth.

v2e is a source clone at D:\\Projects\\v2e used via sys.path (see materials/03-tooling).
The EventEmulator Python API is the integration point; construct it with device="cpu".
"""
from __future__ import annotations

import numpy as np

from .data import Clip
from .trajectories import TrajectorySpec


def render_frames(spec: TrajectorySpec, cfg) -> np.ndarray:
    """Sampled path -> high-rate frame stack (T, H, W) float32, blob drawn in distorted px."""
    raise NotImplementedError


def _run_v2e(frames: np.ndarray, cfg) -> np.ndarray:
    """Frame stack -> (N, 4) [t, x, y, polarity] via v2ecore.emulator.EventEmulator."""
    raise NotImplementedError


def simulate(spec: TrajectorySpec, camera_cfg, sim_cfg, seed: int = 0) -> Clip:
    raise NotImplementedError
