"""The framework-agnostic boundary: Localiser + TrajectoryMemory Protocols.

The Stage-1 memory is `snn.SpikingMemory`; the baselines are in `baseline.py`. The
reservoir and localiser below are placeholders.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Localiser(Protocol):
    def locate(self, obs) -> tuple[float, float]: ...
    def reset(self) -> None: ...


@runtime_checkable
class TrajectoryMemory(Protocol):
    def fit(self, tracks) -> None: ...                   # offline pretrain on many position tracks
    def observe(self, x: float, y: float) -> None: ...   # stream one step
    def predict(self, horizon_s: float) -> tuple[float, float]: ...
    def deviation_score(self) -> float: ...              # 0 = on-pattern, high = break
    def period(self) -> float: ...                       # the remembered period, s; NaN if none yet
    def path_points(self, fractions): ...                # the remembered path at these fractions of its cycle, (N, 2); None if none yet
    def reset(self) -> None: ...


class CentroidLocaliser:
    """Wraps frontend.to_position."""

    def locate(self, obs) -> tuple[float, float]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class ReservoirMemory:
    """NumPy echo-state reservoir + ridge/RLS readout. Provisional until G-F closes."""

    def __init__(self, **params):
        self.params = params

    def fit(self, tracks) -> None:
        raise NotImplementedError

    def observe(self, x: float, y: float) -> None:
        raise NotImplementedError

    def predict(self, horizon_s: float) -> tuple[float, float]:
        raise NotImplementedError

    def deviation_score(self) -> float:
        raise NotImplementedError

    def period(self) -> float:
        raise NotImplementedError

    def path_points(self, fractions):
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class Model:
    """Localiser + TrajectoryMemory."""

    def __init__(self, localiser: Localiser, memory: TrajectoryMemory):
        self.localiser = localiser
        self.memory = memory

    def step(self, obs) -> tuple[tuple[float, float], float]:
        raise NotImplementedError
