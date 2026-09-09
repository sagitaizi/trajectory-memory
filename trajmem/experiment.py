"""Harness: build corpus -> pretrain on sim -> freeze -> evaluate on real -> results table.

Deterministic given a seed.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Report:
    table: dict = field(default_factory=dict)
    plots: list = field(default_factory=list)
    config: dict = field(default_factory=dict)


def run(config) -> Report:
    raise NotImplementedError
